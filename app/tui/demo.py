"""Reproducible, live TUI recording for ``lls demo-tui`` (VHS + FFmpeg)."""

from __future__ import annotations

import contextlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import httpx

from app.logger import logger
from app.tools import check_path_safe

from .palette import load_palette


class DemoError(Exception):
    """An actionable recording failure, without replacing the previous demo."""


@dataclass(frozen=True)
class DemoResult:
    video: Path
    poster: Path
    preview: Path
    duration: float
    size: int


def tape_quote(value: str) -> str:
    """VHS strings use a literal delimiter, not shell/JSON backslash escapes."""
    if not any(char in value for char in "\r\n\x00"):
        for delimiter in ('"', "'", "`"):
            if delimiter not in value:
                return f"{delimiter}{value}{delimiter}"
    raise DemoError("Recording paths cannot contain newlines or all three quote characters.")


def render_tape(video: Path, poster: Path, palette: Path | None) -> str:
    # The shell-local function uses the invoking installation, even when a different
    # lls is first on PATH. It does not change the user's shell, aliases or history.
    setup = (
        "unset HISTFILE PROMPT_COMMAND; PS1='$ '; "
        f'lls() {{ {shlex.quote(sys.executable)} -m app.cli "$@"; }}; clear'
    )
    command = "lls tui" + (f" --palette {shlex.quote(str(palette))}" if palette else "")
    replacements = {
        "@VIDEO@": str(video),
        "@POSTER@": str(poster),
        "@SETUP@": setup,
        "@COMMAND@": command,
    }
    tape = files("app.tui").joinpath("demo.tape").read_text(encoding="utf-8")
    for token, value in replacements.items():
        tape = tape.replace(token, tape_quote(value))
    return tape


def run_tool(command: list[str], *, env: dict[str, str], timeout: int = 300) -> str:
    """Bound subprocess lifetime, including recorder children on cancellation."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    if process.returncode:
        raise DemoError(f"{Path(command[0]).name} failed:\n{output[-4000:]}")
    return output


def request(client: httpx.Client, method: str, path: str, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


@contextlib.contextmanager
def demo_conversation(client: httpx.Client):
    """Use only our temporary chat; restore the previously active conversation."""
    conversations = request(client, "GET", "/api/chat/conversations")["conversations"]
    previous = next((item["id"] for item in conversations if item.get("is_active")), None)
    demo_id = request(client, "POST", "/api/chat/new")["id"]
    try:
        request(
            client,
            "POST",
            f"/api/chat/rename/{demo_id}",
            json={"title": "Local AI. Your terminal."},
        )
        yield
    finally:
        try:
            current = request(client, "GET", "/api/chat/conversations")["conversations"]
            active = next((item["id"] for item in current if item.get("is_active")), None)
            # Do not undo a concurrent switch made by the user in another client.
            if active == demo_id and previous and any(c["id"] == previous for c in current):
                request(client, "POST", f"/api/chat/switch/{previous}")
            request(client, "DELETE", f"/api/chat/{demo_id}")
        except httpx.HTTPError as exc:
            logger.warning("Could not clean up demo conversation %s: %s", demo_id, exc)


def validate_video(path: Path, env: dict[str, str]) -> float:
    if not path.is_file() or path.stat().st_size == 0:
        raise DemoError("The recorder produced no MP4. VHS 0.11.0 is the tested version.")
    try:
        data = json.loads(
            run_tool(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    str(path),
                ],
                env=env,
                timeout=30,
            )
        )
        stream = next(s for s in data["streams"] if s["codec_type"] == "video")
        duration = float(data["format"]["duration"])
        valid = (
            stream["codec_name"] == "h264"
            and stream["pix_fmt"] == "yuv420p"
            and (stream["width"], stream["height"]) == (1920, 1080)
            and math.isfinite(duration)
            and duration >= 15
        )
    except (ValueError, KeyError, StopIteration, TypeError) as exc:
        raise DemoError("FFprobe could not validate the recorded video.") from exc
    if not valid:
        raise DemoError("Expected a complete 1920x1080 H.264/yuv420p tour (at least 15 seconds).")
    run_tool(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"], env=env)
    return duration


def make_preview(video: Path, preview: Path, env: dict[str, str]) -> None:
    """A README-compatible loop; the linked MP4 retains full resolution."""
    run_tool(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            "fps=3,scale=960:-2:flags=lanczos,split[v][p];"
            "[p]palettegen=max_colors=96:reserve_transparent=0[pal];"
            "[v][pal]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
            "-gifflags",
            "-transdiff",
            "-loop",
            "0",
            str(preview),
        ],
        env=env,
    )
    if not preview.is_file() or not 0 < preview.stat().st_size < 10_000_000:
        raise DemoError("README GIF must be below 10,000,000 bytes; previous media is unchanged.")
    run_tool(["ffmpeg", "-v", "error", "-xerror", "-i", str(preview), "-f", "null", "-"], env=env)


def record_demo(base_url: str, output: str, palette: str | None = None) -> DemoResult:
    """Record real keystrokes and live data, then atomically publish verified media."""
    target = check_path_safe(output)
    if target.suffix.lower() != ".mp4":
        raise DemoError("Use an .mp4 output filename.")
    poster = check_path_safe(str(target.with_name(f"{target.stem}-poster.png")))
    preview = check_path_safe(str(target.with_suffix(".gif")))
    load_palette(palette)  # Fail before any chat state or output is touched.
    palette_path = check_path_safe(palette) if palette else None
    missing = [
        name for name in ("vhs", "ffmpeg", "ffprobe", "ttyd", "bash") if not shutil.which(name)
    ]
    if missing:
        raise DemoError(
            f"Missing recording tools: {', '.join(missing)}. See README: Recording the demo."
        )
    env = os.environ.copy()
    env.pop("NO_COLOR", None)
    env.update(TERM="xterm-256color", COLORTERM="truecolor")
    try:
        with httpx.Client(base_url=base_url, timeout=10) as client:
            status = request(client, "GET", "/api/server/status")
            if (
                not status.get("running")
                or not status.get("current_model")
                or status.get("is_loading")
            ):
                raise DemoError(
                    "A model must already be loaded. Check lls status before recording."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            # Same filesystem as the destination: replacements below remain atomic.
            with tempfile.TemporaryDirectory(prefix=".tui-demo-", dir=target.parent) as directory:
                temporary = check_path_safe(directory)
                env["TMPDIR"] = str(temporary)  # VHS frames/browser scratch stay in the workspace.
                raw = temporary / "capture.mp4"
                staged = temporary / "demo.mp4"
                thumbnail = temporary / "poster.png"
                animation = temporary / "preview.gif"
                tape = temporary / "demo.tape"
                tape.write_text(render_tape(raw, thumbnail, palette_path), encoding="utf-8")
                run_tool(["vhs", "validate", str(tape)], env=env, timeout=30)
                with demo_conversation(client):
                    run_tool(["vhs", str(tape)], env=env)
                if not raw.is_file() or not raw.stat().st_size:
                    raise DemoError("VHS produced no video. Try the tested VHS 0.11.0 release.")
                run_tool(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(raw),
                        "-an",
                        "-c:v",
                        "libx264",
                        "-crf",
                        "18",
                        "-preset",
                        "medium",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(staged),
                    ],
                    env=env,
                )
                duration = validate_video(staged, env)
                if not thumbnail.is_file() or thumbnail.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                    raise DemoError(
                        "VHS did not produce the README poster; previous media is unchanged."
                    )
                make_preview(staged, animation, env)
                staged.replace(target)
                thumbnail.replace(poster)
                animation.replace(preview)
    except (httpx.HTTPError, OSError, subprocess.TimeoutExpired) as exc:
        raise DemoError(f"Recording failed: {exc}") from exc
    return DemoResult(target, poster, preview, duration, target.stat().st_size)
