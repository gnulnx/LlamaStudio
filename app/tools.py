"""
Safe Tool Executive for LLamaStudio.
Defines OpenAI-compatible schemas and implements execution for workspace-sandboxed tools.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config_store import config_loader
from .logger import logger

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_AUDIO_SECONDS = 10 * 60
MODEL_AUDIO_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class ToolResult:
    """A tool result plus non-text media to include in the next model turn."""

    content: str
    images: list[dict] = field(default_factory=list)
    audios: list[dict] = field(default_factory=list)


def _detect_image_mime(header: bytes) -> str | None:
    """Return a supported raster MIME type based on file signature."""
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"BM"):
        return "image/bmp"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def _detect_audio_mime(header: bytes) -> str | None:
    """Return a supported audio MIME type based on file signature."""
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "audio/wav"
    if header.startswith(b"fLaC"):
        return "audio/flac"
    if header.startswith(b"ID3") or (
        len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
    ):
        return "audio/mpeg"
    return None


def prepare_audio_for_model(audio_bytes: bytes, mime_type: str) -> tuple[bytes, str, str]:
    """Return llama.cpp-compatible WAV/MP3 bytes, MIME type, and input_audio format."""
    if mime_type == "audio/wav":
        return audio_bytes, mime_type, "wav"
    if mime_type == "audio/mpeg":
        return audio_bytes, mime_type, "mp3"
    if mime_type != "audio/flac":
        raise ValueError("Only WAV, MP3, and FLAC audio files are supported.")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("FLAC input requires ffmpeg so it can be converted to WAV for llama.cpp.")

    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-t",
                str(MAX_AUDIO_SECONDS),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(MODEL_AUDIO_SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                "pipe:1",
            ],
            input=audio_bytes,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("FLAC conversion timed out.") from exc

    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        logger.warning("[tools] FLAC conversion failed: %s", detail)
        raise ValueError("The FLAC file could not be decoded.")

    maximum_pcm_bytes = MAX_AUDIO_SECONDS * MODEL_AUDIO_SAMPLE_RATE * 2
    if len(completed.stdout) >= maximum_pcm_bytes:
        raise ValueError(f"Audio attachments must be shorter than {MAX_AUDIO_SECONDS} seconds.")
    return completed.stdout, "audio/wav", "wav"


def check_path_safe(file_path: str) -> Path:
    """Resolve file path and guarantee it remains strictly within the workspace root unless sandboxing is disabled."""
    target = Path(file_path)
    if config_loader.sandbox_disabled():
        return target.resolve()

    workspace_root = Path(config_loader.get_workspace_root()).resolve()
    # If relative, resolve against workspace root
    if not target.is_absolute():
        target = workspace_root / target

    target = target.resolve()

    # Check if target is indeed inside workspace_root
    if not str(target).startswith(str(workspace_root)):
        raise ValueError(
            f"Permission Denied: Target path '{file_path}' lies outside the workspace directory."
        )
    return target


def write_file(file_path: str, content: str) -> str:
    """Create or overwrite a file in the workspace directory with the specified content."""
    try:
        safe_path = check_path_safe(file_path)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        size = safe_path.stat().st_size
        return f"Successfully wrote {size} bytes to '{file_path}'"
    except Exception as e:
        logger.error(f"[tools] Error in write_file: {e}")
        return f"Error executing write_file: {e}"


def read_file(file_path: str) -> str | ToolResult:
    """Read text, or return supported images/audio as token-safe multimodal content."""
    try:
        safe_path = check_path_safe(file_path)
        if not safe_path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not safe_path.is_file():
            return f"Error: '{file_path}' is a directory, not a file."

        with open(safe_path, "rb") as media_file:
            header = media_file.read(16)
            image_mime_type = _detect_image_mime(header)
            if image_mime_type:
                size = safe_path.stat().st_size
                if size > MAX_IMAGE_BYTES:
                    return (
                        f"Error: Image '{file_path}' is {size} bytes; the maximum supported "
                        f"image size is {MAX_IMAGE_BYTES} bytes."
                    )
                media_file.seek(0)
                encoded = base64.b64encode(media_file.read()).decode("ascii")
                return ToolResult(
                    content=(f"Loaded image '{file_path}' ({size} bytes) as multimodal input."),
                    images=[
                        {
                            "name": safe_path.name,
                            "mime_type": image_mime_type,
                            "data_url": f"data:{image_mime_type};base64,{encoded}",
                            "size": size,
                        }
                    ],
                )

            audio_mime_type = _detect_audio_mime(header)
            if audio_mime_type:
                size = safe_path.stat().st_size
                if size > MAX_AUDIO_BYTES:
                    return (
                        f"Error: Audio '{file_path}' is {size} bytes; the maximum supported "
                        f"audio size is {MAX_AUDIO_BYTES} bytes."
                    )
                media_file.seek(0)
                encoded = base64.b64encode(media_file.read()).decode("ascii")
                return ToolResult(
                    content=(f"Loaded audio '{file_path}' ({size} bytes) as multimodal input."),
                    audios=[
                        {
                            "name": safe_path.name,
                            "mime_type": audio_mime_type,
                            "data_url": f"data:{audio_mime_type};base64,{encoded}",
                            "size": size,
                        }
                    ],
                )

        with open(safe_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"[tools] Error in read_file: {e}")
        return f"Error executing read_file: {e}"


def list_dir(dir_path: str = ".") -> str:
    """List the names and types of files inside the specified workspace subdirectory."""
    try:
        safe_path = check_path_safe(dir_path)
        if not safe_path.exists():
            return f"Error: Directory '{dir_path}' does not exist."
        if not safe_path.is_dir():
            return f"Error: '{dir_path}' is a file, not a directory."

        workspace_root = Path(config_loader.get_workspace_root()).resolve()
        entries = []
        for p in safe_path.iterdir():
            try:
                rel = p.relative_to(workspace_root)
            except ValueError:
                rel = p.name
            suffix = "/" if p.is_dir() else ""
            size = f" ({p.stat().st_size} bytes)" if p.is_file() else ""
            entries.append(f"{rel}{suffix}{size}")

        if not entries:
            return f"Directory '{dir_path}' is empty."
        return "\n".join(sorted(entries))
    except Exception as e:
        logger.error(f"[tools] Error in list_dir: {e}")
        return f"Error executing list_dir: {e}"


def get_absolute_path(file_path: str = ".") -> str:
    """Return the absolute system path of a file or directory in the workspace directory."""
    try:
        safe_path = check_path_safe(file_path)
        return str(safe_path)
    except Exception as e:
        logger.error(f"[tools] Error in get_absolute_path: {e}")
        return f"Error executing get_absolute_path: {e}"


def run_command(command: str) -> str:
    """Execute a shell command inside the workspace root directory with a 15-second safety timeout."""
    try:
        logger.info(f"[tools] Executing command: {command}")
        workspace_root = Path(config_loader.get_workspace_root()).resolve()
        res = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=15.0,
        )
        output = []
        if res.stdout:
            output.append(f"--- Standard Output ---\n{res.stdout}")
        if res.stderr:
            output.append(f"--- Standard Error ---\n{res.stderr}")

        result_text = "\n".join(output) if output else "Command completed with no output."
        return f"Command returned exit code {res.returncode}\n{result_text}"
    except subprocess.TimeoutExpired:
        logger.warning(f"[tools] Command timed out: {command}")
        return "Error: Command timed out after 15 seconds."
    except Exception as e:
        logger.error(f"[tools] Error in run_command: {e}")
        return f"Error executing command: {e}"


# OpenAI compatible tool specifications
ALL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file in the workspace directory. Use this to create new files or completely overwrite existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write, relative to the workspace directory (e.g. 'HelloFromMe.txt').",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete content to write to the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file or load supported image/audio media as multimodal input from inside the workspace directory. Images: PNG, JPEG, WebP, GIF, BMP. Audio: WAV, MP3, FLAC.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to read, relative to the workspace directory.",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List all files and subdirectories inside a specific workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "The directory path to scan, relative to the workspace (defaults to '.').",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command locally in the workspace directory. Use this with care only when needed (e.g. running build scripts, checking git statuses).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_absolute_path",
            "description": "Return the absolute system path of a file or directory in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to resolve (relative to workspace). Defaults to '.' for workspace root.",
                    }
                },
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str | ToolResult:
    """Central tool dispatcher."""
    if name == "write_file":
        return write_file(arguments.get("file_path"), arguments.get("content"))
    elif name == "read_file":
        return read_file(arguments.get("file_path"))
    elif name == "list_dir":
        return list_dir(arguments.get("dir_path", "."))
    elif name == "run_command":
        return run_command(arguments.get("command"))
    elif name == "get_absolute_path":
        fp = arguments.get("file_path") or arguments.get("filename", ".")
        return get_absolute_path(fp)
    else:
        return f"Error: Tool '{name}' is not recognized."
