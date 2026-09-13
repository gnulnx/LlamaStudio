"""Recorder safety/packaging contracts; no recorder, GPU or live backend needed."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner

from app.cli import cli
from app.tui.demo import (
    DemoError,
    DemoResult,
    demo_conversation,
    make_preview,
    record_demo,
    render_tape,
    run_tool,
    tape_quote,
    validate_video,
)


def test_packaged_tape_uses_live_cli_and_all_sections():
    tape = render_tape(Path("/workspace/clip.mp4"), Path("/workspace/poster.png"), None)
    assert 'Type@120ms "lls tui"' in tape
    assert "LOAD SETTINGS" in tape
    assert "Saved conversation . ready" in tape
    assert "tokens.s|Response complete" in tape
    assert "Following" in tape
    assert "Ctrl+Q" in tape
    assert "@VIDEO@" not in tape
    assert "@POSTER@" not in tape
    assert "-m app.cli" in tape
    assert "unset HISTFILE PROMPT_COMMAND" in tape


def test_palette_paths_are_quoted_for_both_shell_and_vhs():
    tape = render_tape(Path("/workspace/a b.mp4"), Path("poster.png"), Path("/workspace/a b.json"))
    assert 'Output "/workspace/a b.mp4"' in tape
    assert "Type@120ms \"lls tui --palette '/workspace/a b.json'\"" in tape
    for invalid in ("x\nEnter", "x\rEnter", "x\0", "\"'`"):
        with pytest.raises(DemoError):
            tape_quote(invalid)


class ChatBackend:
    def __init__(self):
        self.active = "original"
        self.calls = []
        self.fail_rename = False

    def __call__(self, request):
        path = request.url.path
        self.calls.append((request.method, path))
        data = {}
        if path == "/api/chat/conversations":
            data = {
                "conversations": [
                    {"id": item, "is_active": self.active == item}
                    for item in ("original", "demo", "other")
                ]
            }
        elif path == "/api/chat/new":
            self.active = "demo"
            data = {"id": "demo"}
        elif path == "/api/chat/switch/original":
            self.active = "original"
        elif path == "/api/chat/rename/demo" and self.fail_rename:
            return httpx.Response(503)
        return httpx.Response(200, json=data)


@pytest.mark.parametrize("failure", [False, True])
def test_only_demo_chat_is_deleted_and_original_restored_on_success_or_failure(failure):
    backend = ChatBackend()
    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(backend)) as client:
        try:
            with demo_conversation(client):
                assert backend.active == "demo"
                if failure:
                    raise DemoError("recording failed")
        except DemoError:
            assert failure
    assert backend.active == "original"
    assert backend.calls[-2:] == [
        ("POST", "/api/chat/switch/original"),
        ("DELETE", "/api/chat/demo"),
    ]
    assert all("model" not in path for _, path in backend.calls)


def test_concurrent_user_switch_is_not_undone():
    backend = ChatBackend()
    with (
        httpx.Client(base_url="http://test", transport=httpx.MockTransport(backend)) as client,
        demo_conversation(client),
    ):
        backend.active = "other"
    assert backend.active == "other"
    assert ("POST", "/api/chat/switch/original") not in backend.calls
    assert backend.calls[-1] == ("DELETE", "/api/chat/demo")


def test_rename_failure_still_cleans_up():
    backend = ChatBackend()
    backend.fail_rename = True
    with (
        httpx.Client(base_url="http://test", transport=httpx.MockTransport(backend)) as client,
        pytest.raises(httpx.HTTPStatusError),
        demo_conversation(client),
    ):
        pytest.fail("Rename must fail before recording")
    assert backend.active == "original"
    assert backend.calls[-1] == ("DELETE", "/api/chat/demo")


def test_missing_dependencies_fail_before_backend_or_output_changes(tmp_path):
    target = tmp_path / "demo.mp4"
    target.write_bytes(b"previous")
    with (
        patch("app.tui.demo.check_path_safe", side_effect=lambda p: Path(p)),
        patch("app.tui.demo.shutil.which", return_value=None),
        patch("app.tui.demo.httpx.Client") as client,
        pytest.raises(DemoError, match="Missing recording tools"),
    ):
        record_demo("http://test", str(target))
    assert target.read_bytes() == b"previous"
    client.assert_not_called()


@pytest.mark.parametrize(
    "state",
    [
        {"running": False},
        {"running": True, "current_model": "model", "is_loading": True},
    ],
)
def test_requires_loaded_model_without_mutation(tmp_path, state):
    with (
        patch("app.tui.demo.check_path_safe", side_effect=lambda p: Path(p)),
        patch("app.tui.demo.shutil.which", return_value="tool"),
        patch("app.tui.demo.request", return_value=state) as request,
        pytest.raises(DemoError, match="already be loaded"),
    ):
        record_demo("http://test", str(tmp_path / "demo.mp4"))
    assert request.call_count == 1
    assert request.call_args.args[1:] == ("GET", "/api/server/status")


def test_failed_capture_preserves_existing_media_and_child_env(tmp_path):
    target, poster = tmp_path / "demo.mp4", tmp_path / "demo-poster.png"
    target.write_bytes(b"previous video")
    poster.write_bytes(b"previous poster")
    with (
        patch("app.tui.demo.check_path_safe", side_effect=lambda p: Path(p)),
        patch("app.tui.demo.shutil.which", return_value="tool"),
        patch("app.tui.demo.request", return_value={"running": True, "current_model": "model"}),
        patch("app.tui.demo.demo_conversation") as conversation,
        patch("app.tui.demo.run_tool", side_effect=["valid", DemoError("capture failed")]) as run,
        patch.dict("os.environ", {"NO_COLOR": "1"}),
        pytest.raises(DemoError, match="capture failed"),
    ):
        record_demo("http://test", str(target))
    assert "NO_COLOR" not in run.call_args.kwargs["env"]
    assert run.call_args.kwargs["env"]["COLORTERM"] == "truecolor"
    conversation.return_value.__exit__.assert_called_once()
    assert target.read_bytes() == b"previous video"
    assert poster.read_bytes() == b"previous poster"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["demo-poster.png", "demo.mp4"]


def test_missing_video_is_not_mistaken_for_success(tmp_path):
    with pytest.raises(DemoError, match="no MP4"):
        validate_video(tmp_path / "absent.mp4", {})


@pytest.mark.parametrize("duration", ["nan", "0", "12", "45"])
def test_video_metadata_and_full_decode_are_checked(tmp_path, duration):
    target = tmp_path / "demo.mp4"
    target.write_bytes(b"fixture")
    metadata = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 1920,
                "height": 1080,
            }
        ],
        "format": {"duration": duration},
    }
    with patch("app.tui.demo.run_tool", return_value=json.dumps(metadata)) as run:
        if duration == "45":
            assert validate_video(target, {}) == 45
            assert run.call_count == 2
            assert "-xerror" in run.call_args.args[0]
        else:
            with pytest.raises(DemoError, match="complete"):
                validate_video(target, {})


def test_timeout_terminates_only_recorder_process_group():
    with (
        patch("app.tui.demo.subprocess.Popen") as spawn,
        patch("app.tui.demo.os.killpg") as kill,
    ):
        process = spawn.return_value
        process.pid = 54321
        process.communicate.side_effect = [subprocess.TimeoutExpired("vhs", 1), ("", None)]
        with pytest.raises(subprocess.TimeoutExpired):
            run_tool(["vhs", "demo.tape"], env={}, timeout=1)
    assert spawn.call_args.kwargs["start_new_session"]
    assert kill.call_args.args[0] == 54321


def test_cli_forwards_output_and_palette_without_starting_backend():
    result = DemoResult(Path("demo.mp4"), Path("demo-poster.png"), Path("demo.gif"), 45, 1_000_000)
    with (
        patch("app.cli.record_demo", return_value=result) as record,
        patch("app.cli.start_server_background") as start,
    ):
        response = CliRunner().invoke(
            cli, ["demo-tui", "--output", "demo.mp4", "--palette", "colors.json"]
        )
    assert response.exit_code == 0, response.output
    assert record.call_args.args[1:] == ("demo.mp4", "colors.json")
    assert "45.0s" in response.output
    start.assert_not_called()


@pytest.mark.parametrize("size", [0, 9_999_999, 10_000_000])
def test_preview_enforces_decimal_github_limit(tmp_path, size):
    preview = tmp_path / "demo.gif"
    preview.write_bytes(b"x" * size)
    with patch("app.tui.demo.run_tool") as run:
        if size == 9_999_999:
            make_preview(tmp_path / "demo.mp4", preview, {})
            assert run.call_count == 2
        else:
            with pytest.raises(DemoError, match="10,000,000"):
                make_preview(tmp_path / "demo.mp4", preview, {})
