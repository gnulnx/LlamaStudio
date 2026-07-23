import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.cli import cli, load, load_saved_model_settings, oneshot, select_launch_view_for_cli, start
from app.tools import ToolResult


class FakeLoadResponse:
    status_code = 200
    text = "ok"

    def json(self):
        return {"text": "Hello from the microphone."}


class FakeStatusResponse:
    def __init__(self, running):
        self.running = running

    def json(self):
        return {"running": self.running}


class FakeChatStreamResponse:
    status_code = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self):
        yield 'data: {"type":"start"}'
        yield 'data: {"content":"A bicycle."}'
        yield 'data: {"type":"metrics","metrics":{"prompt_tokens":19,"completion_tokens":7,"total_tokens":26,"elapsed_seconds":0.21,"tokens_per_second":33.3}}'
        yield 'data: {"type":"end"}'


class TestCliLoadSettings(unittest.TestCase):
    def test_load_saved_model_settings_matches_equivalent_resolved_path(self):
        with patch(
            "app.cli.config_loader.get_model_profile_settings",
            return_value={"ctx_size": 128000, "gpu_layers": 999},
        ):
            loaded_settings = load_saved_model_settings("/models/model.gguf")

        self.assertEqual(loaded_settings["ctx_size"], 128000)
        self.assertEqual(loaded_settings["gpu_layers"], 999)

    def test_load_uses_saved_model_settings_and_applies_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "Qwen3.6-27B-Q8_0.gguf"
            model_path.touch()
            scanned_model = SimpleNamespace(
                name="Qwen3.6-27B-Q8_0",
                path=str(model_path),
                size_human="28 GB",
            )

            with (
                patch(
                    "app.cli.config_loader.get_model_profile_settings",
                    return_value={
                        "ctx_size": 128000,
                        "gpu_layers": 999,
                        "kv_cache_type": "q8_0",
                    },
                ),
                patch("app.cli.is_server_online", return_value=True),
                patch("app.model_manager.scan_models", return_value=[scanned_model]),
                patch("app.cli.httpx.post", return_value=FakeLoadResponse()) as mock_post,
            ):
                result = CliRunner().invoke(load, ["Qwen3.6", "--ctx-size", "64000"])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["settings"]["ctx_size"], 64000)
        self.assertEqual(payload["settings"]["gpu_layers"], 999)
        self.assertEqual(payload["settings"]["kv_cache_type"], "q8_0")

    def test_load_can_ignore_saved_model_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "Qwen3.6-27B-Q8_0.gguf"
            model_path.touch()
            scanned_model = SimpleNamespace(
                name="Qwen3.6-27B-Q8_0",
                path=str(model_path),
                size_human="28 GB",
            )

            with (
                patch(
                    "app.cli.config_loader.get_model_profile_settings",
                    return_value={"ctx_size": 128000},
                ),
                patch("app.cli.is_server_online", return_value=True),
                patch("app.model_manager.scan_models", return_value=[scanned_model]),
                patch("app.cli.httpx.post", return_value=FakeLoadResponse()) as mock_post,
            ):
                result = CliRunner().invoke(load, ["Qwen3.6", "--no-saved-settings"])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["settings"], {})

    def test_start_opens_browser_to_selected_view_when_server_already_running(self):
        with (
            patch("app.cli.config_loader.initialize_for_launch"),
            patch("app.cli.select_launch_view_for_cli", return_value="models"),
            patch("app.cli.is_server_online", return_value=True),
            patch("app.cli.should_open_browser", return_value=True),
            patch.dict(os.environ, {"LLAMASTUDIO_BROWSER_HOST": "b2"}),
            patch("app.cli.webbrowser.open") as mock_open,
        ):
            result = CliRunner().invoke(start)

        self.assertEqual(result.exit_code, 0, result.output)
        mock_open.assert_called_once_with("http://b2:8765/?view=models")

    def test_start_skips_browser_in_headless_session(self):
        with (
            patch("app.cli.config_loader.initialize_for_launch"),
            patch("app.cli.select_launch_view_for_cli", return_value="models"),
            patch("app.cli.is_server_online", return_value=True),
            patch("app.cli.should_open_browser", return_value=False),
            patch.dict(os.environ, {"LLAMASTUDIO_BROWSER_HOST": "b2"}),
            patch("app.cli.webbrowser.open") as mock_open,
        ):
            result = CliRunner().invoke(start)

        self.assertEqual(result.exit_code, 0, result.output)
        mock_open.assert_not_called()
        self.assertIn("Open", result.output)

    def test_start_launches_background_server_when_offline(self):
        with (
            patch("app.cli.config_loader.initialize_for_launch"),
            patch("app.cli.is_server_online", return_value=False),
            patch("app.cli.start_server_background", return_value=True) as mock_start,
        ):
            result = CliRunner().invoke(start)

        self.assertEqual(result.exit_code, 0, result.output)
        mock_start.assert_called_once()

    def test_cli_launch_view_prefers_chat_when_model_is_running(self):
        with (
            patch("app.cli.is_server_online", return_value=True),
            patch("app.cli.httpx.get", return_value=FakeStatusResponse(True)),
            patch("app.model_manager.scan_models", return_value=[]),
            patch("app.cli.config_loader.get_launch_view", return_value="chat") as mock_view,
        ):
            view = select_launch_view_for_cli()

        self.assertEqual(view, "chat")
        mock_view.assert_called_once_with(
            model_loaded=True,
            models_available=False,
            consume_first_launch=True,
        )

    def test_oneshot_image_uses_token_safe_multimodal_payload(self):
        tool_result = ToolResult(
            content="Loaded image.",
            images=[
                {
                    "name": "bike.png",
                    "mime_type": "image/png",
                    "data_url": "data:image/png;base64,iVBORw0KGgo=",
                    "size": 8,
                }
            ],
        )

        with (
            patch("app.cli.is_server_online", return_value=True),
            patch("app.cli.httpx.get", return_value=FakeStatusResponse(True)),
            patch("app.cli.httpx.post", return_value=FakeLoadResponse()),
            patch("app.tools.read_file", return_value=tool_result),
            patch(
                "app.cli.httpx.stream",
                return_value=FakeChatStreamResponse(),
            ) as mock_stream,
        ):
            result = CliRunner().invoke(
                oneshot,
                [
                    "--no-thinking",
                    "--image",
                    "test_imgs/bike.png",
                    "What is in this image?",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        payload = mock_stream.call_args.kwargs["json"]
        self.assertEqual(payload["message"], "What is in this image?")
        self.assertEqual(payload["images"], tool_result.images)
        self.assertIs(payload["enable_thinking"], False)
        self.assertIn("A bicycle.", result.output)
        self.assertIn("26 tokens", result.output)
        self.assertIn("19 in / 7 out", result.output)
        self.assertIn("33.3 tok/s", result.output)

    def test_oneshot_audio_uses_structured_multimodal_payload(self):
        tool_result = ToolResult(
            content="Loaded audio.",
            audios=[
                {
                    "name": "hello.flac",
                    "mime_type": "audio/flac",
                    "data_url": "data:audio/flac;base64,ZkxhQw==",
                    "size": 4,
                }
            ],
        )

        with (
            patch("app.cli.is_server_online", return_value=True),
            patch("app.cli.httpx.get", return_value=FakeStatusResponse(True)),
            patch("app.cli.httpx.post", return_value=FakeLoadResponse()),
            patch("app.tools.read_file", return_value=tool_result),
            patch(
                "app.cli.httpx.stream",
                return_value=FakeChatStreamResponse(),
            ) as mock_stream,
        ):
            result = CliRunner().invoke(
                oneshot,
                [
                    "--no-thinking",
                    "--audio",
                    "test_imgs/hello.flac",
                    "Transcribe this audio.",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        payload = mock_stream.call_args.kwargs["json"]
        self.assertEqual(payload["message"], "Transcribe this audio.")
        self.assertEqual(payload["audios"], tool_result.audios)
        self.assertIs(payload["enable_thinking"], False)

    def test_speech_status_reports_local_install_without_desktop_server(self):
        speech_status = {
            "engine": "whisper.cpp",
            "version": "v1.9.1",
            "installed": True,
            "binary": "/runtime/whisper-server",
            "running": False,
            "model": "small.en",
            "model_path": "/runtime/ggml-small.en.bin",
            "model_installed": True,
            "use_gpu": False,
        }
        with (
            patch("app.cli.is_server_online", return_value=False),
            patch("app.speech_manager.speech.get_status", return_value=speech_status),
        ):
            result = CliRunner().invoke(cli, ["speech", "status"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("small.en", result.output)
        self.assertIn("Stopped", result.output)

    def test_speech_transcribe_uses_raw_audio_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "voice.wav"
            audio_path.write_bytes(b"RIFF recording")
            with (
                patch("app.tools.check_path_safe", return_value=audio_path),
                patch("app.cli.ensure_server_online", return_value=True),
                patch("app.cli.httpx.post", return_value=FakeLoadResponse()) as post,
            ):
                result = CliRunner().invoke(
                    cli,
                    ["speech", "transcribe", "voice.wav"],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Hello from the microphone.", result.output)
        self.assertEqual(post.call_args.kwargs["content"], b"RIFF recording")
        self.assertEqual(post.call_args.kwargs["headers"]["Content-Type"], "audio/wav")


if __name__ == "__main__":
    unittest.main()
