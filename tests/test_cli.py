import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.cli import load, load_saved_model_settings, select_launch_view_for_cli, start


class FakeLoadResponse:
    status_code = 200
    text = "ok"


class FakeStatusResponse:
    def __init__(self, running):
        self.running = running

    def json(self):
        return {"running": self.running}


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
            patch("app.cli.webbrowser.open") as mock_open,
        ):
            result = CliRunner().invoke(start)

        self.assertEqual(result.exit_code, 0, result.output)
        mock_open.assert_called_once_with("http://127.0.0.1:8765/?view=models")

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


if __name__ == "__main__":
    unittest.main()
