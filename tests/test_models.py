import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app


class TestModelsAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("app.main.server")
    def test_delete_model_success(self, mock_server):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "models"
            model_dir.mkdir()
            model_file = model_dir / "test_model.gguf"
            model_file.touch()

            mock_server._current_model = ""
            mock_server.is_running = False

            with patch(
                "app.main.config_loader.get_model_directories",
                return_value=[str(model_dir)],
            ):
                response = self.client.request(
                    "DELETE",
                    "/api/models/delete",
                    json={"path": str(model_file)},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "ok")
            self.assertFalse(model_file.exists())

    @patch("app.main.server")
    def test_delete_model_currently_loaded(self, mock_server):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "models"
            model_dir.mkdir()
            model_file = model_dir / "test_model.gguf"
            model_file.touch()

            mock_server._current_model = str(model_file)
            mock_server.is_running = True

            with patch(
                "app.main.config_loader.get_model_directories",
                return_value=[str(model_dir)],
            ):
                response = self.client.request(
                    "DELETE",
                    "/api/models/delete",
                    json={"path": str(model_file)},
                )

            self.assertEqual(response.status_code, 400)
            self.assertIn("loaded", response.json()["detail"])
            self.assertTrue(model_file.exists())

    def test_delete_model_unsafe_path(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            allowed_dir = Path(tmp_dir) / "allowed"
            allowed_dir.mkdir()
            unsafe_file = Path(tmp_dir) / "unsafe.gguf"
            unsafe_file.touch()

            with patch(
                "app.main.config_loader.get_model_directories",
                return_value=[str(allowed_dir)],
            ):
                response = self.client.request(
                    "DELETE",
                    "/api/models/delete",
                    json={"path": str(unsafe_file)},
                )

            self.assertEqual(response.status_code, 403)
            self.assertIn("Access denied", response.json()["detail"])
            self.assertTrue(unsafe_file.exists())

    @patch("app.main.server")
    def test_index_embeds_launch_view_from_config_loader(self, mock_server):
        mock_server.get_status.return_value = {
            "running": False,
            "current_model": None,
            "current_model_name": None,
            "is_loading": False,
        }
        with (
            patch("app.model_manager.get_models", return_value=[]),
            patch("app.main.config_loader.get_launch_view", return_value="discover"),
            patch(
                "app.main.config_loader.get_chat_defaults",
                return_value={
                    "system_prompt": "You are a helpful assistant.",
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2048,
                },
            ),
        ):
            response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("let initialLaunchView = 'discover';", response.text)


if __name__ == "__main__":
    unittest.main()
