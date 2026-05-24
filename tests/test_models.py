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
    @patch("app.main.settings")
    def test_delete_model_success(self, mock_settings, mock_server):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "models"
            model_dir.mkdir()
            model_file = model_dir / "test_model.gguf"
            model_file.touch()

            mock_settings.MODEL_DIRS = [str(model_dir)]
            mock_server._current_model = ""
            mock_server.is_running = False

            response = self.client.request(
                "DELETE",
                "/api/models/delete",
                json={"path": str(model_file)},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "ok")
            self.assertFalse(model_file.exists())

    @patch("app.main.server")
    @patch("app.main.settings")
    def test_delete_model_currently_loaded(self, mock_settings, mock_server):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "models"
            model_dir.mkdir()
            model_file = model_dir / "test_model.gguf"
            model_file.touch()

            mock_settings.MODEL_DIRS = [str(model_dir)]
            mock_server._current_model = str(model_file)
            mock_server.is_running = True

            response = self.client.request(
                "DELETE",
                "/api/models/delete",
                json={"path": str(model_file)},
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("loaded", response.json()["detail"])
            self.assertTrue(model_file.exists())

    @patch("app.main.settings")
    def test_delete_model_unsafe_path(self, mock_settings):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            allowed_dir = Path(tmp_dir) / "allowed"
            allowed_dir.mkdir()
            unsafe_file = Path(tmp_dir) / "unsafe.gguf"
            unsafe_file.touch()

            mock_settings.MODEL_DIRS = [str(allowed_dir)]

            response = self.client.request(
                "DELETE",
                "/api/models/delete",
                json={"path": str(unsafe_file)},
            )

            self.assertEqual(response.status_code, 403)
            self.assertIn("Access denied", response.json()["detail"])
            self.assertTrue(unsafe_file.exists())


if __name__ == "__main__":
    unittest.main()
