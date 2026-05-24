import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the app package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import contextlib

from app.downloader import downloader
from app.model_manager import (
    get_huggingface_model_details,
    get_huggingface_model_readme,
    search_huggingface_models,
)


class TestModelManagerHF(unittest.IsolatedAsyncioTestCase):
    """Test Suite for Hugging Face integration functions in model_manager.py."""

    @patch("httpx.AsyncClient.get")
    async def test_search_huggingface_models_success(self, mock_get):
        # Mock successful JSON response from Hugging Face
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
                "author": "Qwen",
                "downloads": 15000,
                "likes": 420,
                "tags": ["gguf", "text-generation"],
            }
        ]
        mock_get.return_value = mock_response

        results = await search_huggingface_models("qwen", "downloads")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "Qwen/Qwen2.5-7B-Instruct-GGUF")
        self.assertEqual(results[0]["author"], "Qwen")
        mock_get.assert_called_once()

    @patch("httpx.AsyncClient.get")
    async def test_search_huggingface_models_failure(self, mock_get):
        # Mock API error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        results = await search_huggingface_models("invalid", "downloads")
        self.assertEqual(results, [])

    @patch("httpx.AsyncClient.get")
    async def test_get_huggingface_model_details_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "google/gemma-2-9b-it-GGUF",
            "siblings": [{"rpath": "gemma-2-9b-it.Q4_K_M.gguf"}],
        }
        mock_get.return_value = mock_response

        details = await get_huggingface_model_details("google/gemma-2-9b-it-GGUF")
        self.assertIsNotNone(details)
        self.assertEqual(details["id"], "google/gemma-2-9b-it-GGUF")
        self.assertEqual(len(details["siblings"]), 1)

    @patch("httpx.AsyncClient.get")
    async def test_get_huggingface_model_readme_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "# Gemma 2\nThis is a cool model."
        mock_get.return_value = mock_response

        readme = await get_huggingface_model_readme("google/gemma-2-9b-it-GGUF")
        self.assertIn("Gemma 2", readme)


class TestModelDownloader(unittest.IsolatedAsyncioTestCase):
    """Test Suite for ModelDownloader background manager in downloader.py."""

    async def asyncSetUp(self):
        # Make sure downloader is reset to idle state
        downloader.status = "idle"
        downloader.repo_id = None
        downloader.filename = None
        downloader.downloaded_bytes = 0
        downloader.total_bytes = 0
        downloader.error_message = None
        if downloader._active_task and not downloader._active_task.done():
            downloader._active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await downloader._active_task
        downloader._active_task = None
        downloader._cancel_event = None

    async def test_downloader_initial_state(self):
        self.assertFalse(downloader.is_active)
        progress = downloader.get_progress()
        self.assertEqual(progress["status"], "idle")

    @patch("app.downloader.ModelDownloader._download_loop", new_callable=AsyncMock)
    async def test_start_download_success(self, mock_download_loop):
        # Mock download loop to run cleanly without actually downloading anything
        mock_download_loop.return_value = None

        started = await downloader.start_download("test/repo", "model.gguf")
        self.assertTrue(started)
        self.assertTrue(downloader.is_active)
        self.assertEqual(downloader.repo_id, "test/repo")
        self.assertEqual(downloader.filename, "model.gguf")
        self.assertEqual(downloader.status, "downloading")

        # Cleanup task
        await downloader.cancel_download()

    @patch("app.downloader.ModelDownloader._download_loop", new_callable=AsyncMock)
    async def test_double_download_prevention(self, mock_download_loop):
        mock_download_loop.return_value = None

        started1 = await downloader.start_download("test/repo1", "model1.gguf")
        self.assertTrue(started1)

        # Attempting second download should return False
        started2 = await downloader.start_download("test/repo2", "model2.gguf")
        self.assertFalse(started2)

        # Cleanup
        await downloader.cancel_download()

    @patch("app.model_manager.refresh_models")
    @patch("app.downloader.settings")
    async def test_download_loop_uses_direct_huggingface_download(
        self, mock_settings, mock_refresh
    ):
        with tempfile.TemporaryDirectory() as tmp:
            mock_settings.MODEL_DIRS = [tmp]

            async def fake_download(repo_id, filename, tmp_path):
                self.assertEqual(repo_id, "author/repo")
                self.assertEqual(filename, "model.gguf")
                target = Path(tmp_path)
                target.write_bytes(b"gguf")

            with patch.object(
                downloader,
                "_download_from_huggingface",
                side_effect=fake_download,
            ) as mock_direct_download:
                await downloader._download_loop("author/repo", "model.gguf")

            mock_direct_download.assert_called_once()
            mock_refresh.assert_called_once()
            self.assertEqual(downloader.status, "completed")
            self.assertTrue((Path(tmp) / "author" / "repo" / "model.gguf").exists())

    @patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=False)
    def test_direct_download_helpers_escape_paths_and_pass_token(self):
        self.assertEqual(
            downloader._resolve_url("author/repo name", "sub dir/model Q4.gguf"),
            "https://huggingface.co/author/repo%20name/resolve/main/sub%20dir/model%20Q4.gguf",
        )

        headers = downloader._hf_headers()
        self.assertEqual(headers["User-Agent"], "LLamaStudio-Client")
        self.assertEqual(headers["Authorization"], "Bearer test-token")


class TestFastAPIPoints(unittest.TestCase):
    """Test Suite for FastAPI Endpoints in main.py."""

    @patch("app.model_manager.search_huggingface_models", new_callable=AsyncMock)
    def test_search_endpoint(self, mock_search):
        from app.main import search_models

        mock_search.return_value = [{"id": "test/model", "downloads": 10}]

        response = asyncio.run(search_models(q="test"))
        self.assertEqual(response, {"models": [{"id": "test/model", "downloads": 10}]})
        mock_search.assert_called_once_with("test", "downloads")

    @patch("app.model_manager.get_huggingface_model_details", new_callable=AsyncMock)
    @patch("app.model_manager.get_huggingface_model_readme", new_callable=AsyncMock)
    def test_details_endpoint(self, mock_readme, mock_details):
        from app.main import get_hf_model_details

        mock_details.return_value = {"id": "test/model", "siblings": []}
        mock_readme.return_value = "# Model README"

        response = asyncio.run(get_hf_model_details(repo_id="test/model"))
        self.assertEqual(response["details"]["id"], "test/model")
        self.assertEqual(response["readme"], "# Model README")

    def test_active_download_endpoint(self):
        from app.main import is_download_active

        response = asyncio.run(is_download_active())
        self.assertIn("active", response)


if __name__ == "__main__":
    unittest.main()
