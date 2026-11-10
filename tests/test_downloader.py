import asyncio
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import Response
from fastapi.testclient import TestClient

# Ensure the app package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import contextlib

from app.downloader import downloader
from app.model_manager import (
    _gguf_model_sizes,
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
        mock_details = MagicMock()
        mock_details.status_code = 200
        mock_details.json.return_value = {
            "siblings": [
                {"rfilename": "model-Q4_K_M.gguf", "size": 5 * 1024**3},
                {"rfilename": "mmproj.gguf", "size": 1024**3},
            ]
        }
        mock_get.side_effect = [mock_response, mock_details]

        results = await search_huggingface_models("qwen", "downloads")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "Qwen/Qwen2.5-7B-Instruct-GGUF")
        self.assertEqual(results[0]["author"], "Qwen")
        self.assertEqual(results[0]["gguf_sizes"], {"model-Q4_K_M.gguf": 5 * 1024**3})
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(mock_get.call_args.kwargs["params"], {"blobs": "true"})

    @patch("httpx.AsyncClient.get")
    async def test_search_keeps_results_when_size_lookup_fails(self, mock_get):
        response = MagicMock(status_code=200)
        response.json.return_value = [{"id": "org/model"}]
        mock_get.side_effect = [response, OSError("Hub unavailable")]
        self.assertEqual(
            await search_huggingface_models("model"),
            [{"id": "org/model", "gguf_sizes": {}}],
        )

    @patch("httpx.AsyncClient.get")
    async def test_search_bounds_concurrent_size_lookups(self, mock_get):
        active = peak = 0

        async def get(url, **kwargs):
            nonlocal active, peak
            response = MagicMock(status_code=200)
            if url.endswith("/api/models"):
                response.json.return_value = [{"id": f"org/model-{i}"} for i in range(12)]
            else:
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1
                response.json.return_value = {"siblings": []}
            return response

        mock_get.side_effect = get
        self.assertEqual(len(await search_huggingface_models("model")), 12)
        self.assertEqual(peak, 5)

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
            "siblings": [{"rfilename": "gemma-2-9b-it.Q4_K_M.gguf", "size": 5 * 1024**3}],
        }
        mock_get.return_value = mock_response

        details = await get_huggingface_model_details("google/gemma-2-9b-it-GGUF")
        self.assertIsNotNone(details)
        self.assertEqual(details["id"], "google/gemma-2-9b-it-GGUF")
        self.assertEqual(len(details["siblings"]), 1)
        self.assertEqual(details["gguf_sizes"], {"gemma-2-9b-it.Q4_K_M.gguf": 5 * 1024**3})
        self.assertEqual(details["gguf_files"], ["gemma-2-9b-it.Q4_K_M.gguf"])
        self.assertEqual(mock_get.call_args.kwargs["params"], {"blobs": "true"})

    @patch("httpx.AsyncClient.get")
    async def test_get_huggingface_model_readme_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "# Gemma 2\nThis is a cool model."
        mock_get.return_value = mock_response

        readme = await get_huggingface_model_readme("google/gemma-2-9b-it-GGUF")
        self.assertIn("Gemma 2", readme)


class TestGGUFModelSizes(unittest.TestCase):
    def test_sizes_exclude_projectors_and_non_models_and_use_lfs_fallback(self):
        self.assertEqual(
            _gguf_model_sizes([
                {"rfilename": "model-Q4.gguf", "size": 5 * 1024**3},
                {"rfilename": "model-Q8.GGUF", "lfs": {"size": 9 * 1024**3}},
                {"rfilename": "mmproj-BF16.gguf", "size": 1024**3},
                {"rfilename": "README.md", "size": 100},
                {"rfilename": "unknown.gguf"},
                {"rfilename": "empty.gguf", "size": 0},
                {"rfilename": "imatrix_unsloth.gguf", "size": 1000},
                {"rfilename": "MTP/mtp-model-Q4_0.gguf", "size": 1024**3},
                {"rfilename": "model-FastMTP-32K.gguf", "size": 1024**3},
                {"rfilename": "model-draft-Q4_0.gguf", "size": 1024**3},
            ]),
            {"model-Q4.gguf": 5 * 1024**3, "model-Q8.GGUF": 9 * 1024**3},
        )

    def test_main_model_names_can_contain_mtp_and_imatrix(self):
        name = "model-MTP-IMATRIX-MAX-Q4_K_M.gguf"
        self.assertEqual(_gguf_model_sizes([{"rfilename": name, "size": 42}]), {name: 42})

    def test_shards_use_complete_model_size_and_stay_separate_by_directory(self):
        siblings = [
            {"rfilename": "Q8/model-00001-of-00002.gguf", "size": 8 * 1024**3},
            {"rfilename": "Q8/model-00002-of-00002.gguf", "size": 2 * 1024**3},
            {"rfilename": "Q4/model-00001-of-00002.gguf", "size": 3 * 1024**3},
            {"rfilename": "Q4/model-00002-of-00002.gguf", "size": 2 * 1024**3},
        ]
        sizes = _gguf_model_sizes(siblings)
        for sibling in siblings:
            name = sibling["rfilename"]
            self.assertEqual(sizes[name], (10 if name.startswith("Q8/") else 5) * 1024**3)

    def test_incomplete_or_unsized_shards_are_unknown(self):
        for parts in ([1], [0, 1], [1, 3]):
            with self.subTest(parts=parts):
                self.assertEqual(
                    _gguf_model_sizes([
                        {"rfilename": f"model-{part:05}-of-00002.gguf", "size": 1024**3}
                        for part in parts
                    ]),
                    {},
                )
        self.assertEqual(
            _gguf_model_sizes([
                {"rfilename": "model-00001-of-00002.gguf", "size": 1024**3},
                {"rfilename": "model-00002-of-00002.gguf"},
            ]),
            {},
        )


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
        self.payload = b"GGUF" + bytes(range(256)) * 4096
        self.started = threading.Event()
        self.release = threading.Event()
        self.hold = False
        self.response_status = 200
        self.ranges = []
        test = self

        class Origin(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                range_header = self.headers.get("Range", "bytes=0-")
                test.ranges.append(range_header)
                offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
                body = test.payload[offset:]
                self.send_response(206 if offset else test.response_status)
                self.send_header("Content-Length", str(len(body)))
                if offset:
                    self.send_header(
                        "Content-Range",
                        f"bytes {offset}-{len(test.payload) - 1}/{len(test.payload)}",
                    )
                self.end_headers()
                test.started.set()
                if test.hold:
                    test.release.wait(timeout=10)
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(body)

        self.origin = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
        self.thread = threading.Thread(target=self.origin.serve_forever, daemon=True)
        self.thread.start()
        self.directory = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.directory.name)
        from app import model_manager

        self.model_cache = model_manager._model_cache
        self.directories = patch(
            "app.downloader.config_loader.get_model_directories", return_value=[str(self.root)]
        )
        self.directories.start()
        self.origin_url = patch.object(
            downloader,
            "_resolve_url",
            return_value=f"http://127.0.0.1:{self.origin.server_port}/model.gguf",
        )
        self.origin_url.start()

    async def asyncTearDown(self):
        await downloader.cancel_download()
        self.release.set()
        await asyncio.to_thread(self.origin.shutdown)
        self.origin.server_close()
        self.thread.join(timeout=2)
        self.origin_url.stop()
        self.directories.stop()
        self.directory.cleanup()
        from app import model_manager

        model_manager._model_cache = self.model_cache

    async def test_idle_progress_does_not_expose_stale_transfer_metadata(self):
        downloader.repo_id = "old/repo"
        downloader.downloaded_bytes = 42
        self.assertFalse(downloader.is_active)
        progress = downloader.get_progress()
        self.assertEqual(progress, {"status": "idle"})

    async def test_download_transfers_complete_bytes_and_refreshes_the_real_catalog(self):
        from app.model_manager import get_models

        self.assertTrue(await downloader.start_download("author/repo", "model.gguf"))
        progress = await downloader.wait_for_completion()
        target = self.root / "author/repo/model.gguf"
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertFalse(target.with_name("model.gguf.tmp").exists())
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(progress["downloaded_bytes"], len(self.payload))
        self.assertEqual(progress["percent"], 100)
        self.assertIn(str(target), [model.path for model in get_models()])
        from app.main import app

        published = TestClient(app).get("/api/models/download/active").json()
        self.assertFalse(published["active"])
        self.assertEqual(published["progress"]["downloaded_bytes"], len(self.payload))

    async def test_second_download_is_rejected_during_a_real_transfer(self):
        self.hold = True
        self.assertTrue(await downloader.start_download("author/repo", "model.gguf"))
        self.assertTrue(await asyncio.to_thread(self.started.wait, 5))
        from app.main import app

        self.assertTrue(TestClient(app).get("/api/models/download/active").json()["active"])
        self.assertFalse(await downloader.start_download("other/repo", "other.gguf"))
        self.assertEqual(downloader.repo_id, "author/repo")
        self.release.set()
        self.assertEqual((await downloader.wait_for_completion())["status"], "completed")
        self.assertEqual((self.root / "author/repo/model.gguf").read_bytes(), self.payload)
        self.assertFalse((self.root / "other/repo/other.gguf").exists())

    async def test_cancelling_a_real_transfer_removes_partial_files(self):
        self.hold = True
        self.assertTrue(await downloader.start_download("author/repo", "model.gguf"))
        self.assertTrue(await asyncio.to_thread(self.started.wait, 5))
        await downloader.cancel_download()
        self.assertEqual(downloader.status, "cancelled")
        self.assertFalse(downloader.is_active)
        self.assertFalse((self.root / "author/repo/model.gguf").exists())
        self.assertFalse((self.root / "author/repo/model.gguf.tmp").exists())

    async def test_http_failure_never_publishes_a_completed_model(self):
        self.response_status = 403
        self.assertTrue(await downloader.start_download("author/repo", "model.gguf"))
        result = await downloader.wait_for_completion()
        self.assertEqual(result["status"], "failed")
        self.assertFalse((self.root / "author/repo/model.gguf").exists())
        self.assertFalse((self.root / "author/repo/model.gguf.tmp").exists())

    async def test_partial_download_resumes_over_http_without_losing_bytes(self):
        target = self.root / "author/repo/model.gguf"
        target.parent.mkdir(parents=True)
        target.with_name("model.gguf.tmp").write_bytes(self.payload[:128])
        self.assertTrue(await downloader.start_download("author/repo", "model.gguf"))
        self.assertEqual((await downloader.wait_for_completion())["status"], "completed")
        self.assertEqual(self.ranges, ["bytes=128-"])
        self.assertEqual(target.read_bytes(), self.payload)


class TestDownloadUrl(unittest.TestCase):
    @patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=False)
    def test_direct_download_helpers_escape_paths_and_pass_token(self):
        self.assertEqual(
            downloader._resolve_url("author/repo name", "sub dir/model Q4.gguf"),
            "https://huggingface.co/author/repo%20name/resolve/main/sub%20dir/model%20Q4.gguf",
        )

        headers = downloader._hf_headers()
        self.assertEqual(headers["User-Agent"], "LLamaStudio-Client")
        self.assertEqual(headers["Authorization"], "Bearer test-token")


class TestDownloadSpeedMeasurement(unittest.TestCase):
    """Rate and ETA describe the current transfer, not the file's whole history."""

    GiB = 1024**3
    MiB = 1024**2

    def setUp(self):
        self.clock = 1000.0
        patcher = patch("app.downloader.time.monotonic", side_effect=lambda: self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        downloader.status = "downloading"
        downloader.error_message = None

    def transfer(self, seconds: int, bytes_per_second: int) -> None:
        for _ in range(seconds):
            self.clock += 1.0
            downloader.downloaded_bytes += bytes_per_second
            downloader._note_progress()

    def test_resumed_download_excludes_bytes_already_on_disk(self):
        """Counting the existing file as this session's transfer invents huge rates."""
        downloader.total_bytes = 50 * self.GiB
        downloader.downloaded_bytes = 40 * self.GiB
        downloader._reset_speed_samples()

        self.transfer(seconds=10, bytes_per_second=10 * self.MiB)

        progress = downloader.get_progress()
        self.assertAlmostEqual(progress["speed_mb"], 10.0, places=1)
        # ~9.9 GiB left at 10 MiB/s. The pre-existing 40 GiB must not shorten it.
        self.assertGreater(progress["eta_seconds"], 900)

    def test_rate_follows_a_slowdown_instead_of_averaging_it_away(self):
        downloader.total_bytes = 50 * self.GiB
        downloader.downloaded_bytes = 0
        downloader._reset_speed_samples()

        self.transfer(seconds=50, bytes_per_second=100 * self.MiB)
        self.assertGreater(downloader.get_progress()["speed_mb"], 90)

        self.transfer(seconds=10, bytes_per_second=1 * self.MiB)
        self.assertAlmostEqual(downloader.get_progress()["speed_mb"], 1.0, places=1)

    def test_a_stalled_transfer_decays_to_zero(self):
        downloader.total_bytes = 50 * self.GiB
        downloader.downloaded_bytes = 0
        downloader._reset_speed_samples()
        self.transfer(seconds=5, bytes_per_second=100 * self.MiB)

        self.clock += 30.0  # nothing arrives

        progress = downloader.get_progress()
        self.assertEqual(progress["speed_mb"], 0.0)
        self.assertEqual(progress["eta_seconds"], 0)

    def test_restarting_the_transfer_restarts_the_measurement(self):
        """A server that ignores the resume range rewinds the bytes."""
        downloader.total_bytes = 50 * self.GiB
        downloader.downloaded_bytes = 10 * self.GiB
        downloader._reset_speed_samples()
        self.transfer(seconds=5, bytes_per_second=50 * self.MiB)

        downloader.downloaded_bytes = 0
        downloader._reset_speed_samples()
        self.transfer(seconds=10, bytes_per_second=20 * self.MiB)

        self.assertAlmostEqual(downloader.get_progress()["speed_mb"], 20.0, places=1)

    def test_no_samples_yet_reports_no_rate_rather_than_dividing_by_zero(self):
        downloader.total_bytes = 0
        downloader.downloaded_bytes = 0
        downloader._samples = None
        progress = downloader.get_progress()
        self.assertEqual(progress["speed_mb"], 0.0)
        self.assertEqual(progress["eta_seconds"], 0)


class TestFastAPIPoints(unittest.TestCase):
    """Test Suite for FastAPI Endpoints in main.py."""

    @patch("app.model_manager.search_huggingface_models", new_callable=AsyncMock)
    def test_search_endpoint(self, mock_search):
        from app.main import search_models

        mock_search.return_value = [{"id": "test/model", "downloads": 10}]

        http_response = Response()
        response_data = asyncio.run(search_models(response=http_response, q="test"))
        self.assertEqual(response_data, {"models": [{"id": "test/model", "downloads": 10}]})
        self.assertEqual(http_response.headers["Cache-Control"], "no-store, max-age=0")
        mock_search.assert_called_once_with("test", "downloads")

    @patch("app.model_manager.get_huggingface_model_details", new_callable=AsyncMock)
    @patch("app.model_manager.get_huggingface_model_readme", new_callable=AsyncMock)
    def test_details_endpoint(self, mock_readme, mock_details):
        from app.main import get_hf_model_details

        mock_details.return_value = {"id": "test/model", "siblings": []}
        mock_readme.return_value = "# Model README"

        http_response = Response()
        response_data = asyncio.run(
            get_hf_model_details(repo_id="test/model", response=http_response)
        )
        self.assertEqual(response_data["details"]["id"], "test/model")
        self.assertEqual(response_data["readme"], "# Model README")
        self.assertEqual(http_response.headers["Cache-Control"], "no-store, max-age=0")


if __name__ == "__main__":
    unittest.main()
