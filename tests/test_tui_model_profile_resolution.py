"""Saved model profiles must be found even when the scanner reports a symlinked path.

Hugging Face Hub caches (and similar layouts) store GGUF files as
`snapshots/<hash>/file.gguf` symlinks pointing at `blobs/<sha256>`. The model
scanner reports the symlink path, while saved profiles are keyed by the
resolved path, so a naive string-equality lookup never matches.
"""

import os
import sys
import unittest
from pathlib import Path

import httpx
from test_tui import Backend
from textual.widgets import Input, Select

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tui.application import StudioApp
from app.tui.client import StudioClient
from app.tui.models import _resolved_path


class TestResolvedPath(unittest.IsolatedAsyncioTestCase):
    def test_resolves_symlinked_snapshot_to_its_blob_target(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            blobs_dir = Path(tmp) / "blobs"
            blobs_dir.mkdir()
            blob = blobs_dir / "abc123"
            blob.write_text("fake model")

            snapshot_dir = Path(tmp) / "snapshots" / "deadbeef"
            snapshot_dir.mkdir(parents=True)
            symlink = snapshot_dir / "model.gguf"
            symlink.symlink_to(blob)

            self.assertEqual(_resolved_path(str(symlink)), str(blob.resolve()))

    async def test_profile_keyed_by_blob_is_applied_to_the_actual_model_settings_form(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            blobs_dir = Path(tmp) / "blobs"
            blobs_dir.mkdir()
            blob = blobs_dir / "abc123"
            blob.write_text("fake model")

            snapshot_dir = Path(tmp) / "snapshots" / "deadbeef"
            snapshot_dir.mkdir(parents=True)
            symlink = snapshot_dir / "model.gguf"
            symlink.symlink_to(blob)

            backend = Backend()
            backend.models = [{**backend.models[0], "path": str(symlink)}]
            backend.profiles = {str(blob.resolve()): {"ctx_size": 131072, "kv_cache_type": "q4_0"}}
            app = StudioApp(
                "http://studio",
                initial_view="models",
                client=StudioClient("http://studio", transport=httpx.MockTransport(backend)),
            )
            async with app.run_test(size=(180, 48)) as pilot:
                for _ in range(3):
                    await app.workers.wait_for_complete()
                    await pilot.pause(0.1)
                self.assertEqual(app.query_one("#model-context", Input).value, "131072")
                self.assertEqual(app.query_one("#model-cache", Select).value, "q4_0")


if __name__ == "__main__":
    unittest.main()
