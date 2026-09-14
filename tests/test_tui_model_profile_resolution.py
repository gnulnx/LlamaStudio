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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tui.models import _resolved_path


class TestResolvedPath(unittest.TestCase):
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

    def test_matches_a_profile_keyed_by_the_resolved_path(self):
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

            profiles = {str(blob.resolve()): {"ctx_size": 131072, "kv_cache_type": "q4_0"}}
            scanned_path = str(symlink)

            profile = profiles.get(scanned_path) or profiles.get(_resolved_path(scanned_path))

            self.assertIsNotNone(profile)
            self.assertEqual(profile["ctx_size"], 131072)


if __name__ == "__main__":
    unittest.main()
