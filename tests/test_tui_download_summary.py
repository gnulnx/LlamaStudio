import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tui.widgets import download_summary, duration_label


class TestDurationLabel(unittest.TestCase):
    def test_formats_by_magnitude(self):
        self.assertEqual(duration_label(45), "45s")
        self.assertEqual(duration_label(800), "13m 20s")
        self.assertEqual(duration_label(7_530), "2h 05m")

    def test_unknown_or_absent_time_renders_nothing(self):
        for value in (0, None, -1):
            with self.subTest(value=value):
                self.assertEqual(duration_label(value), "")


class TestDownloadSummary(unittest.TestCase):
    def test_reports_transferred_total_speed_and_eta(self):
        summary = download_summary({
            "filename": "Tiny-Q4_K_M.gguf",
            "downloaded_bytes": 3 * 1024**3,
            "total_bytes": 8 * 1024**3,
            "speed_mb": 41.4,
            "eta_seconds": 800,
        })
        self.assertEqual(
            summary, "Tiny-Q4_K_M.gguf  ·  3.00 / 8.00 GiB  ·  41.4 MiB/s  ·  13m 20s left"
        )

    def test_unknown_total_still_reports_bytes_on_disk(self):
        """A response without Content-Length has no percentage, but has progress."""
        summary = download_summary({
            "filename": "Tiny.gguf",
            "downloaded_bytes": 1024**3,
            "speed_mb": 9.0,
        })
        self.assertEqual(summary, "Tiny.gguf  ·  1.00 GiB  ·  9.0 MiB/s")

    def test_survives_an_empty_progress_payload(self):
        self.assertEqual(download_summary({}), "Downloading...  ·  0.0 MiB/s")


if __name__ == "__main__":
    unittest.main()
