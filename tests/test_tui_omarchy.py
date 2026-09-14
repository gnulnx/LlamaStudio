"""Omarchy theme following: inert without Omarchy, fixed status colors, safe fallback."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.tui import omarchy
from app.tui.palette import load_palette

HINTERLANDS = (
    "accent\t#686868\nbackground\t#222222\nforeground\t#ffffff\n"
    "selection_background\t#686868\nselection_foreground\t#ffffff\nmode\tdark\n"
)


def resolver(stdout: str):
    return patch("app.tui.omarchy.subprocess.run", return_value=SimpleNamespace(stdout=stdout))


class TestOmarchy(unittest.TestCase):
    def test_detected_only_with_theme_file_and_resolver_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            colors = Path(tmp) / "colors.toml"
            with patch("app.tui.omarchy.THEME_COLORS", colors):
                with patch(
                    "app.tui.omarchy.shutil.which", return_value="/usr/bin/omarchy-theme-color"
                ):
                    self.assertFalse(omarchy.omarchy_detected())
                    self.assertIsNone(omarchy.theme_signature())
                    colors.write_text('background = "#222222"\n')
                    self.assertTrue(omarchy.omarchy_detected())
                    self.assertIsNotNone(omarchy.theme_signature())
                with patch("app.tui.omarchy.shutil.which", return_value=None):
                    self.assertFalse(omarchy.omarchy_detected())

    def test_maps_theme_keeps_status_colors_and_reports_mode(self):
        bundled = load_palette()
        with resolver(HINTERLANDS):
            palette, dark = omarchy.omarchy_palette()
        self.assertTrue(dark)
        self.assertEqual(
            (palette.background, palette.text, palette.primary, palette.panel),
            ("#222222", "#ffffff", "#686868", "#2d2d2d"),
        )
        for role in ("success", "teal", "warning", "error"):
            self.assertEqual(getattr(palette, role), getattr(bundled, role))
        with resolver(HINTERLANDS.replace("mode\tdark", "mode\tlight")):
            self.assertFalse(omarchy.omarchy_palette()[1])

    def test_unusable_theme_raises_value_error(self):
        failures = (
            patch("app.tui.omarchy.subprocess.run", side_effect=FileNotFoundError("missing")),
            patch(
                "app.tui.omarchy.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "omarchy-theme-color"),
            ),
            resolver("background\t#222222\n"),
            resolver(HINTERLANDS.replace("#222222", "#22222280")),
            resolver(HINTERLANDS.replace("#686868", "rgba(104,104,104,1)")),
        )
        for failure in failures:
            with self.subTest(failure=failure), failure, self.assertRaises(ValueError):
                omarchy.omarchy_palette()
