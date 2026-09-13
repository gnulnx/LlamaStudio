"""Palette schema, packaged defaults, safe overrides, and readable color roles."""

import json
import unittest
from dataclasses import asdict
from unittest.mock import patch

from app.tui.palette import Palette, load_palette


def contrast(first: str, second: str) -> float:
    def luminance(color):
        values = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            for value in values
        ]
        return sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


class TestPalette(unittest.TestCase):
    def test_packaged_palette_drives_theme_and_every_css_role(self):
        palette = load_palette()
        theme = palette.theme()
        self.assertEqual(theme.primary, palette.primary)
        self.assertEqual(theme.success, palette.success)
        for name, color in asdict(palette).items():
            self.assertEqual(theme.variables[f"studio-{name.replace('_', '-')}"], color)

    def test_text_and_selection_contrast(self):
        palette = load_palette()
        for foreground, background in (
            (palette.text, palette.surface),
            (palette.muted, palette.panel),
            (palette.on_accent, palette.selection),
            (palette.on_accent, palette.selection_inactive),
            (palette.warning, palette.surface),
            (palette.teal, palette.surface),
            (palette.success, palette.panel),
        ):
            with self.subTest(foreground=foreground, background=background):
                self.assertGreaterEqual(contrast(foreground, background), 4.5)
        # Primary buttons use bold labels.
        self.assertGreaterEqual(contrast(palette.on_accent, palette.primary), 3)

    def test_rejects_unknown_missing_and_invalid_colors(self):
        colors = asdict(load_palette())
        for value in ("red", "#abc", "#gggggg", "#123456; display:none", None, 123):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "primary"):
                Palette.from_mapping({**colors, "primary": value})
        with self.assertRaisesRegex(ValueError, "Unknown.*typo"):
            Palette.from_mapping({**colors, "typo": "#123456"})
        colors.pop("primary")
        with self.assertRaisesRegex(ValueError, "Missing.*primary"):
            Palette.from_mapping(colors)

    def test_workspace_override_can_contain_only_the_roles_being_tuned(self):
        default = load_palette()
        with patch("app.tui.palette.check_path_safe") as safe:
            safe.return_value.read_text.return_value = json.dumps({"primary": "#aabbcc"})
            palette = load_palette("colors.json")
        safe.assert_called_once_with("colors.json")
        safe.return_value.read_text.assert_called_once_with(encoding="utf-8")
        self.assertEqual(palette.primary, "#aabbcc")
        self.assertEqual(palette.background, default.background)

    def test_bad_json_and_wrong_top_level_are_rejected(self):
        for content in ("{broken", "[]", "null", '"purple"'):
            with self.subTest(content=content), patch("app.tui.palette.check_path_safe") as safe:
                safe.return_value.read_text.return_value = content
                with self.assertRaises(ValueError):
                    load_palette("colors.json")

    def test_every_reload_revalidates_workspace_path(self):
        with (
            patch("app.tui.palette.check_path_safe", side_effect=ValueError("outside workspace")),
            self.assertRaisesRegex(ValueError, "outside workspace"),
        ):
            load_palette("../outside.json")
