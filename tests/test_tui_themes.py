"""Theme source contracts, fallback, override precedence and bundled contrast."""

import json
import unittest
from unittest.mock import patch

from test_tui_palette import contrast

from app.tui.palette import load_palette
from app.tui.themes import (
    THEME_NAMES,
    BundledThemeAdapter,
    ThemeAdapter,
    create_theme_adapter,
    resolve_initial_theme,
)


class TestThemes(unittest.TestCase):
    def test_default_is_unchanged_and_system_fallback_is_explicit_and_idle(self):
        default = create_theme_adapter().resolve()
        self.assertEqual(default.palette, load_palette())
        self.assertTrue(default.dark)
        system = create_theme_adapter("system")
        fallback = system.resolve()
        self.assertEqual(fallback.palette, default.palette)
        self.assertIn("using Default", fallback.notice)
        self.assertIsNone(system.refresh_interval)

    def test_bundled_modes_and_contrast(self):
        for name in THEME_NAMES[:3]:
            resolved = create_theme_adapter(name).resolve()
            self.assertEqual(resolved.dark, name != "light")
            palette = resolved.palette
            for foreground, background in (
                (palette.text, palette.surface),
                (palette.muted, palette.panel),
                (palette.on_accent, palette.selection),
                (palette.on_accent, palette.selection_inactive),
                (palette.warning, palette.surface),
                (palette.teal, palette.surface),
                (palette.success, palette.panel),
                (palette.error, palette.surface),
            ):
                with self.subTest(name=name, foreground=foreground, background=background):
                    self.assertGreaterEqual(contrast(foreground, background), 4.5)

    def test_custom_palette_overrides_colors_and_inherits_explicit_mode(self):
        with patch("app.tui.palette.check_path_safe") as safe:
            safe.return_value.read_text.return_value = json.dumps({"primary": "#123456"})
            adapter = create_theme_adapter("light", "colors.json")
            custom = adapter.resolve()
            self.assertFalse(custom.dark)
            self.assertEqual(custom.palette.primary, "#123456")
            self.assertEqual(custom.palette.panel, "#ffffff")
            safe.side_effect = ValueError("outside workspace")
            with self.assertRaisesRegex(ValueError, "outside workspace"):
                adapter.resolve()

    def test_explicit_palette_and_bundled_choices_never_detect_system(self):
        with (
            patch("app.tui.themes.SYSTEM_ADAPTERS", (lambda: self.fail("detected system"),)),
            patch("app.tui.palette.check_path_safe") as safe,
        ):
            safe.return_value.read_text.return_value = "{}"
            for name in THEME_NAMES:
                create_theme_adapter(name, "colors.json").resolve()
            for name in THEME_NAMES[:3]:
                create_theme_adapter(name).resolve()

    def test_system_delegates_and_only_initial_failure_falls_back(self):
        class Source(ThemeAdapter):
            refresh_interval = 0.25

            def resolve(self):
                return BundledThemeAdapter("light").resolve()

        source = Source()
        with patch("app.tui.themes.SYSTEM_ADAPTERS", (lambda: None, lambda: source)):
            adapter = create_theme_adapter("system")
            self.assertFalse(adapter.resolve().dark)
            self.assertEqual(adapter.refresh_interval, 0.25)
            with patch.object(source, "resolve", side_effect=ValueError("bad theme")):
                self.assertIn("bad theme", resolve_initial_theme(adapter).notice)
                with self.assertRaisesRegex(ValueError, "bad theme"):
                    adapter.resolve()
