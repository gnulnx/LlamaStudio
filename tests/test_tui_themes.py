"""Theme source contracts, fallback, override precedence and bundled contrast."""

import asyncio
import json
import subprocess
import unittest
from unittest.mock import patch

from test_tui_palette import contrast

from app.tui.application import StudioApp
from app.tui.palette import load_palette
from app.tui.themes import (
    BUNDLED_THEME_NAMES,
    THEME_NAMES,
    BundledThemeAdapter,
    GtkThemeAdapter,
    MacOSThemeAdapter,
    ThemeAdapter,
    create_theme_adapter,
    detect_gtk_theme,
    detect_macos_theme,
    resolve_initial_theme,
)


class TestThemes(unittest.TestCase):
    def test_polling_source_can_be_constructed_before_event_loop_starts(self):
        source = MacOSThemeAdapter()
        with patch.object(source, "is_dark", return_value=True):
            app = StudioApp("http://studio", theme_name="system", theme_adapter=source)
        self.assertIsNone(app._theme_timer)
        self.assertTrue(app.resolved_theme.dark)
        asyncio.run(app.client.close())

    def test_gtk_preference_precedence_and_legacy_theme_fallback(self):
        for scheme, theme, dark in (
            ("prefer-dark", "Pop", True),
            ("prefer-light", "Pop-dark", False),
            ("default", "Pop-dark", True),
            ("default", "Adwaita", False),
            (None, "Adwaita-dark", True),
            (None, "Pop", False),
        ):

            def read(command, **kwargs):
                if command[-1] == "color-scheme":
                    return subprocess.CompletedProcess(
                        command,
                        1 if scheme is None else 0,
                        f"'{scheme}'",
                        "No such key “color-scheme”" if scheme is None else "",
                    )
                return subprocess.CompletedProcess(command, 0, f"'{theme}'", "")

            with (
                self.subTest(scheme=scheme, theme=theme),
                patch("app.tui.themes.subprocess.run", side_effect=read) as run,
            ):
                resolved = GtkThemeAdapter("gsettings").resolve()
                self.assertEqual(resolved.dark, dark)
                self.assertEqual(
                    run.call_count, 1 if scheme in ("prefer-dark", "prefer-light") else 2
                )
                self.assertEqual(run.call_args.kwargs["timeout"], 2)

    def test_macos_dark_light_absent_key_and_failure(self):
        for status, output, error, dark in (
            (0, "Dark\n", "", True),
            (0, "Light\n", "", False),
            (
                1,
                "",
                "The domain/default pair of (kCFPreferencesAnyApplication, AppleInterfaceStyle) does not exist",
                False,
            ),
            (1, "", "Permission denied", None),
            (0, "unexpected", "", None),
        ):
            with (
                self.subTest(output=output, error=error),
                patch(
                    "app.tui.themes.subprocess.run",
                    return_value=subprocess.CompletedProcess([], status, output, error),
                ) as run,
            ):
                if dark is None:
                    with self.assertRaises(ValueError):
                        MacOSThemeAdapter().resolve()
                else:
                    self.assertEqual(MacOSThemeAdapter().resolve().dark, dark)
                self.assertEqual(
                    run.call_args.args[0],
                    ["/usr/bin/defaults", "read", "-g", "AppleInterfaceStyle"],
                )

    def test_native_failure_never_silently_switches_to_light(self):
        for adapter in (GtkThemeAdapter("gsettings"), MacOSThemeAdapter()):
            for error in (FileNotFoundError(), subprocess.TimeoutExpired("settings", 2)):
                with (
                    self.subTest(adapter=adapter, error=error),
                    patch("app.tui.themes.subprocess.run", side_effect=error),
                    self.assertRaises(ValueError),
                ):
                    adapter.resolve()
        with (
            patch(
                "app.tui.themes.subprocess.run",
                return_value=subprocess.CompletedProcess([], 1, "", "DBus unavailable"),
            ),
            self.assertRaises(ValueError),
        ):
            GtkThemeAdapter("gsettings").resolve()

    def test_platform_detection_does_not_use_installed_gnome_on_other_desktops(self):
        with (
            patch("app.tui.themes.sys.platform", "linux"),
            patch("app.tui.themes.shutil.which", return_value="/usr/bin/gsettings"),
        ):
            for desktop in ("", "KDE", "sway"):
                with patch.dict("os.environ", {"XDG_CURRENT_DESKTOP": desktop}):
                    self.assertIsNone(detect_gtk_theme())
            with patch.dict("os.environ", {"XDG_CURRENT_DESKTOP": "pop:GNOME"}):
                self.assertIsInstance(detect_gtk_theme(), GtkThemeAdapter)
                self.assertIsNone(detect_macos_theme())
        with patch("app.tui.themes.sys.platform", "darwin"):
            self.assertIsInstance(detect_macos_theme(), MacOSThemeAdapter)
            self.assertIsNone(detect_gtk_theme())

    def test_native_adapter_follows_mode_changes_and_reuses_unchanged_palette(self):
        adapter = MacOSThemeAdapter()
        with patch.object(adapter, "is_dark", side_effect=[True, True, False]):
            first = adapter.resolve()
            self.assertIs(adapter.resolve(), first)
            self.assertFalse(adapter.resolve().dark)

    def test_default_is_unchanged_and_system_fallback_is_explicit_and_idle(self):
        default = create_theme_adapter().resolve()
        self.assertEqual(default.palette, load_palette())
        self.assertTrue(default.dark)
        system = create_theme_adapter("system")
        with patch("app.tui.themes.SYSTEM_ADAPTERS", ()):
            fallback = system.resolve()
        self.assertEqual(fallback.palette, default.palette)
        self.assertIn("using Default", fallback.notice)
        self.assertIsNone(system.refresh_interval)

    def test_bundled_modes_and_contrast(self):
        for name in BUNDLED_THEME_NAMES:
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
            for name in BUNDLED_THEME_NAMES:
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
