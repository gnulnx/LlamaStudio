"""Launch lifecycle checks: no browser, no backend shutdown, useful CLI failures."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from app.cli import cli, start_server_background


class TestTuiCommand(unittest.TestCase):
    def terminal(self):
        return patch(
            "app.cli.sys",
            SimpleNamespace(
                stdin=SimpleNamespace(isatty=lambda: True),
                stdout=SimpleNamespace(isatty=lambda: True),
            ),
        )

    def test_reuses_existing_backend_without_start_or_browser(self):
        with (
            self.terminal(),
            patch("app.cli.is_server_online", return_value=True),
            patch("app.cli.wait_for_server_ready", return_value=True),
            patch("app.cli.start_server_background") as start,
            patch("app.cli.webbrowser.open") as browser,
            patch("app.cli.StudioApp") as application,
        ):
            result = CliRunner().invoke(cli, ["tui", "--view", "chat"])
        self.assertEqual(result.exit_code, 0, result.output)
        start.assert_not_called()
        browser.assert_not_called()
        self.assertEqual(application.call_args.kwargs["initial_view"], "chat")
        application.return_value.run.assert_called_once()

    def test_starts_backend_without_opening_browser(self):
        with (
            self.terminal(),
            patch("app.cli.is_server_online", return_value=False),
            patch("app.cli.config_loader.initialize_for_launch"),
            patch("app.cli.start_server_background", return_value=True) as start,
            patch("app.cli.StudioApp"),
        ):
            result = CliRunner().invoke(cli, ["tui"])
        self.assertEqual(result.exit_code, 0, result.output)
        start.assert_called_once_with(open_browser=False)

    def test_no_start_fails_without_launching(self):
        with (
            self.terminal(),
            patch("app.cli.is_server_online", return_value=False),
            patch("app.cli.start_server_background") as start,
        ):
            result = CliRunner().invoke(cli, ["tui", "--no-start"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("offline", result.output)
        start.assert_not_called()

    def test_non_terminal_fails_before_any_launch(self):
        with patch("app.cli.start_server_background") as start:
            result = CliRunner().invoke(cli, ["tui"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("interactive terminal", result.output)
        start.assert_not_called()

    def test_occupied_non_studio_port_is_not_replaced(self):
        with (
            self.terminal(),
            patch("app.cli.is_server_online", return_value=True),
            patch("app.cli.wait_for_server_ready", return_value=False),
            patch("app.cli.start_server_background") as start,
        ):
            result = CliRunner().invoke(cli, ["tui"])
        self.assertNotEqual(result.exit_code, 0)
        start.assert_not_called()

    def test_browser_suppression_reaches_child_environment(self):
        with (
            patch("app.cli.subprocess.Popen") as spawn,
            patch("app.cli.wait_for_server_ready", return_value=True),
        ):
            self.assertTrue(start_server_background(open_browser=False))
        self.assertEqual(spawn.call_args.kwargs["env"]["LLAMASTUDIO_OPEN_BROWSER"], "0")
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])

    def test_screenshot_path_uses_workspace_validator(self):
        with patch("app.cli.check_path_safe", side_effect=ValueError("outside workspace")) as safe:
            result = CliRunner().invoke(cli, ["tui", "--screenshot", "../outside.svg"])
        self.assertNotEqual(result.exit_code, 0)
        safe.assert_called_once_with("../outside.svg")
        self.assertIn("outside workspace", result.output)

    def test_custom_palette_reaches_app_and_is_loaded_before_backend_start(self):
        with (
            self.terminal(),
            patch("app.cli.load_palette") as palette,
            patch("app.cli.is_server_online", return_value=True),
            patch("app.cli.wait_for_server_ready", return_value=True),
            patch("app.cli.StudioApp") as application,
        ):
            result = CliRunner().invoke(cli, ["tui", "--palette", "colors.json"])
        self.assertEqual(result.exit_code, 0, result.output)
        palette.assert_called_once_with("colors.json")
        self.assertEqual(application.call_args.kwargs["palette_path"], "colors.json")
        self.assertIs(application.call_args.kwargs["palette"], palette.return_value)

    def test_invalid_palette_fails_without_starting_backend(self):
        for error in (ValueError("unknown color role"), OSError("cannot read palette")):
            with (
                self.subTest(error=error),
                self.terminal(),
                patch("app.cli.load_palette", side_effect=error),
                patch("app.cli.start_server_background") as start,
            ):
                result = CliRunner().invoke(cli, ["tui", "--palette", "colors.json"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--palette", result.output)
            self.assertIn(str(error), result.output)
            start.assert_not_called()


class TestStatusMemory(unittest.TestCase):
    def test_status_handles_measured_unknown_and_legacy_memory(self):
        for memory, expected in (
            ({"total_vram": 24, "free_vram": 18}, "18.00 GiB / 24.00 GiB free"),
            ({"total_vram": 24, "free_vram": None}, "usage unavailable"),
            ({"total_vram": None, "free_vram": None}, "memory unavailable"),
            ({"vram": 24}, "24.00 GiB total; usage unavailable"),
        ):
            with (
                self.subTest(memory=memory),
                patch("app.cli.is_server_online", return_value=True),
                patch(
                    "app.cli.httpx.get",
                    side_effect=[
                        SimpleNamespace(json=lambda: {"running": False}),
                        SimpleNamespace(json=lambda: {"name": "GPU", **memory}),
                    ],
                ),
            ):
                result = CliRunner().invoke(cli, ["status"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn(expected, result.output)
