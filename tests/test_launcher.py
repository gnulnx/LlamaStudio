import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.launch_context import browser_host
from app.launcher import app_url, open_browser, select_launch_view


class TestLauncherRouting(unittest.TestCase):
    def test_any_interface_opens_loopback_for_microphone_secure_context(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(browser_host("0.0.0.0"), "127.0.0.1")

    def test_app_url_includes_launch_view_when_supplied(self):
        with patch.dict(os.environ, {"LLAMASTUDIO_BROWSER_HOST": "b2"}):
            self.assertEqual(app_url("discover"), "http://b2:8765/?view=discover")

    def test_select_launch_view_uses_config_loader_and_model_scan(self):
        with (
            patch("app.launcher.scan_models", return_value=[object()]),
            patch("app.launcher.config_loader.get_launch_view", return_value="models") as mock_view,
        ):
            view = select_launch_view()

        self.assertEqual(view, "models")
        mock_view.assert_called_once_with(
            model_loaded=False,
            models_available=True,
            consume_first_launch=True,
        )

    def test_open_browser_uses_selected_launch_view(self):
        with (
            patch("app.launcher.time.sleep"),
            patch("app.launcher.select_launch_view", return_value="discover"),
            patch("app.launcher.should_open_browser", return_value=True),
            patch.dict(os.environ, {"LLAMASTUDIO_BROWSER_HOST": "b2"}),
            patch("app.launcher.webbrowser.open") as mock_open,
        ):
            open_browser()

        mock_open.assert_called_once_with("http://b2:8765/?view=discover")

    def test_open_browser_skips_headless_session(self):
        with (
            patch("app.launcher.time.sleep"),
            patch("app.launcher.select_launch_view", return_value="discover"),
            patch("app.launcher.should_open_browser", return_value=False),
            patch("app.launcher.webbrowser.open") as mock_open,
        ):
            open_browser()

        mock_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
