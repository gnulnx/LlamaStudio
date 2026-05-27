import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tools import check_path_safe, list_dir


class TestWorkspaceSandboxing(unittest.TestCase):
    def test_default_sandboxing_enabled(self):
        """Test default sandboxing restricts to the default repository root."""
        with (
            patch("app.tools.config_loader.sandbox_disabled", return_value=False),
            patch("app.tools.config_loader.get_workspace_root", return_value=str(Path.cwd())),
        ):
            # Relative path inside workspace should be safe
            safe_path = check_path_safe("hello.txt")
            self.assertTrue(safe_path.name == "hello.txt")

            # Path outside workspace should raise ValueError
            with self.assertRaises(ValueError):
                check_path_safe("/etc/passwd")

            with self.assertRaises(ValueError):
                check_path_safe("../outside.txt")

    def test_custom_workspace_root(self):
        """Test sandboxing respects custom WORKSPACE_ROOT setting."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir).resolve()
            with (
                patch("app.tools.config_loader.get_workspace_root", return_value=str(tmp_path)),
                patch("app.tools.config_loader.sandbox_disabled", return_value=False),
            ):
                # Inside new workspace root should pass
                safe_path = check_path_safe("inside.txt")
                self.assertEqual(safe_path, tmp_path / "inside.txt")

                # Outside new workspace root should fail
                with self.assertRaises(ValueError):
                    check_path_safe("/etc/passwd")

    def test_disable_sandbox(self):
        """Test sandboxing is bypassed completely when DISABLE_SANDBOX is True."""
        with patch("app.tools.config_loader.sandbox_disabled", return_value=True):
            # Paths outside repository should not raise ValueError
            safe_path = check_path_safe("/etc/passwd")
            self.assertEqual(safe_path, Path("/etc/passwd").resolve())

            safe_path = check_path_safe("../outside.txt")
            self.assertEqual(safe_path, Path("../outside.txt").resolve())

    def test_list_dir_with_sandbox_disabled(self):
        """Test list_dir fallback relative paths when listing outside the workspace."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir).resolve()
            # Create a file inside
            test_file = tmp_path / "test_file.txt"
            test_file.touch()

            # Enable DISABLE_SANDBOX
            with (
                patch("app.tools.config_loader.sandbox_disabled", return_value=True),
                patch(
                    "app.tools.config_loader.get_workspace_root",
                    return_value="/nonexistent_workspace_root_path",
                ),
            ):
                result = list_dir(str(tmp_path))
                self.assertIn("test_file.txt", result)


if __name__ == "__main__":
    unittest.main()
