import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tools import (
    check_path_safe,
    get_effective_workspace_root,
    list_dir,
    read_file,
    reset_current_workspace_root,
    run_command,
    set_current_workspace_root,
    write_file,
)


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

    def test_task_local_workspace_root_override(self):
        """Test task-local workspace root override dynamically switches sandbox target."""
        with (
            tempfile.TemporaryDirectory() as default_dir,
            tempfile.TemporaryDirectory() as custom_dir,
        ):
            default_path = Path(default_dir).resolve()
            custom_path = Path(custom_dir).resolve()
            (default_path / "default.txt").touch()
            (custom_path / "custom.txt").touch()

            with (
                patch("app.tools.config_loader.get_workspace_root", return_value=str(default_path)),
                patch("app.tools.config_loader.sandbox_disabled", return_value=False),
            ):
                # By default, default_path is effective
                self.assertEqual(get_effective_workspace_root(), default_path)
                safe_default = check_path_safe("default.txt")
                self.assertEqual(safe_default, default_path / "default.txt")

                # Set task-local override
                token = set_current_workspace_root(custom_path)
                try:
                    self.assertEqual(get_effective_workspace_root(), custom_path)
                    safe_custom = check_path_safe("custom.txt")
                    self.assertEqual(safe_custom, custom_path / "custom.txt")

                    # Attempting to access default.txt now fails because it's outside custom_path
                    with self.assertRaises(ValueError):
                        check_path_safe(str(default_path / "default.txt"))

                    # list_dir lists within custom_path
                    listing = list_dir(".")
                    self.assertIn("custom.txt", listing)
                    self.assertNotIn("default.txt", listing)

                    # run_command runs inside custom_path
                    cmd_res = run_command("pwd")
                    self.assertIn(str(custom_path), cmd_res)
                finally:
                    reset_current_workspace_root(token)

                # After reset, returns to default_path
                self.assertEqual(get_effective_workspace_root(), default_path)
                self.assertEqual(check_path_safe("default.txt"), default_path / "default.txt")

    def test_sandbox_rejects_prefix_siblings_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            sibling = Path(tmp_dir) / "workspace-other"
            workspace.mkdir()
            sibling.mkdir()
            outside_file = sibling / "secret.txt"
            outside_file.write_text("keep this content", encoding="utf-8")
            (workspace / "escape").symlink_to(sibling, target_is_directory=True)

            with (
                patch("app.tools.config_loader.get_workspace_root", return_value=str(workspace)),
                patch("app.tools.config_loader.sandbox_disabled", return_value=False),
            ):
                for file_path in (
                    str(outside_file),
                    "../workspace-other/secret.txt",
                    "escape/secret.txt",
                ):
                    with self.subTest(file_path=file_path):
                        with self.assertRaises(ValueError):
                            check_path_safe(file_path)
                        self.assertIn("Permission Denied", read_file(file_path))
                        self.assertIn("Permission Denied", write_file(file_path, "overwrite"))
                self.assertEqual(outside_file.read_text(encoding="utf-8"), "keep this content")

    def test_file_tools_use_workspace_when_sandbox_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir).resolve()
            (workspace / "notes.txt").write_text("selected workspace", encoding="utf-8")
            with (
                patch("app.tools.config_loader.get_workspace_root", return_value=str(workspace)),
                patch("app.tools.config_loader.sandbox_disabled", return_value=True),
            ):
                self.assertEqual(check_path_safe("notes.txt"), workspace / "notes.txt")
                self.assertEqual(read_file("notes.txt"), "selected workspace")
                self.assertIn("Successfully wrote", write_file("created.txt", "new content"))
                self.assertEqual((workspace / "created.txt").read_text(), "new content")
                self.assertIn("notes.txt", list_dir("."))
                self.assertIn(str(workspace), run_command("pwd"))

    def test_disabled_sandbox_respects_task_local_workspace(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir).resolve()
            with (
                patch("app.tools.config_loader.get_workspace_root", return_value=str(Path.cwd())),
                patch("app.tools.config_loader.sandbox_disabled", return_value=True),
            ):
                token = set_current_workspace_root(workspace)
                try:
                    self.assertEqual(check_path_safe("inside.txt"), workspace / "inside.txt")
                    self.assertEqual(check_path_safe("/etc/passwd"), Path("/etc/passwd").resolve())
                finally:
                    reset_current_workspace_root(token)


if __name__ == "__main__":
    unittest.main()
