"""A checkout in the workspace must not replace the CLI's backend package."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from app import cli


@pytest.mark.parametrize("injected_path", [False, True])
def test_backend_launch_uses_cli_package_and_preserves_workspace(
    tmp_path, monkeypatch, injected_path
):
    selected = tmp_path / "selected package's path"
    package = selected / "app"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "launcher.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "print(json.dumps({'launcher': str(Path(__file__).resolve()), 'workspace': os.getcwd()}))\n"
    )
    workspace = tmp_path / "older checkout"
    shadow = workspace / "app"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("")
    (shadow / "launcher.py").write_text("raise RuntimeError('Imported the older checkout')\n")
    monkeypatch.setattr(cli, "__file__", str(package / "cli.py"))
    env = os.environ.copy()
    env.pop("PYTHONSAFEPATH", None)
    if injected_path:
        env["PYTHONPATH"] = str(workspace)
    else:
        env.pop("PYTHONPATH", None)
    result = subprocess.run(
        cli.server_launch_command(),
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "launcher": str(Path(package / "launcher.py").resolve()),
        "workspace": str(workspace),
    }
