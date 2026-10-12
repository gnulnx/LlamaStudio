"""Real upstream download and native loader smoke test from an installed wheel.

Opt in with LLAMASTUDIO_INFERENCE_E2E=1; downloads a small CPU bundle into a
pytest temporary directory. The same macOS bundle also supports Metal.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LLAMASTUDIO_INFERENCE_E2E") != "1",
    reason="Opt-in native Linux/macOS managed engine download",
)


def test_wheel_cli_installs_and_uses_native_server(tmp_path):
    env = dict(os.environ)
    env.pop("LLAMASTUDIO_LLAMA_SERVER_BIN", None)
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("DYLD_LIBRARY_PATH", None)
    env["LLAMASTUDIO_NO_UPDATE_CHECK"] = "1"
    env["LLAMASTUDIO_CONFIG_DIR"] = str(tmp_path / "config")
    for setting, filename in (
        ("MODEL_PROFILES_FILE", "profiles.json"),
        ("MODEL_SETTINGS_FILE", "legacy.json"),
        ("CONVERSATIONS_FILE", "conversations.json"),
        ("LOG_DIR", "logs"),
    ):
        env[f"LLAMASTUDIO_{setting}"] = str(tmp_path / filename)
    lls = str(Path(sys.executable).parent / "lls")

    def run(*args):
        result = subprocess.run(
            [lls, *args], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    output = run("setup", "--backend", "cpu", "--repair", "--yes")
    assert "Inference is ready" in output
    diagnosis = json.loads(run("doctor", "--json"))
    assert diagnosis["usable"]
    assert str(tmp_path) in diagnosis["binary"]
    assert "11565" in diagnosis["version"]
    assert "Keeping working" in run("setup", "--yes")
