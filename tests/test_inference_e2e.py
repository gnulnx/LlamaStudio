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
from test_update_e2e import terminal

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

    # Exercise the installed TUI command, its real HTTP setup API and upstream
    # download. Only hardware choice and the final interactive screen are stubbed.
    env.pop("CI", None)
    env["LLAMASTUDIO_CONFIG_DIR"] = str(tmp_path / "tui-config")
    env["LLAMASTUDIO_MODEL_DIRS"] = "[]"
    script = """
import os, socket, threading, time
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    os.environ['LLAMASTUDIO_APP_PORT'] = str(sock.getsockname()[1])
os.environ['LLAMASTUDIO_APP_HOST'] = '127.0.0.1'
from app import config, inference_setup, inference_runtime
config.resolve_llama_server_bin = lambda *a, **kw: (
    inference_runtime.managed_binary() or '/missing/test/llama-server'
)
inference_setup.recommended_backend = lambda: 'cpu'
import uvicorn
from app.main import app
server = uvicorn.Server(uvicorn.Config(
    app, host='127.0.0.1', port=config.settings.APP_PORT, log_level='error'
))
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
try:
    deadline = time.monotonic() + 20
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline
        time.sleep(0.05)
    from app.cli import cli, StudioApp
    StudioApp.run = lambda self: print('TUI_READY')
    cli(['--no-update-check', 'tui', '--no-start'], standalone_mode=False)
    assert inference_setup.diagnose().usable
finally:
    server.should_exit = True
    thread.join(timeout=10)
    assert not thread.is_alive()
"""
    output = terminal([sys.executable, "-c", script], cwd=tmp_path, env=env, answer="y")
    assert "Set up local inference now?" in output
    assert "Local inference is ready." in output
    assert "TUI_READY" in output
