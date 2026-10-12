"""Restart a real backend on native Linux/macOS, using isolated ports/config."""

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from app import cli, launcher


def test_process_identity_without_linux_proc(monkeypatch):
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path).startswith("/proc/"):
            raise FileNotFoundError
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; print('ready',flush=True); time.sleep(15)",
            "app.launcher",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        assert launcher.is_llamastudio_pid(process.pid)
        assert not launcher.is_llamastudio_pid(os.getpid())
    finally:
        process.terminate()
        process.wait(timeout=5)


@pytest.mark.skipif(
    os.environ.get("LLAMASTUDIO_BACKEND_E2E") != "1", reason="Opt-in native backend restart"
)
@pytest.mark.parametrize("owned", [True, False])
def test_reload_replaces_live_backend_before_reporting_success(tmp_path, owned):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ)
    env.update(
        LLAMASTUDIO_CONFIG_DIR=str(tmp_path / "config"),
        LLAMASTUDIO_APP_HOST="127.0.0.1",
        LLAMASTUDIO_APP_PORT=str(port),
        LLAMASTUDIO_LLAMA_SERVER_PORT=str(port + 1),
        LLAMASTUDIO_SPEECH_SERVER_PORT=str(port + 2),
        LLAMASTUDIO_MODEL_DIRS="[]",
        LLAMASTUDIO_NO_UPDATE_CHECK="1",
        LLAMASTUDIO_OPEN_BROWSER="0",
        LLAMASTUDIO_LOG_DIR=str(tmp_path / "logs"),
        LLAMASTUDIO_MODEL_PROFILES_FILE=str(tmp_path / "profiles.json"),
        LLAMASTUDIO_CONVERSATIONS_FILE=str(tmp_path / "conversations.json"),
    )
    pid_file = tmp_path / "config" / "app.pid"
    if owned:
        command = cli.server_launch_command()
    else:
        script = (
            "import os; from pathlib import Path; "
            "from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            " def do_GET(self):\n"
            "  self.send_response(200); self.end_headers(); self.wfile.write(b'{}')\n"
            f"p=Path({str(pid_file)!r}); p.parent.mkdir(parents=True); p.write_text(str(os.getpid()))\n"
            f"ThreadingHTTPServer(('127.0.0.1',{port}),Handler).serve_forever()\n"
        )
        command = [sys.executable, "-c", script]
    startup_log = tmp_path / "initial-server.log"
    with startup_log.open("w") as output:
        first = subprocess.Popen(command, cwd=tmp_path, env=env, stdout=output, stderr=output)
    new_pid = None
    try:
        deadline = time.monotonic() + 15
        last_result = "No response"
        while time.monotonic() < deadline:
            assert first.poll() is None, startup_log.read_text()
            try:
                response = httpx.get(f"http://127.0.0.1:{port}/api/server/status", timeout=0.5)
                last_result = f"HTTP {response.status_code}, PID file exists: {pid_file.exists()}"
                if response.status_code == 200 and pid_file.exists():
                    break
            except httpx.HTTPError as exc:
                last_result = repr(exc)
            time.sleep(0.1)
        else:
            pytest.fail(
                f"Initial backend failed to start: {last_result}\n{startup_log.read_text()}"
            )
        assert int(pid_file.read_text()) == first.pid
        package_root = str(Path(cli.__file__).resolve().parent.parent)
        code = f"import sys; sys.path.insert(0,{package_root!r}); from app.cli import cli; cli(['--no-update-check','reload'])"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=35,
        )
        if not owned:
            assert result.returncode != 0, result.stdout + result.stderr
            assert "Failed to reload" in result.stdout
            assert first.poll() is None
            assert int(pid_file.read_text()) == first.pid
            return
        assert result.returncode == 0, result.stdout + result.stderr
        assert "successfully reloaded" in result.stdout
        new_pid = int(pid_file.read_text())
        assert new_pid != first.pid, "CLI reported success while old backend still served the port"
        first.wait(timeout=5)
        assert httpx.get(f"http://127.0.0.1:{port}/api/server/status").status_code == 200
        assert httpx.get(f"http://127.0.0.1:{port}/api/settings/sandbox").status_code == 200
    finally:
        if first.poll() is None:
            first.terminate()
        first.wait(timeout=5)
        if pid_file.exists():
            new_pid = int(pid_file.read_text())
        if new_pid and new_pid != first.pid:
            with contextlib.suppress(ProcessLookupError):
                os.kill(new_pid, signal.SIGTERM)
