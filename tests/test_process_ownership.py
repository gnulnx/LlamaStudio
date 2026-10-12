"""Cleanup must not signal unrelated listeners or connected clients."""

import socket
import subprocess
import sys
from unittest.mock import patch

from app.config import settings
from app.server_manager import ServerManager


def test_ejection_preserves_unowned_listener_and_client(monkeypatch):
    listener = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            """
import socket, time
s = socket.socket()
s.bind(('127.0.0.1', 0))
s.listen()
print(s.getsockname()[1], flush=True)
connection, _ = s.accept()
time.sleep(60)
""",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    client = None
    try:
        port = int(listener.stdout.readline())
        monkeypatch.setattr(settings, "LLAMA_SERVER_PORT", port)
        client = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                f"""
import socket, time
s = socket.create_connection(('127.0.0.1', {port}))
print('connected', flush=True)
time.sleep(60)
""",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert client.stdout.readline().strip() == "connected"
        manager = ServerManager()
        assert manager._port_in_use()
        assert manager.eject_model() is False
        assert listener.poll() is None
        assert client.poll() is None
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    finally:
        for process in (client, listener):
            if process is not None:
                process.terminate()
                process.wait(timeout=5)


def test_ejection_stops_owned_process():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    manager = ServerManager()
    manager._process = process
    try:
        assert manager.eject_model() is True
        assert process.poll() is not None
        assert manager._process is None
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_load_reports_unowned_port_without_starting_or_stopping_processes(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    manager = ServerManager()
    with (
        patch.object(manager, "_port_in_use", return_value=True),
        patch("app.model_manager.find_mmproj", return_value=""),
        patch("app.server_manager.subprocess.Popen") as spawn,
        patch.object(manager, "eject_model") as eject,
    ):
        assert manager.load_model(str(model), {}) is False
        assert "owned by another process" in manager.last_error
        spawn.assert_not_called()
        eject.assert_not_called()
