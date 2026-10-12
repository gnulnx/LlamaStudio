"""Keep ordinary tests away from user configuration and live service ports."""

import atexit
import os
import shutil
import socket
import tempfile

import pytest

if os.environ.get("RUN_LOCAL_ONLY") != "1":
    _test_home = tempfile.mkdtemp(prefix="llamastudio-tests-")
    atexit.register(shutil.rmtree, _test_home, ignore_errors=True)
    for key, name in (
        ("CONFIG_DIR", "config"),
        ("MODEL_PROFILES_FILE", "profiles.json"),
        ("MODEL_SETTINGS_FILE", "legacy.json"),
        ("CONVERSATIONS_FILE", "conversations.json"),
        ("LOG_DIR", "logs"),
        ("SPEECH_DIR", "speech"),
    ):
        os.environ[f"LLAMASTUDIO_{key}"] = os.path.join(_test_home, name)
    # Reserve distinct ports until all three have been selected.
    _sockets = []
    for key in ("APP_PORT", "LLAMA_SERVER_PORT", "SPEECH_SERVER_PORT"):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        _sockets.append(sock)
        os.environ[f"LLAMASTUDIO_{key}"] = str(sock.getsockname()[1])
    for sock in _sockets:
        sock.close()


@pytest.fixture(autouse=True)
def isolated_server_manager():
    """Tests sharing the singleton must not inherit another test's process mocks."""
    if os.environ.get("RUN_LOCAL_ONLY") == "1":
        yield
        return
    from app.server_manager import ServerManager

    manager = ServerManager()
    saved = manager.__dict__.copy()
    manager._process = None
    manager._current_model = None
    manager._current_model_name = None
    manager._current_params = {}
    manager._is_loading = False
    manager._last_error = None
    yield
    manager.__dict__.clear()
    manager.__dict__.update(saved)
