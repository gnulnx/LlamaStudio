"""Opt-in native model/tool tests, isolated from the user's running application."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_ONLY") != "1", reason="Opt-in local model matrix: tests/test_all.sh"
)
ROOT = Path(__file__).resolve().parent.parent
MODELS = json.loads((Path(__file__).parent / "local_test_models.json").read_text())


@pytest.fixture
def local_app(tmp_path):
    from app.config_store import config_loader

    env = dict(os.environ)
    env.update(
        LLAMASTUDIO_CONFIG_DIR=str(tmp_path / "config"),
        LLAMASTUDIO_CONFIG_FILE=str(tmp_path / "config/config.json"),
        LLAMASTUDIO_MODEL_PROFILES_FILE=str(tmp_path / "profiles.json"),
        LLAMASTUDIO_MODEL_SETTINGS_FILE=str(tmp_path / "legacy.json"),
        LLAMASTUDIO_CONVERSATIONS_FILE=str(tmp_path / "conversations.json"),
        LLAMASTUDIO_LOG_DIR=str(tmp_path / "logs"),
        LLAMASTUDIO_SPEECH_DIR=str(tmp_path / "speech"),
        LLAMASTUDIO_WORKSPACE_ROOT=str(tmp_path),
        LLAMASTUDIO_MODEL_DIRS=json.dumps(config_loader.get_model_directories()),
        LLAMASTUDIO_OPEN_BROWSER="0",
        LLAMASTUDIO_NO_UPDATE_CHECK="1",
        PYTHONPATH=str(ROOT),
    )
    sockets = []
    try:
        for key in ("APP_PORT", "LLAMA_SERVER_PORT", "SPEECH_SERVER_PORT"):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            env[f"LLAMASTUDIO_{key}"] = str(sock.getsockname()[1])
    finally:
        for sock in sockets:
            sock.close()
    url = f"http://127.0.0.1:{env['LLAMASTUDIO_APP_PORT']}"
    with (tmp_path / "backend.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "app.launcher"],
            cwd=tmp_path,
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 15
            while True:
                assert process.poll() is None, (tmp_path / "backend.log").read_text()
                try:
                    if httpx.get(url + "/api/server/status", timeout=0.5).status_code == 200:
                        break
                except httpx.RequestError:
                    pass
                assert time.monotonic() < deadline, "Isolated backend failed to start"
                time.sleep(0.05)
            yield url, env
        finally:
            # This group was created above and contains only this test's backend/engine.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


@pytest.mark.parametrize("model_query", MODELS)
def test_model_tool_calling(local_app, tmp_path, model_query):
    url, env = local_app
    models = httpx.get(url + "/api/models").json()["models"]
    model = next((m for m in models if model_query.lower() in m["name"].lower()), None)
    assert model is not None, f"Configured test model is missing: {model_query}"
    loaded = httpx.post(
        url + "/api/models/load",
        json={
            "path": model["path"],
            "settings": {"cpu_mode": True, "gpu_layers": 0, "mmproj": None},
        },
        timeout=240,
    )
    assert loaded.status_code == 200, loaded.text
    marker = "LlamaStudio-tool-test-" + uuid.uuid4().hex
    target = tmp_path / "tool-output.txt"
    command = [
        str(Path(sys.executable).parent / "lls"),
        "--no-update-check",
        "oneshot",
        "--no-thinking",
        "--max-tokens",
        "1000",
        f"Use write_file to create tool-output.txt containing exactly {marker}. Do not use a shell command.",
    ]
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert target.read_text().strip() == marker
    saved = json.loads((tmp_path / "conversations.json").read_text())
    calls = [
        call
        for conversation in saved["conversations"].values()
        for message in conversation["messages"]
        for call in message.get("tool_calls") or []
    ]
    assert any(call["function"]["name"] == "write_file" for call in calls)
