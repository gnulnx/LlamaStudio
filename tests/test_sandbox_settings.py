"""Real persistent settings and request-local file/command permissions."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app import main, sandbox, sandbox_settings
from app.config_store import ConfigLoader
from app.sandbox import SandboxPolicy, current_policy
from app.tools import check_path_safe, list_dir, read_file, run_command, write_file


@pytest.fixture
def loader(tmp_path, monkeypatch):
    instance = ConfigLoader(config_dir=tmp_path / "config")
    workspace = tmp_path / "initial"
    workspace.mkdir()
    instance.initialize_for_launch(workspace)
    for module in (sandbox, sandbox_settings):
        monkeypatch.setattr(module, "config_loader", instance)
    return instance


def test_settings_persist_without_changing_other_configuration(loader, tmp_path):
    directory = tmp_path / "new"
    directory.mkdir()
    before = loader.load_app_config()
    client = TestClient(main.app)
    response = client.put(
        "/api/settings/sandbox",
        headers={"X-LlamaStudio-Settings": "1"},
        json={
            "root": str(directory),
            "disable_sandbox": True,
            "command_policy": "deny",
        },
    )
    assert response.status_code == 200
    restarted = ConfigLoader(config_dir=loader.config_dir)
    assert restarted.get_workspace_config() == response.json()
    after = restarted.load_app_config()
    assert {k: v for k, v in before.items() if k != "workspace"} == {
        k: v for k, v in after.items() if k != "workspace"
    }
    assert client.get("/api/settings/sandbox").json() == response.json()


@pytest.mark.parametrize(
    "fields",
    [
        {"root": "relative"},
        {"root": "\0"},
        {"root": 42},
        {"disable_sandbox": "false"},
        {"command_policy": "unknown"},
    ],
)
def test_invalid_settings_do_not_change_saved_configuration(loader, fields):
    before = loader.config_file.read_text()
    payload = {**loader.get_workspace_config(), **fields}
    response = TestClient(main.app).put(
        "/api/settings/sandbox", json=payload, headers={"X-LlamaStudio-Settings": "1"}
    )
    assert response.status_code == 400
    assert loader.config_file.read_text() == before


def test_settings_reject_cross_origin_and_missing_explicit_action(loader):
    client = TestClient(main.app)
    payload = loader.get_workspace_config()
    assert client.put("/api/settings/sandbox", json=payload).status_code == 403
    assert (
        client.put(
            "/api/settings/sandbox",
            json=payload,
            headers={
                "X-LlamaStudio-Settings": "1",
                "Origin": "https://unrelated.example",
            },
        ).status_code
        == 403
    )


def test_malformed_settings_do_not_change_saved_configuration(loader):
    before = loader.config_file.read_text()
    response = TestClient(main.app).put(
        "/api/settings/sandbox", content="{", headers={"X-LlamaStudio-Settings": "1"}
    )
    assert response.status_code == 400
    assert loader.config_file.read_text() == before


def test_saved_changes_do_not_change_an_existing_request_policy(loader, tmp_path):
    original = SandboxPolicy.capture()
    other = tmp_path / "other"
    other.mkdir()
    secret = other / "outside.txt"
    secret.write_text("outside")
    token = current_policy.set(original)
    try:
        config = loader.load_app_config()
        config["workspace"].update(root=str(other), disable_sandbox=True, command_policy="deny")
        loader.save_app_config(config)
        assert "Permission Denied" in read_file(str(secret))
        assert "wrote" in write_file("inside.txt", "original").lower()
        assert Path(original.root, "inside.txt").read_text() == "original"
        assert "inside.txt" in list_dir(".")
        assert "original" in run_command("cat inside.txt")
    finally:
        current_policy.reset(token)
    next_request = SandboxPolicy.capture()
    token = current_policy.set(next_request)
    try:
        assert check_path_safe(str(secret)) == secret.resolve()
        assert "Permission Denied" in run_command("echo should-not-run")
    finally:
        current_policy.reset(token)


def test_api_captures_policy_before_stream_is_consumed(loader, monkeypatch, tmp_path):
    monkeypatch.setattr(main.server, "_port_in_use", lambda: True)
    monkeypatch.setattr(main, "StreamingResponse", lambda events, **kwargs: events)
    stream = Mock(return_value=iter(()))
    monkeypatch.setattr(main.chat, "stream_chat", stream)
    request = SimpleNamespace(json=AsyncMock(return_value={"message": "hello"}))
    response = asyncio.run(main.send_message(request))
    original = loader.get_workspace_config()
    config = loader.load_app_config()
    config["workspace"]["command_policy"] = "deny"
    loader.save_app_config(config)
    events = list(response)
    assert json.loads(events[0][6:])["sandbox"] == original
    assert stream.call_args.kwargs["sandbox_policy"].to_dict() == original


def test_stream_tools_keep_captured_permissions_after_settings_change(loader, monkeypatch):
    from test_chat import FakeToolHttpClient

    from app.chat import chat

    (Path(loader.get_workspace_root()) / "original.txt").write_text("original")
    policy = SandboxPolicy.capture()
    monkeypatch.setattr("app.chat.httpx.Client", FakeToolHttpClient)
    monkeypatch.setattr(chat, "_save_to_disk", lambda: None)
    monkeypatch.setattr(chat, "_conversations", {})
    monkeypatch.setattr(chat, "_active_id", "")
    monkeypatch.setattr(
        "app.chat.config_loader.get_chat_defaults",
        lambda: {**loader.get_chat_defaults(), "max_tool_iterations": 1},
    )
    stream = chat.stream_chat("List files", sandbox_policy=policy)
    results = []
    for event in stream:
        data = json.loads(event[6:])
        if data.get("type") == "tool_exec_start":
            config = loader.load_app_config()
            config["workspace"].update(
                root=str(loader.config_dir), disable_sandbox=True, command_policy="deny"
            )
            loader.save_app_config(config)
        if data.get("type") == "tool_exec_end":
            results.append(data["result"])
    assert any("original.txt" in result for result in results)
    assert current_policy.get() is None
