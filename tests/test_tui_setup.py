"""TUI setup uses the connected backend and preserves active model sessions."""

from __future__ import annotations

import io
from unittest.mock import Mock

import httpx
import pytest
import rich_click as click
from rich.console import Console

from app import tui_setup


def response(path, payload, code=200):
    return httpx.Response(code, json=payload, request=httpx.Request("GET", f"http://studio{path}"))


@pytest.fixture
def flow(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    output = io.StringIO()
    console = Console(file=output)
    get = Mock()
    post = Mock(return_value=response("/api/inference/setup", {"running": True}))
    confirm = Mock(return_value=True)
    restart = Mock(return_value=True)
    monkeypatch.setattr(tui_setup.httpx, "get", get)
    monkeypatch.setattr(tui_setup.httpx, "post", post)
    monkeypatch.setattr(tui_setup.click, "confirm", confirm)
    monkeypatch.setattr(tui_setup.time, "sleep", lambda seconds: None)

    def run(allow_restart=True):
        tui_setup.prepare_inference(
            "http://studio", console, allow_restart=allow_restart, restart=restart
        )

    return run, get, post, confirm, restart, output


def test_working_engine_does_not_prompt_or_install(flow):
    run, get, post, confirm, restart, _ = flow
    get.return_value = response("/api/inference/status", {"usable": True})
    run()
    confirm.assert_not_called()
    post.assert_not_called()
    restart.assert_not_called()


def test_broken_engine_offers_setup_and_waits_for_validated_engine(flow):
    run, get, post, confirm, restart, output = flow
    get.side_effect = [
        response("/api/inference/status", {"usable": False, "problem": "libcudart missing"}),
        response("/api/inference/status", {"running": True, "message": "Downloading"}),
        response("/api/inference/status", {"running": False, "usable": True}),
    ]
    run()
    confirm.assert_called_once_with("Set up local inference now?", default=True)
    post.assert_called_once_with(
        "http://studio/api/inference/setup", headers={"X-LlamaStudio-Setup": "1"}, timeout=10
    )
    restart.assert_not_called()
    assert "Local inference is ready" in output.getvalue()


def test_declining_setup_keeps_browsing_available(flow):
    run, get, post, confirm, _restart, output = flow
    get.return_value = response("/api/inference/status", {"usable": False})
    confirm.return_value = False
    run()
    post.assert_not_called()
    assert "run lls setup later" in output.getvalue()


@pytest.mark.parametrize("busy", [{"running": True}, {"is_loading": True}])
def test_stale_backend_with_active_model_is_never_restarted(flow, busy):
    run, get, post, confirm, restart, output = flow
    get.side_effect = [
        response("/api/inference/status", {}, 404),
        response("/api/server/status", busy),
    ]
    run()
    confirm.assert_not_called()
    restart.assert_not_called()
    post.assert_not_called()
    assert "When the backend is idle" in output.getvalue()


def test_no_start_prevents_stale_backend_restart(flow):
    run, get, _post, confirm, restart, _ = flow
    get.side_effect = [
        response("/api/inference/status", {}, 404),
        response("/api/server/status", {"running": False}),
    ]
    run(allow_restart=False)
    confirm.assert_not_called()
    restart.assert_not_called()


def test_idle_stale_backend_restart_requires_consent_and_waits_for_new_api(flow):
    run, get, post, confirm, restart, _ = flow
    get.side_effect = [
        response("/api/inference/status", {}, 404),
        response("/api/server/status", {"running": False}),
        response("/api/server/status", {"running": False}),
        httpx.ConnectError("Old server stopping"),
        response("/api/inference/status", {}, 404),
        response("/api/inference/status", {"usable": True}),
    ]
    run()
    confirm.assert_called_once_with(
        "Restart the idle backend to enable local inference setup?", default=True
    )
    restart.assert_called_once()
    post.assert_not_called()


def test_declining_backend_restart_preserves_it(flow):
    run, get, _post, confirm, restart, _ = flow
    get.side_effect = [
        response("/api/inference/status", {}, 404),
        response("/api/server/status", {}),
    ]
    confirm.return_value = False
    run()
    restart.assert_not_called()


def test_already_running_setup_is_observed_without_starting_another(flow):
    run, get, post, confirm, _restart, _ = flow
    get.side_effect = [
        response("/api/inference/status", {"running": True}),
        response("/api/inference/status", {"usable": True}),
    ]
    run()
    post.assert_not_called()
    confirm.assert_not_called()


def test_setup_failure_shows_actual_error(flow):
    run, get, _post, _confirm, _restart, _ = flow
    get.side_effect = [
        response("/api/inference/status", {"usable": False}),
        response("/api/inference/status", {"error": "Checksum mismatch"}),
    ]
    with pytest.raises(click.ClickException, match="Checksum mismatch"):
        run()


def test_ci_never_prompts_or_downloads(flow, monkeypatch):
    run, get, post, confirm, _restart, _ = flow
    monkeypatch.setenv("CI", "1")
    run()
    get.assert_not_called()
    post.assert_not_called()
    confirm.assert_not_called()


def test_model_started_during_restart_confirmation_is_preserved(flow):
    run, get, _post, confirm, restart, output = flow
    get.side_effect = [
        response("/api/inference/status", {}, 404),
        response("/api/server/status", {}),
        response("/api/server/status", {"running": True}),
    ]
    run()
    confirm.assert_called_once()
    restart.assert_not_called()
    assert "leaving the backend running" in output.getvalue()
