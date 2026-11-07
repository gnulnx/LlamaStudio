"""Browser -> real API -> real download/file -> engine process -> image retry.

RUN_BROWSER_TESTS=1 pytest tests/test_browser_workflows.py --basetemp=.runtime/browser-tests
Only the external Hub and inference engine are fixtures. No application route,
download method, model loader, chat handler, DOM renderer, or browser fetch is mocked.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, Response

from app.tools import check_path_safe

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BROWSER_TESTS") != "1", reason="Run by the browser integration CI job"
)
PROJECTOR = b"GGUF" + bytes(range(256)) * 4096
FILENAME = "mmproj-fixture-f16.gguf"


@contextlib.contextmanager
def serve(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    runner = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    thread = threading.Thread(target=runner.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not runner.started:
            assert thread.is_alive() and time.monotonic() < deadline, "Test server failed to start"
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        runner.should_exit = True
        thread.join(timeout=10)
        sock.close()
        assert not thread.is_alive(), "Test server failed to shut down"


@pytest.fixture
def studio(tmp_path, monkeypatch):
    from app import model_manager
    from app.chat import chat
    from app.config import settings
    from app.config_store import config_loader
    from app.main import app

    root = check_path_safe(str(tmp_path))
    image = root / "attached.png"
    image.write_bytes(check_path_safe("app/static/social-card.png").read_bytes())
    model_root = root / 'models & "quoted"'
    model = model_root / "fixture" / "vision-GGUF" / "vision-Q4_K_M.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"GGUF model fixture")
    events = root / "engine-events.jsonl"
    release = root / "release-inference"
    release.touch()
    startup = root / "release-startup"
    startup.touch()
    engine = root / "llama-server"
    engine.write_text(
        f"#!{sys.executable}\n" + check_path_safe("tests/fixtures/llama_server.py").read_text()
    )
    engine.chmod(0o700)
    monkeypatch.setenv("LLAMASTUDIO_LLAMA_SERVER_BIN", str(engine))
    monkeypatch.setenv("LLS_TEST_ENGINE_EVENTS", str(events))
    monkeypatch.setenv("LLS_TEST_INFERENCE_RELEASE", str(release))
    monkeypatch.setenv("LLS_TEST_STARTUP_RELEASE", str(startup))
    monkeypatch.setattr(settings, "CONVERSATIONS_FILE", str(root / "conversations.json"))
    monkeypatch.setattr(settings, "LOG_DIR", str(root / "logs"))
    monkeypatch.setattr(settings, "MODEL_DIRS", [str(model_root)])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        monkeypatch.setattr(settings, "LLAMA_SERVER_PORT", sock.getsockname()[1])
    for attr, path in (
        ("config_dir", root / "config"),
        ("config_file", root / "config" / "config.json"),
        ("model_profiles_file", root / "config" / "profiles.json"),
        ("legacy_model_settings_file", root / "config" / "legacy.json"),
    ):
        monkeypatch.setattr(config_loader, attr, path)
    monkeypatch.setattr(chat, "_conversations", {})
    monkeypatch.setattr(chat, "_active_id", "")
    monkeypatch.setattr(model_manager, "_model_cache", None)
    config_loader.ensure_initialized(workspace_root=root)

    hub = FastAPI()
    transfers = []
    hub_state = {"download_status": 200}

    @hub.get("/api/models/fixture/vision-GGUF")
    def metadata():
        return {
            "id": "fixture/vision-GGUF",
            "siblings": [
                {"rfilename": model.name, "size": model.stat().st_size},
                {"rfilename": FILENAME, "size": len(PROJECTOR)},
            ],
        }

    @hub.get("/fixture/vision-GGUF/resolve/main/{filename}")
    def download(filename: str):
        transfers.append(filename)
        assert filename == FILENAME
        return Response(
            PROJECTOR,
            media_type="application/octet-stream",
            status_code=hub_state["download_status"],
        )

    @hub.get("/{path:path}")
    def other(path: str):
        return PlainTextResponse(f"Fixture Hub has no resource {path}", status_code=404)

    with serve(hub) as hub_url:
        original_client = httpx.AsyncClient

        class HubTransport(httpx.AsyncHTTPTransport):
            async def handle_async_request(self, request):
                if request.url.host != "huggingface.co":
                    return await super().handle_async_request(request)
                url = httpx.URL(hub_url).copy_with(raw_path=request.url.raw_path)
                redirected = httpx.Request(request.method, url, headers=request.headers)
                return await super().handle_async_request(redirected)

        class HubClient(original_client):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("transport", HubTransport())
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", HubClient)
        with serve(app) as url:
            params = {"cpu_mode": True, "ctx_size": 2048, "threads": 2, "gpu_layers": 0}
            response = httpx.post(
                f"{url}/api/models/load",
                json={"path": str(model), "settings": dict(params, mmproj=None)},
                timeout=15,
            )
            assert response.status_code == 200, response.text
            yield {
                "url": url,
                "model": model,
                "events": events,
                "transfers": transfers,
                "projector": model.parent / FILENAME,
                "params": params,
                "image": image,
                "release": release,
                "startup": startup,
                "hub_state": hub_state,
            }


@pytest.mark.parametrize("restore_from_history", [False, True], ids=["streamed", "saved"])
def test_missing_projector_button_downloads_attaches_and_retries_image(
    studio, restore_from_history
):
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        )
        page = browser.new_page()
        dialogs, errors = [], []
        page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(studio["url"] + "/?view=chat", wait_until="networkidle")
        if page.locator("#inferenceSetupCard").is_visible():
            page.get_by_role("button", name="Later", exact=True).click()
        page.locator("#chatImageInput").set_input_files(str(studio["image"]))
        page.locator("#chatInput").fill("Describe the attached image.")
        page.locator("#sendMsgBtn").click()
        recovery = page.get_by_role("button", name="Download projector", exact=False)
        expect(recovery).to_be_visible(timeout=15000)
        assert not studio["projector"].exists()
        if restore_from_history:
            page.reload(wait_until="networkidle")
            expect(recovery).to_be_visible()
        studio["release"].unlink()
        recovery.click()
        try:
            expect(page.locator("#stopMsgBtn")).to_be_visible(timeout=15000)
            expect(page.locator("#processLoaderOverlay")).not_to_be_visible()
            expect(page.locator("#messagesList")).not_to_contain_text("IMAGE_RECOVERY_OK")
        finally:
            studio["release"].touch()
        expect(page.locator("#messagesList")).to_contain_text("IMAGE_RECOVERY_OK", timeout=30000)
        assert studio["transfers"] == [FILENAME]
        assert (
            hashlib.sha256(studio["projector"].read_bytes()).digest()
            == hashlib.sha256(PROJECTOR).digest()
        )
        assert not studio["projector"].with_name(FILENAME + ".tmp").exists()
        events = [json.loads(line) for line in studio["events"].read_text().splitlines()]
        launches = [event for event in events if event["type"] == "launch"]
        assert len(launches) == 2
        assert "--mmproj" not in launches[0]["args"]
        attached = launches[1]["args"]
        assert attached[attached.index("--mmproj") + 1] == str(studio["projector"])
        assert attached[attached.index("--ctx-size") + 1] == "2048"
        completions = [event["payload"] for event in events if event["type"] == "completion"]
        assert len(completions) == 1
        images = [
            part
            for message in completions[0]["messages"]
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image_url"
        ]
        assert len(images) == 1
        image = images[0]["image_url"]["url"].split(",", 1)[1]
        assert base64.b64decode(image) == studio["image"].read_bytes()
        assert not dialogs, dialogs
        assert not errors, errors
        browser.close()


def test_load_button_prepares_vision_before_the_first_image_prompt(studio):
    from playwright.sync_api import expect, sync_playwright

    assert httpx.post(studio["url"] + "/api/models/eject").status_code == 200
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        )
        page = browser.new_page()
        dialogs = []
        page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
        page.goto(studio["url"] + "/?view=models", wait_until="networkidle")
        if page.locator("#inferenceSetupCard").is_visible():
            page.get_by_role("button", name="Later", exact=True).click()
        page.locator("#modelsTableBody tr").first.click()
        page.locator("#sidebarLoadBtn").click()
        expect(page.locator("#processLoaderOverlay")).not_to_be_visible(timeout=30000)
        assert studio["projector"].read_bytes() == PROJECTOR
        assert studio["transfers"] == [FILENAME]
        status = httpx.get(studio["url"] + "/api/server/status").json()
        assert status["current_params"]["mmproj"] == str(studio["projector"])
        assert status["current_params"]["ctx_size"] == 2048
        for index in range(2):
            page.locator("#chatImageInput").set_input_files(str(studio["image"]))
            page.locator("#chatInput").fill("Read this image.")
            page.locator("#sendMsgBtn").click()
            expect(page.locator("#sendMsgBtn")).to_be_visible(timeout=15000)
            replies = page.locator("#messagesList .msg.assistant")
            expect(replies).to_have_count(index + 1)
            expect(replies.nth(index)).to_contain_text("IMAGE_RECOVERY_OK")
        expect(page.locator("#messagesList")).to_contain_text("IMAGE_RECOVERY_OK")
        assert page.locator(".vision-recovery").count() == 0
        assert studio["transfers"] == [FILENAME]
        events = [json.loads(line) for line in studio["events"].read_text().splitlines()]
        assert len([event for event in events if event["type"] == "completion"]) == 2
        assert not dialogs, dialogs
        browser.close()


def test_tui_load_prepares_vision_through_the_real_backend(studio):
    from app.tui.application import StudioApp
    from app.tui.models import ModelsView

    assert httpx.post(studio["url"] + "/api/models/eject").status_code == 200
    response = httpx.post(
        studio["url"] + "/api/models/settings",
        json={"path": str(studio["model"]), "settings": dict(studio["params"], mmproj="")},
    )
    assert response.status_code == 200, response.text

    async def load_from_terminal():
        app = StudioApp(studio["url"], initial_view="models")
        async with app.run_test(size=(180, 48)) as pilot:
            for _ in range(3):
                await asyncio.wait_for(app.workers.wait_for_complete(), timeout=20)
                await pilot.pause(0.1)
            assert app.query_one(ModelsView).selected_path == str(studio["model"])
            await pilot.click("#local-load")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=20)
            await pilot.pause()
            assert app.server_status["running"]
            assert app.server_status["current_params"]["mmproj"] == str(studio["projector"])
            assert not app.query_one(ModelsView).busy

    asyncio.run(load_from_terminal())
    assert studio["projector"].read_bytes() == PROJECTOR
    assert studio["transfers"] == [FILENAME]
    events = [json.loads(line) for line in studio["events"].read_text().splitlines()]
    launch = [event for event in events if event["type"] == "launch"][-1]
    assert launch["args"][launch["args"].index("--mmproj") + 1] == str(studio["projector"])


def test_projector_download_failure_keeps_the_previous_model_and_can_retry(studio):
    studio["hub_state"]["download_status"] = 403
    response = httpx.post(
        studio["url"] + "/api/models/load",
        json={"path": str(studio["model"]), "settings": dict(studio["params"], mmproj="")},
        timeout=15,
    )
    assert response.status_code == 502
    assert "projector" in response.json()["detail"]
    assert not studio["projector"].exists()
    events = [json.loads(line) for line in studio["events"].read_text().splitlines()]
    assert len([event for event in events if event["type"] == "launch"]) == 1
    assert httpx.get(studio["url"] + "/api/server/status").json()["running"]
    studio["hub_state"]["download_status"] = 200
    response = httpx.post(
        studio["url"] + "/api/models/load",
        json={"path": str(studio["model"]), "settings": studio["params"]},
        timeout=15,
    )
    assert response.status_code == 200, response.text
    assert response.json()["multimodal"]
    assert studio["projector"].read_bytes() == PROJECTOR


def test_model_startup_keeps_the_api_responsive_and_rejects_a_second_load(studio):
    studio["startup"].unlink()
    payload = {"path": str(studio["model"]), "settings": studio["params"]}
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            httpx.post, studio["url"] + "/api/models/load", json=payload, timeout=20
        )
        try:
            deadline = time.monotonic() + 10
            while True:
                launches = [
                    json.loads(line)
                    for line in studio["events"].read_text().splitlines()
                    if json.loads(line)["type"] == "launch"
                ]
                if len(launches) == 2:
                    break
                assert time.monotonic() < deadline, "Engine never began starting"
                time.sleep(0.01)
            status = httpx.get(studio["url"] + "/api/server/status", timeout=2).json()
            assert status["is_loading"]
            assert status["current_model"] == str(studio["model"])
            assert (
                httpx.post(studio["url"] + "/api/models/load", json=payload, timeout=2).status_code
                == 409
            )
        finally:
            studio["startup"].touch()
        response = pending.result(timeout=10)
        assert response.status_code == 200, response.text
        assert response.json()["multimodal"]
