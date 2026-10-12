"""Exercise installation boundaries and diagnostics without needing a GPU."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from app import inference_setup as engine
from app.cli import cli
from app.config import resolve_llama_server_bin
from app.inference_runtime import managed_binary, runtime_environment
from app.main import app
from app.server_manager import ServerManager


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.config_store import ConfigLoader

    monkeypatch.setenv("LLAMASTUDIO_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LLAMASTUDIO_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.setattr("app.tools.config_loader", ConfigLoader(tmp_path))
    monkeypatch.setattr(engine, "recommended_backend", lambda: "cpu")
    return tmp_path


def fake_binary(path: Path, *, failure: str = "", devices: str = "Available devices:") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if failure:
        content = f"#!/bin/sh\necho '{failure}' >&2\nexit 127\n"
    else:
        content = f"#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'version: b11565'; else echo '{devices}'; fi\n"
    path.write_text(content)
    path.chmod(0o755)
    return path


def activate(root: Path, name="existing") -> Path:
    binary = fake_binary(root / "inference" / "releases" / name / "llama-server")
    (root / "inference" / "current.json").write_text(
        json.dumps({"binary": str(binary.relative_to(root / "inference")), "backend": "cpu"})
    )
    return binary


def bundle_download(asset, target, progress):
    with tarfile.open(target, "w:gz") as archive:
        content = b"#!/bin/sh\nif [ \"$1\" = --version ]; then echo version: b11565; else echo 'Available devices:'; fi\n"
        member = tarfile.TarInfo("bundle/llama-server")
        member.size = len(content)
        member.mode = 0o755
        archive.addfile(member, io.BytesIO(content))
        content = b"private library"
        member = tarfile.TarInfo("bundle/libggml.so.1")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
        member = tarfile.TarInfo("bundle/libggml.so")
        member.type = tarfile.SYMTYPE
        member.linkname = "libggml.so.1"
        archive.addfile(member)


def test_setup_activates_verified_release_and_repair_keeps_previous(isolated, monkeypatch):
    monkeypatch.setattr(engine, "_download", bundle_download)
    monkeypatch.setattr(
        "app.config.resolve_llama_server_bin",
        lambda **kw: managed_binary() or str(isolated / "missing"),
    )
    result = engine.setup(backend="cpu")
    assert result.usable
    assert managed_binary() == result.binary
    first = Path(result.binary)
    assert (first.parent / "libggml.so").read_bytes() == b"private library"
    assert engine.diagnose().usable
    repaired = engine.setup(backend="cpu", repair=True)
    assert repaired.binary != result.binary
    assert first.exists()
    assert managed_binary() == repaired.binary


def test_working_external_server_is_preserved(isolated, monkeypatch):
    external = fake_binary(isolated / "external" / "llama-server")
    monkeypatch.setattr("app.config.resolve_llama_server_bin", lambda **kw: str(external))
    with patch.object(engine, "_download") as download:
        result = engine.setup()
    assert result.binary == str(external)
    download.assert_not_called()
    assert not (isolated / "inference").exists()


def test_failed_repair_does_not_change_manifest_or_external_files(isolated, monkeypatch):
    previous = activate(isolated)
    original = (isolated / "inference" / "current.json").read_bytes()

    def fail(*args):
        raise engine.InferenceError("SHA256 mismatch")

    monkeypatch.setattr(engine, "_download", fail)
    with pytest.raises(engine.InferenceError, match="SHA256"):
        engine.setup(backend="cpu", repair=True)
    assert managed_binary() == str(previous)
    assert (isolated / "inference" / "current.json").read_bytes() == original
    assert list(previous.parent.parent.iterdir()) == [previous.parent]


def test_explicit_override_wins_and_cannot_be_silently_replaced(isolated, monkeypatch):
    activate(isolated)
    external = isolated / "external" / "missing"
    monkeypatch.setenv("LLAMASTUDIO_LLAMA_SERVER_BIN", str(external))
    assert resolve_llama_server_bin() == str(external)
    with pytest.raises(engine.InferenceError, match="unset the override"):
        engine.setup(repair=True)


@pytest.mark.parametrize("value", ["/tmp/external", "../../outside", "releases/link/llama-server"])
def test_manifest_cannot_select_external_binary(isolated, value):
    root = isolated / "inference"
    root.mkdir()
    outside = fake_binary(isolated / "outside" / "llama-server")
    (root / "releases").mkdir()
    (root / "releases" / "link").symlink_to(outside.parent, target_is_directory=True)
    (root / "current.json").write_text(json.dumps({"binary": value}))
    assert managed_binary() is None


def test_managed_libraries_only_reach_managed_processes_including_cpu_fallback(isolated):
    binary = activate(isolated)
    (binary.parent / "libggml.so").touch()
    variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    with patch.dict(os.environ, {variable: "/custom"}):
        env = runtime_environment(str(binary))
        assert str(binary.parent) in env[variable]
        assert env[variable].endswith("/custom")
        assert runtime_environment("/external/server")[variable] == "/custom"
        manager = ServerManager()
        with patch("app.config.resolve_llama_server_bin", return_value=str(binary)):
            assert manager._build_env(force_cpu=True)[variable] == env[variable]
            with patch(
                "app.server_manager.subprocess.check_output", return_value="--load-mode"
            ) as probe:
                assert manager._supports_load_mode(str(binary))
                assert probe.call_args.kwargs["env"][variable] == env[variable]


def test_doctor_missing_library_and_no_gpu_have_actionable_failures(isolated):
    binary = fake_binary(
        isolated / "broken", failure="error while loading shared libraries: libcudart.so.12"
    )
    result = engine.diagnose(str(binary))
    assert not result.usable
    assert "libcudart.so.12" in result.problem
    assert "lls setup --repair" in result.problem
    binary = fake_binary(isolated / "cpu")
    assert not engine.diagnose(str(binary), backend="cuda").usable
    assert engine.diagnose(str(binary), backend="cpu").usable
    binary = fake_binary(isolated / "gpu", devices="Available devices:\n  CUDA0: Test GPU")
    assert engine.diagnose(str(binary), backend="cuda").usable


def test_doctor_timeout_is_actionable(isolated):
    binary = fake_binary(isolated / "server")
    with patch.object(
        engine.subprocess, "run", side_effect=subprocess.TimeoutExpired("server", 30)
    ):
        assert "lls setup --repair" in engine.diagnose(str(binary)).problem


@pytest.mark.parametrize(
    "system,arch,backend,expected",
    [
        ("Darwin", "arm64", "metal", ["macos-arm64"]),
        ("Darwin", "x86_64", "cpu", ["macos-x64"]),
        ("Linux", "aarch64", "cpu", ["ubuntu-arm64"]),
        ("Linux", "x86_64", "cuda", ["ubuntu-cuda-12.8-x64", "cudart-ubuntu-cuda-12.8-x64"]),
        ("Linux", "aarch64", "vulkan", ["ubuntu-vulkan-arm64"]),
    ],
)
def test_platform_assets(system, arch, backend, expected, monkeypatch):
    monkeypatch.setattr(engine.platform, "system", lambda: system)
    monkeypatch.setattr(engine.platform, "machine", lambda: arch)
    assert engine.selected_assets(backend) == (backend, expected)


@pytest.mark.parametrize(
    "system,arch,expected", [("Darwin", "arm64", "metal"), ("Darwin", "x86_64", "cpu")]
)
def test_macos_automatic_backend(system, arch, expected, monkeypatch):
    monkeypatch.setattr(engine.platform, "system", lambda: system)
    monkeypatch.setattr(engine.platform, "machine", lambda: arch)
    assert engine.recommended_backend() == expected


def test_unsupported_backend_fails_before_download(monkeypatch):
    monkeypatch.setattr(engine.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(engine.platform, "machine", lambda: "arm64")
    with pytest.raises(engine.InferenceError, match="No managed cuda"):
        engine.selected_assets("cuda")


@pytest.mark.parametrize(
    "name,link,type_",
    [
        ("../escape", "", tarfile.REGTYPE),
        ("/tmp/escape", "", tarfile.REGTYPE),
        ("bundle/link", "../../escape", tarfile.SYMTYPE),
        ("bundle/link", "/tmp/escape", tarfile.SYMTYPE),
        ("device", "", tarfile.CHRTYPE),
    ],
)
def test_extract_rejects_unsafe_entries_even_with_app_sandbox_disabled(
    isolated, name, link, type_, monkeypatch
):
    monkeypatch.setattr("app.tools.config_loader.sandbox_disabled", lambda: True)
    stage = isolated / "stage"
    stage.mkdir()
    archive = stage / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(name)
        member.type = type_
        member.linkname = link
        bundle.addfile(member)
    with pytest.raises((engine.InferenceError, ValueError)):
        engine._extract(archive, stage)
    assert not (isolated / "escape").exists()


def test_download_verifies_size_and_checksum(isolated, monkeypatch):
    contents = b"archive data"
    monkeypatch.setitem(
        engine.ASSETS, "test", (len(contents), hashlib.sha256(contents).hexdigest())
    )

    def stream(*args, **kwargs):
        return httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=contents))
        ).stream("GET", "https://example.test/archive")

    monkeypatch.setattr(engine.httpx, "stream", stream)
    with engine._safe_scope(isolated):
        engine._download("test", isolated / "good", lambda message: None)
        assert (isolated / "good").read_bytes() == contents
        monkeypatch.setitem(engine.ASSETS, "test", (len(contents), "0" * 64))
        with pytest.raises(engine.InferenceError, match="SHA256"):
            engine._download("test", isolated / "bad", lambda message: None)


def test_doctor_json_exit_and_noninteractive_setup_no_download(isolated, monkeypatch):
    monkeypatch.setattr(
        "app.config.resolve_llama_server_bin", lambda **kw: str(isolated / "missing")
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.output)["usable"] is False
    with patch.object(engine, "_download") as download:
        result = runner.invoke(cli, ["setup"])
    assert result.exit_code == 1
    assert "--yes" in result.output
    download.assert_not_called()


def test_install_lock_prevents_concurrent_activation(isolated):
    with engine._INSTALL_LOCK, pytest.raises(engine.InferenceError, match="already running"):
        engine.setup(backend="cpu", repair=True)


def test_api_requires_explicit_same_origin_action_and_reports_failure(isolated, monkeypatch):
    import app.main as main

    monkeypatch.setattr(
        engine, "diagnose", lambda: engine.Diagnosis("missing", False, problem="Run lls setup")
    )

    def fail(**kwargs):
        raise engine.InferenceError("Test setup failure")

    monkeypatch.setattr(engine, "setup", fail)
    monkeypatch.setattr(main, "_inference_setup_task", None)
    monkeypatch.setattr(
        main, "_inference_setup_state", {"running": False, "message": "", "error": ""}
    )
    with TestClient(app) as client:
        assert client.get("/api/inference/status").json()["usable"] is False
        assert client.post("/api/inference/setup").status_code == 403
        assert (
            client.post(
                "/api/inference/setup",
                headers={"X-LlamaStudio-Setup": "1", "Origin": "https://evil.test"},
            ).status_code
            == 403
        )
        response = client.post(
            "/api/inference/setup",
            headers={"X-LlamaStudio-Setup": "1", "Origin": "http://testserver"},
        )
        assert response.status_code == 200
        import time

        for _ in range(30):
            state = client.get("/api/inference/status").json()
            if not state["running"]:
                break
            time.sleep(0.01)
        assert state["error"] == "Test setup failure"


@pytest.mark.parametrize(
    "error",
    [
        "error while loading shared libraries: libcudart.so.12",
        "dyld: Library not loaded: libllama.dylib",
    ],
)
def test_model_load_reports_library_failure_without_cpu_retry(isolated, error):
    from unittest.mock import Mock

    model = isolated / "model.gguf"
    model.touch()
    log = isolated / "server.log"
    manager = ServerManager()

    def failed_start():
        log.write_text(error)
        return False

    with (
        patch.object(manager, "_write_log", return_value=log),
        patch.object(manager, "_build_command", return_value=["/bin/true"]),
        patch.object(manager, "_wait_for_ready", side_effect=failed_start),
        patch.object(manager, "eject_model"),
        patch(
            "app.server_manager.subprocess.Popen", return_value=Mock(poll=lambda: None)
        ) as launch,
    ):
        assert not manager.load_model(str(model), {"mmproj": ""})
        assert "lls setup --repair" in manager.last_error
        assert launch.call_count == 1


def test_environment_uses_launched_release_when_manifest_changes(isolated):
    previous = activate(isolated, "old")
    selected = activate(isolated, "new")
    manager = ServerManager()
    env = manager._build_env(binary_path=str(previous))
    variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    assert str(previous.parent) in env[variable]
    assert str(selected.parent) not in env[variable]


@pytest.mark.parametrize(
    "device", ["MTL0: Apple M4 Pro (16384 MiB, 8000 MiB free)", "Metal: Apple M1"]
)
def test_metal_device_names_are_accepted(isolated, device):
    binary = fake_binary(isolated / "metal", devices=f"Available devices:\n  {device}")
    assert engine.diagnose(str(binary), backend="metal").usable


def test_metal_startup_log_is_not_a_device(isolated):
    binary = fake_binary(
        isolated / "metal",
        devices="ggml_metal_init: found Metal support\nAvailable devices:\n  (none)",
    )
    assert not engine.diagnose(str(binary), backend="metal").usable


@pytest.mark.parametrize("failure", [504, "interrupted"])
def test_download_retries_gateway_and_interrupted_stream_from_scratch(
    isolated, monkeypatch, failure
):
    contents = b"verified archive"
    monkeypatch.setitem(
        engine.ASSETS, "test", (len(contents), hashlib.sha256(contents).hexdigest())
    )
    monkeypatch.setattr(engine.time, "sleep", lambda _: None)
    urls = []

    class Interrupted(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            raise httpx.ReadError("Connection lost")

    def stream(method, url, **kwargs):
        urls.append(url)

        def respond(request):
            if len(urls) == 1:
                if failure == 504:
                    return httpx.Response(504)
                return httpx.Response(200, stream=Interrupted())
            return httpx.Response(200, content=contents)

        return httpx.Client(transport=httpx.MockTransport(respond)).stream(method, url)

    monkeypatch.setattr(engine.httpx, "stream", stream)
    messages = []
    with engine._safe_scope(isolated):
        engine._download("test", isolated / "bundle", messages.append)
    assert (isolated / "bundle").read_bytes() == contents
    assert len(urls) == 2
    assert urls[0] != urls[1]
    assert any("retrying" in message for message in messages)


@pytest.mark.parametrize("status,attempts", [(504, 3), (404, 1)])
def test_download_retry_limit_and_permanent_failure(isolated, monkeypatch, status, attempts):
    monkeypatch.setitem(engine.ASSETS, "test", (1, "0" * 64))
    monkeypatch.setattr(engine.time, "sleep", lambda _: None)
    calls = []

    def stream(method, url, **kwargs):
        calls.append(url)
        return httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(status))).stream(
            method, url
        )

    monkeypatch.setattr(engine.httpx, "stream", stream)
    error = engine.InferenceError if status == 504 else httpx.HTTPStatusError
    with engine._safe_scope(isolated), pytest.raises(error) as raised:
        engine._download("test", isolated / "bundle", lambda _: None)
    assert len(calls) == attempts
    if status == 504:
        assert "after 3 attempts" in str(raised.value)
        assert "preserved" in str(raised.value)
    assert not (isolated / "bundle").exists()
