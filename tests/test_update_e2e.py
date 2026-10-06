"""Real terminal → HTTP index → pip/uv → upgraded console entry point.

Run with LLAMASTUDIO_UPDATE_E2E=1 python -m pytest tests/test_update_e2e.py
--basetemp=.runtime/update-e2e. Requires build and uv; downloads dependencies
from PyPI but publishes nothing. Only the metadata/index URLs are redirected;
the prompt, HTTP requests, package installers, and verification are real.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import select
import shutil
import subprocess
import sys
import threading
import time
import venv
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import httpx
import pytest
from packaging.utils import canonicalize_name, parse_wheel_filename

from app.tools import check_path_safe, reset_current_workspace_root, set_current_workspace_root

pytestmark = pytest.mark.skipif(
    os.environ.get("LLAMASTUDIO_UPDATE_E2E") != "1" or os.name != "posix",
    reason="Opt-in Linux/macOS package installation and terminal test",
)
ROOT = Path(__file__).resolve().parents[1]
OLD_VERSION = "1.4.0"
NEW_VERSION = "1.5.0"


def run(command, *, cwd, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def package_index():
    token = set_current_workspace_root(ROOT)
    try:
        work = check_path_safe(str(ROOT / ".runtime" / "update-e2e-packages"))
    finally:
        reset_current_workspace_root(token)
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source"
    if source.exists():
        shutil.rmtree(source)
    source.mkdir()
    tracked = run(["git", "ls-files", "app"], cwd=ROOT).splitlines()
    for filename in tracked:
        destination = source / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, destination)
    shutil.copyfile(ROOT / "README.md", source / "README.md")
    project = (ROOT / "pyproject.toml").read_text()
    wheels = work / "wheels"
    wheels.mkdir(exist_ok=True)
    for version in ("0.0.0", OLD_VERSION, NEW_VERSION):
        (source / "pyproject.toml").write_text(
            re.sub(r'(?m)^version = "[^"]+"', f'version = "{version}"', project)
        )
        run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheels),
                str(source),
            ],
            cwd=ROOT,
        )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--index-url",
            "https://pypi.org/simple",
            "--dest",
            str(wheels),
            str(wheels / f"llamastudio-{NEW_VERSION}-py3-none-any.whl"),
        ],
        cwd=ROOT,
    )
    links = {}
    for wheel in wheels.glob("*.whl"):
        name, _, _, _ = parse_wheel_filename(wheel.name)
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        links.setdefault(canonicalize_name(name), []).append(
            f'<a href="../../wheels/{wheel.name}#sha256={digest}">{wheel.name}</a>'
        )
    for name, entries in links.items():
        directory = work / "simple" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.html").write_text("\n".join(entries))
    metadata = work / "pypi" / "llamastudio"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "json").write_text(
        json.dumps({
            "info": {"version": NEW_VERSION},
            "urls": [
                {
                    "packagetype": "bdist_wheel",
                    "filename": f"llamastudio-{NEW_VERSION}-py3-none-any.whl",
                    "requires_python": ">=3.10",
                    "yanked": False,
                }
            ],
        })
    )

    class Handler(SimpleHTTPRequestHandler):
        downloaded: ClassVar[list[str]] = []

        def do_GET(self):
            self.downloaded.append(self.path)
            super().do_GET()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(work)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield work, f"http://127.0.0.1:{server.server_port}", Handler.downloaded
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def terminal(command, *, cwd, env, answer=None):
    """Use a real PTY; wait for the prompt before submitting an answer."""
    import pty

    master, slave = pty.openpty()
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=slave, stdout=slave, stderr=slave)
    os.close(slave)
    output = bytearray()
    answered = False
    deadline = time.monotonic() + 90
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
                if answer is not None and not answered and b"now?" in output:
                    os.write(master, answer.encode() + b"\n")
                    answered = True
            if process.poll() is not None and not ready:
                break
        else:
            pytest.fail(f"Terminal timed out: {output.decode(errors='replace')}")
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        os.close(master)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output.decode(errors="replace"))
    assert process.returncode == 0, text
    if answer is not None:
        assert answered, text
    return text


def prepare_environment(package_index, installer, label, version):
    work, url, _ = package_index
    environment = work / f"venv-{label}-{installer}"
    if environment.exists():
        shutil.rmtree(environment)
    venv.EnvBuilder(with_pip=installer == "pip").create(environment)
    python = str(environment / "bin" / "python")
    lls = str(environment / "bin" / "lls")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("LLAMASTUDIO_", "PIP_", "UV_")) and key not in ("PYTHONPATH", "CI")
    }
    env["PIP_CONFIG_FILE"] = os.devnull
    config = work / f"config-{label}-{installer}"
    if config.exists():
        shutil.rmtree(config)
    env["LLAMASTUDIO_CONFIG_DIR"] = str(config)
    for setting, filename in (
        ("CONFIG_FILE", "config.json"),
        ("MODEL_PROFILES_FILE", "model_profiles.json"),
        ("MODEL_SETTINGS_FILE", "model_settings.json"),
        ("CONVERSATIONS_FILE", "conversations.json"),
        ("LOG_DIR", "logs"),
    ):
        env[f"LLAMASTUDIO_{setting}"] = str(config / filename)
    env["LLAMASTUDIO_WORKSPACE_ROOT"] = str(work)
    env["LLAMASTUDIO_APP_PORT"] = "1"
    env["LLAMASTUDIO_OPEN_BROWSER"] = "0"
    if installer == "pip":
        seed = [python, "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
    else:
        uv = shutil.which("uv")
        assert uv, "Install uv to exercise uv-managed environments"
        seed = [uv, "--no-config", "pip", "install", "--python", python]
    run([*seed, "--index-url", f"{url}/simple", f"llamastudio=={version}"], cwd=work, env=env)
    return environment, python, lls, env


@pytest.mark.parametrize("installer", ["pip", "uv"])
def test_real_prompt_download_install_and_next_launch(package_index, installer):
    work, url, downloads = package_index
    environment, python, lls, env = prepare_environment(
        package_index, installer, "candidate", OLD_VERSION
    )
    assert OLD_VERSION in run([lls, "--version"], cwd=work, env=env)
    entrypoint = (
        "from app import updates; "
        f"updates.PYPI_URL={url + '/pypi/llamastudio/json'!r}; "
        f"updates.INDEX_URL={url + '/simple'!r}; "
        "from app.cli import cli; cli()"
    )
    command = [python, "-c", entrypoint, "status"]
    declined = terminal(command, cwd=work, env=env, answer="n")
    assert "update available" in declined
    assert "OFFLINE" in declined
    assert OLD_VERSION in run([lls, "--version"], cwd=work, env=env)
    cached = terminal(command, cwd=work, env=env)
    assert "Install LlamaStudio" not in cached
    env["LLAMASTUDIO_CONFIG_DIR"] = str(work / f"accepted-config-{installer}")
    accepted_config = Path(env["LLAMASTUDIO_CONFIG_DIR"])
    if accepted_config.exists():
        shutil.rmtree(accepted_config)
    downloads.clear()
    accepted = terminal(command, cwd=work, env=env, answer="y")
    assert f"Installed LlamaStudio {NEW_VERSION}" in accepted
    assert "OFFLINE" not in accepted
    assert any(f"llamastudio-{NEW_VERSION}" in path for path in downloads)
    assert NEW_VERSION in run([lls, "--version"], cwd=work, env=env)
    location = run([python, "-c", "import app; print(app.__file__)"], cwd=work, env=env)
    assert str(environment) in location
    next_launch = terminal(command, cwd=work, env=env)
    assert "Install LlamaStudio" not in next_launch
    assert "OFFLINE" in next_launch


@pytest.mark.parametrize("installer", ["pip", "uv"])
def test_real_pypi_upgrade_and_published_cli(package_index, installer):
    """Exercise unmodified PyPI URLs and download the real published package."""
    work, _, _ = package_index
    _, python, lls, env = prepare_environment(package_index, installer, "pypi", "0.0.0")
    response = httpx.get("https://pypi.org/pypi/llamastudio/json", timeout=10)
    response.raise_for_status()
    expected = response.json()["info"]["version"]
    output = terminal([lls, "update"], cwd=work, env=env, answer="y")
    assert f"Installed LlamaStudio {expected}" in output
    actual = run(
        [
            python,
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('llamastudio'))",
        ],
        cwd=work,
        env=env,
    )
    assert actual.strip() == expected
    status = run([lls, "status"], cwd=work, env=env)
    assert "OFFLINE" in status
