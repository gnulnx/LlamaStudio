"""Update policy, installation boundaries, and CLI integration without real installations."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from click.testing import CliRunner
from packaging.version import Version
from rich.console import Console

from app import updates
from app.cli import cli


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    monkeypatch.setattr(updates.config_loader, "config_dir", tmp_path)
    monkeypatch.setattr(updates.config_loader, "sandbox_disabled", lambda: False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("LLAMASTUDIO_NO_UPDATE_CHECK", raising=False)


def payload(version="1.5.0", **artifact):
    return {
        "info": {"version": version},
        "urls": [{"yanked": False, "requires_python": ">=3.10", **artifact}],
    }


def mock_pypi(monkeypatch, data):
    response = httpx.Response(200, json=data, request=httpx.Request("GET", updates.PYPI_URL))
    request = Mock(return_value=response)
    monkeypatch.setattr(updates.httpx, "get", request)
    return request


@pytest.mark.parametrize("version", ["1.5.0", "1.10.0", "2.0.0"])
def test_fetches_release_with_short_timeout(monkeypatch, version):
    request = mock_pypi(monkeypatch, payload(version))
    assert updates.latest_release() == Version(version)
    request.assert_called_once_with(updates.PYPI_URL, timeout=2.0, follow_redirects=True)


@pytest.mark.parametrize(
    "data",
    [
        payload("1.5.0rc1"),
        payload("1.5.0.dev1"),
        payload(yanked=True),
        payload(requires_python=">=99"),
        {"info": {"version": "1.5.0"}, "urls": []},
        payload(packagetype="bdist_wheel", filename="llamastudio-1.5.0-cp39-cp39-any.whl"),
    ],
)
def test_ignores_prereleases_yanks_and_incompatible_artifacts(monkeypatch, data):
    mock_pypi(monkeypatch, data)
    assert updates.latest_release() is None


@pytest.mark.parametrize("data", [{}, [], payload("not-a-version"), payload(requires_python="bad")])
def test_malformed_responses_raise_useful_error(monkeypatch, data):
    mock_pypi(monkeypatch, data)
    with pytest.raises(updates.UpdateError, match="Could not check PyPI"):
        updates.latest_release()


def test_cache_and_forced_refresh(monkeypatch):
    request = mock_pypi(monkeypatch, payload())
    assert updates.check_release()[0] == Version("1.5.0")
    assert updates.check_release()[0] == Version("1.5.0")
    assert request.call_count == 1
    updates.check_release(force=True)
    assert request.call_count == 2


def test_corrupt_cache_is_refetched(monkeypatch, tmp_path):
    (tmp_path / "update-check.json").write_text("broken")
    request = mock_pypi(monkeypatch, payload())
    assert updates.check_release()[0] == Version("1.5.0")
    assert request.call_count == 1


@pytest.mark.parametrize("timestamp", [None, "bad", float("nan"), float("inf"), 10**400, -1])
def test_invalid_cache_timestamps_are_not_recent(timestamp):
    assert not updates._recent(timestamp, 3600)


def test_failed_checks_back_off(monkeypatch):
    request = Mock(side_effect=httpx.ConnectError("offline"))
    monkeypatch.setattr(updates.httpx, "get", request)
    with pytest.raises(updates.UpdateError):
        updates.check_release()
    assert updates.check_release()[0] is None
    assert request.call_count == 1
    with pytest.raises(updates.UpdateError):
        updates.check_release(force=True)
    assert request.call_count == 2


def fake_installation(monkeypatch, version="1.4.0", installer="pip", blocked_reason=None):
    installed = updates.Installation(Version(version), installer, blocked_reason)
    monkeypatch.setattr(updates, "installation", lambda: installed)
    return installed


def fake_distribution(monkeypatch, direct_url=None, path=None):
    dist = SimpleNamespace(
        version="1.4.0",
        read_text=lambda name: direct_url if name == "direct_url.json" else "pip",
        locate_file=lambda name: path or Path(updates.__file__),
    )
    monkeypatch.setattr(updates.metadata, "distribution", lambda name: dist)
    monkeypatch.setattr(updates.sys, "prefix", "test-virtualenv")


@pytest.mark.parametrize("direct_url", [json.dumps({"dir_info": {"editable": True}}), "malformed"])
def test_editable_and_unknown_source_metadata_are_protected(monkeypatch, direct_url):
    fake_distribution(monkeypatch, direct_url)
    assert "Source install" in updates.installation().blocked_reason


def test_running_checkout_is_protected_even_with_stale_wheel_metadata(monkeypatch, tmp_path):
    fake_distribution(monkeypatch, path=tmp_path / "app" / "updates.py")
    assert "Source install" in updates.installation().blocked_reason


def test_legacy_egg_info_in_checkout_is_protected(monkeypatch):
    fake_distribution(monkeypatch)
    dist = updates.metadata.distribution("llamastudio")
    dist.read_text = lambda name: None
    assert "Source install" in updates.installation().blocked_reason


def test_os_managed_python_is_protected(monkeypatch, tmp_path):
    fake_distribution(monkeypatch)
    monkeypatch.setattr(updates.sys, "prefix", updates.sys.base_prefix)
    monkeypatch.setattr(updates.sysconfig, "get_path", lambda name: str(tmp_path))
    (tmp_path / "EXTERNALLY-MANAGED").touch()
    assert "managed by your OS" in updates.installation().blocked_reason


@pytest.mark.parametrize("installer", ["pip", "uv"])
def test_upgrade_targets_running_interpreter(monkeypatch, installer):
    monkeypatch.setattr(updates.shutil, "which", lambda name: "/tools/uv")
    monkeypatch.setattr(updates.importlib.util, "find_spec", lambda name: object())
    installed = updates.Installation(Version("1.4.0"), installer)
    command = updates.upgrade_command(installed, Version("1.5.0"))
    assert updates.sys.executable in command
    assert command[-1] == "llamastudio==1.5.0"
    assert "--upgrade" in command
    assert updates.INDEX_URL in command
    if installer == "uv":
        assert command[0] == "/tools/uv"
        assert "--python" in command
    else:
        assert command[:3] == [updates.sys.executable, "-m", "pip"]


def test_missing_pip_can_use_uv(monkeypatch):
    monkeypatch.setattr(updates.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(updates.shutil, "which", lambda name: "/tools/uv")
    assert (
        updates.upgrade_command(updates.Installation(Version("1.4.0"), "pip"), Version("1.5.0"))[0]
        == "/tools/uv"
    )


@pytest.mark.parametrize("outcome", [1, "wrong-version", "failed-verification"])
def test_failed_install_or_verification_never_reports_success(monkeypatch, outcome):
    installed = fake_installation(monkeypatch)
    monkeypatch.setattr(updates, "upgrade_command", lambda *args: ["installer"])
    monkeypatch.setattr(
        updates.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1 if outcome == 1 else 0),
    )
    verify = Mock(return_value="1.4.0")
    if outcome == "failed-verification":
        verify.side_effect = subprocess.CalledProcessError(1, "verify")
    monkeypatch.setattr(updates.subprocess, "check_output", verify)
    with pytest.raises(updates.UpdateError):
        updates.install_upgrade(installed, Version("1.5.0"))


def test_refuses_downgrade(monkeypatch):
    installed = fake_installation(monkeypatch, "1.10.0")
    with pytest.raises(updates.UpdateError, match="downgrade"):
        updates.install_upgrade(installed, Version("1.9.0"))


def test_inherited_settings_cannot_redirect_the_install(monkeypatch):
    installed = fake_installation(monkeypatch)
    for key in (
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_ROOT",
        "PIP_USER",
        "PIP_EXTRA_INDEX_URL",
        "UV_INDEX",
    ):
        monkeypatch.setenv(key, "unrelated-location")
    monkeypatch.setattr(updates, "upgrade_command", lambda *args: ["installer"])
    install = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(updates.subprocess, "run", install)
    monkeypatch.setattr(updates.subprocess, "check_output", lambda *args, **kwargs: "1.5.0")
    updates.install_upgrade(installed, Version("1.5.0"))
    child_env = install.call_args.kwargs["env"]
    assert "PIP_TARGET" not in child_env
    assert "PIP_PREFIX" not in child_env
    assert "PIP_ROOT" not in child_env
    assert "PIP_USER" not in child_env
    assert "PIP_EXTRA_INDEX_URL" not in child_env
    assert "UV_INDEX" not in child_env
    assert child_env["PIP_CONFIG_FILE"] == updates.os.devnull


def test_unwritable_cache_does_not_block_check(monkeypatch):
    request = mock_pypi(monkeypatch, payload())
    monkeypatch.setattr(updates, "_cache_path", Mock(side_effect=PermissionError("readonly")))
    assert updates.check_release()[0] == Version("1.5.0")
    request.assert_called_once()


def test_ci_never_automatically_checks_even_with_a_terminal(monkeypatch):
    monkeypatch.setattr(updates, "is_interactive_terminal", lambda: True)
    monkeypatch.setenv("CI", "true")
    request = mock_pypi(monkeypatch, payload())
    assert not updates.run_update(Console(file=io.StringIO()), automatic=True)
    request.assert_not_called()


@pytest.mark.parametrize("platform_name", ["linux", "darwin", "win32"])
def test_installer_has_no_platform_exclusion(monkeypatch, platform_name):
    monkeypatch.setattr(updates.sys, "platform", platform_name)
    monkeypatch.setattr(updates.sys, "executable", "/python path/with spaces/python")
    monkeypatch.setattr(updates.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(updates.shutil, "which", lambda name: None)
    installed = fake_installation(monkeypatch)
    assert updates.upgrade_command(installed, Version("1.5.0"))[:3] == [
        "/python path/with spaces/python",
        "-m",
        "pip",
    ]


def test_noninteractive_automatic_checks_do_no_network_or_install(monkeypatch):
    request = mock_pypi(monkeypatch, payload())
    fake_installation(monkeypatch)
    monkeypatch.setattr(updates, "is_interactive_terminal", lambda: False)
    install = Mock()
    monkeypatch.setattr(updates, "install_upgrade", install)
    assert not updates.run_update(Console(file=io.StringIO()), automatic=True)
    request.assert_not_called()
    install.assert_not_called()


def test_decline_is_remembered_and_new_version_prompts_again(monkeypatch):
    fake_installation(monkeypatch)
    mock_pypi(monkeypatch, payload())
    monkeypatch.setattr(updates, "is_interactive_terminal", lambda: True)
    confirm = Mock(return_value=False)
    monkeypatch.setattr(updates.click, "confirm", confirm)
    console = Console(file=io.StringIO())
    assert not updates.run_update(console, automatic=True)
    assert not updates.run_update(console, automatic=True)
    confirm.assert_called_once()
    mock_pypi(monkeypatch, payload("1.6.0"))
    updates.check_release(force=True)
    assert not updates.run_update(console, automatic=True)
    assert confirm.call_count == 2


def test_accept_installs_exact_candidate(monkeypatch):
    installed = fake_installation(monkeypatch)
    mock_pypi(monkeypatch, payload())
    monkeypatch.setattr(updates, "is_interactive_terminal", lambda: True)
    monkeypatch.setattr(updates.click, "confirm", lambda *args, **kwargs: True)
    install = Mock()
    monkeypatch.setattr(updates, "install_upgrade", install)
    output = io.StringIO()
    assert updates.run_update(Console(file=output), automatic=True)
    install.assert_called_once_with(installed, Version("1.5.0"))
    assert "Installed LlamaStudio 1.5.0" in output.getvalue()
    assert "backend keeps its current version" in output.getvalue()


def test_manual_check_and_yes(monkeypatch):
    fake_installation(monkeypatch)
    mock_pypi(monkeypatch, payload())
    monkeypatch.setattr(updates, "is_interactive_terminal", lambda: False)
    install = Mock()
    monkeypatch.setattr(updates, "install_upgrade", install)
    runner = CliRunner()
    checked = runner.invoke(cli, ["update", "--check"])
    assert checked.exit_code == 0, checked.output
    assert "update available" in checked.output
    install.assert_not_called()
    not_confirmed = runner.invoke(cli, ["update"])
    assert not_confirmed.exit_code == 0
    install.assert_not_called()
    accepted = runner.invoke(cli, ["update", "--yes"])
    assert accepted.exit_code == 0, accepted.output
    install.assert_called_once()


def test_offline_automatic_check_continues_but_explicit_check_fails(monkeypatch):
    fake_installation(monkeypatch)
    monkeypatch.setattr(updates, "is_interactive_terminal", lambda: True)
    monkeypatch.setattr(updates.httpx, "get", Mock(side_effect=httpx.ConnectError("offline")))
    assert not updates.run_update(Console(file=io.StringIO()), automatic=True)
    result = CliRunner().invoke(cli, ["update", "--check"])
    assert result.exit_code == 1
    assert "Could not check PyPI" in result.output


@pytest.mark.parametrize(
    "arguments", [["--no-update-check", "status"], ["--help"], ["status", "--help"], ["--version"]]
)
def test_opt_out_and_help_never_check(monkeypatch, arguments):
    update = Mock()
    monkeypatch.setattr("app.cli.run_update", update)
    monkeypatch.setattr("app.cli.is_server_online", lambda: False)
    result = CliRunner().invoke(cli, arguments)
    assert result.exit_code == 0, result.output
    update.assert_not_called()


def test_environment_opt_out(monkeypatch):
    update = Mock()
    monkeypatch.setattr("app.cli.run_update", update)
    monkeypatch.setattr("app.cli.is_server_online", lambda: False)
    result = CliRunner().invoke(cli, ["status"], env={"LLAMASTUDIO_NO_UPDATE_CHECK": "1"})
    assert result.exit_code == 0
    update.assert_not_called()


@pytest.mark.parametrize("command", ["start", "status", "tui"])
def test_successful_auto_upgrade_exits_before_launching_old_code(monkeypatch, command):
    monkeypatch.setattr("app.cli.run_update", lambda *args, **kwargs: True)
    if command == "tui":
        monkeypatch.setattr(
            "app.cli.sys",
            SimpleNamespace(
                stdin=SimpleNamespace(isatty=lambda: True),
                stdout=SimpleNamespace(isatty=lambda: True),
            ),
        )
    backend = Mock()
    monkeypatch.setattr("app.cli.is_server_online", backend)
    result = CliRunner().invoke(cli, [command])
    assert result.exit_code == 0, result.output
    backend.assert_not_called()
