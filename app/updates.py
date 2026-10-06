"""Optional PyPI checks and upgrades in the environment that owns this CLI."""

from __future__ import annotations

import contextlib
import importlib.metadata as metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import rich_click as click
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version
from rich.console import Console
from rich.markup import escape

from app.config_store import config_loader
from app.logger import logger
from app.tools import check_path_safe, reset_current_workspace_root, set_current_workspace_root

PACKAGE = "llamastudio"
PYPI_URL = "https://pypi.org/pypi/llamastudio/json"
INDEX_URL = "https://pypi.org/simple"
CHECK_INTERVAL = 60 * 60
RETRY_INTERVAL = 5 * 60
PROMPT_INTERVAL = 24 * 60 * 60


class UpdateError(Exception):
    """An explicit check or requested installation could not be completed."""


@dataclass(frozen=True)
class Installation:
    version: Version
    installer: str
    blocked_reason: str | None = None


def installation() -> Installation:
    """Avoid replacing an editable checkout or an OS-managed Python installation."""
    try:
        dist = metadata.distribution(PACKAGE)
        version = Version(dist.version)
    except (metadata.PackageNotFoundError, ValueError) as exc:
        raise UpdateError(
            "LlamaStudio is running from source. Update your checkout with git."
        ) from exc
    installer = (dist.read_text("INSTALLER") or "pip").strip()
    direct_url = dist.read_text("direct_url.json")
    try:
        editable = direct_url and json.loads(direct_url).get("dir_info", {}).get("editable")
    except (ValueError, AttributeError, TypeError):
        editable = True  # Unknown source metadata must not be replaced automatically.
    if (
        editable
        or not dist.read_text("WHEEL")
        or Path(dist.locate_file("app/updates.py")).resolve() != Path(__file__).resolve()
    ):
        return Installation(version, installer, "Source install: update your checkout with git.")
    if (
        sys.prefix == sys.base_prefix
        and (Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED").exists()
    ):
        return Installation(
            version, installer, "This Python is managed by your OS. Use a virtualenv or pipx."
        )
    return Installation(version, installer)


def _cache_path() -> Path:
    # This is app state, confined to the configured config directory, not a model tool path.
    token = set_current_workspace_root(config_loader.config_dir)
    try:
        return check_path_safe("update-check.json")
    finally:
        reset_current_workspace_root(token)


def _read_cache() -> dict:
    try:
        state = json.loads(_cache_path().read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError, RuntimeError):
        return {}


def _write_cache(state: dict) -> None:
    temporary = None
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(state, stream)
        temporary.replace(path)
    except (OSError, ValueError, RuntimeError):
        logger.debug("Could not save the update check cache", exc_info=True)
    finally:
        if temporary:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)


def _recent(timestamp: object, interval: int) -> bool:
    try:
        return (
            isinstance(timestamp, (float, int))
            and math.isfinite(timestamp)
            and 0 <= time.time() - timestamp < interval
        )
    except OverflowError:
        return False


def latest_release() -> Version | None:
    """Return the latest stable, non-yanked release supported by this Python."""
    try:
        response = httpx.get(PYPI_URL, timeout=2.0, follow_redirects=True)
        response.raise_for_status()
        data = response.json()
        candidate = Version(data["info"]["version"])
        if candidate.is_prerelease or candidate.is_devrelease:
            return None
        supported_tags = set(sys_tags())
        for artifact in data["urls"]:
            if artifact.get("packagetype") == "bdist_wheel":
                _, _, _, tags = parse_wheel_filename(artifact["filename"])
                if not supported_tags.intersection(tags):
                    continue
            if not artifact.get("yanked", False) and SpecifierSet(
                artifact.get("requires_python") or ""
            ).contains(platform.python_version(), prereleases=True):
                return candidate
        return None
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise UpdateError(
            "Could not check PyPI for updates. Try again when you are online."
        ) from exc


def check_release(*, force: bool = False) -> tuple[Version | None, dict]:
    state = _read_cache()
    interval = RETRY_INTERVAL if state.get("failed") else CHECK_INTERVAL
    if (
        not force
        and state.get("python") == platform.python_version()
        and state.get("platform") == f"{sys.platform}:{platform.machine()}"
        and _recent(state.get("checked_at"), interval)
    ):
        try:
            candidate = Version(state["latest"]) if state.get("latest") else None
            if candidate and (candidate.is_prerelease or candidate.is_devrelease):
                raise ValueError("Cached prerelease")
            return candidate, state
        except (ValueError, TypeError):
            pass
    state.update(
        checked_at=time.time(),
        python=platform.python_version(),
        platform=f"{sys.platform}:{platform.machine()}",
        latest=None,
    )
    try:
        candidate = latest_release()
        state.update(latest=str(candidate) if candidate else None, failed=False)
    except UpdateError:
        state["failed"] = True
        _write_cache(state)
        raise
    _write_cache(state)
    return candidate, state


def upgrade_command(installed: Installation, candidate: Version) -> list[str]:
    if installed.blocked_reason:
        raise UpdateError(installed.blocked_reason)
    target = f"{PACKAGE}=={candidate}"
    uv = shutil.which("uv")
    if installed.installer == "uv" or importlib.util.find_spec("pip") is None:
        if not uv:
            raise UpdateError(
                "The installer is unavailable. Install pip or restore uv, then retry."
            )
        return [
            uv,
            "--no-config",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--upgrade",
            "--index-url",
            INDEX_URL,
            target,
        ]
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--upgrade",
        "--index-url",
        INDEX_URL,
        target,
    ]


def install_upgrade(installed: Installation, candidate: Version) -> None:
    """Use argument arrays and verify the result in a fresh process before reporting success."""
    if candidate <= installed.version:
        raise UpdateError("Refusing to downgrade LlamaStudio.")
    # Inherited pip settings must not redirect an upgrade into another environment
    # or substitute a different package source for the PyPI release we checked.
    env = os.environ.copy()
    for key in (
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_ROOT",
        "PIP_USER",
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
        "PIP_NO_INDEX",
        "PIP_REQUIREMENT",
        "PIP_EDITABLE",
        "UV_INDEX",
        "UV_DEFAULT_INDEX",
        "UV_EXTRA_INDEX_URL",
        "UV_INDEX_URL",
    ):
        env.pop(key, None)
    env["PIP_CONFIG_FILE"] = os.devnull
    try:
        result = subprocess.run(upgrade_command(installed, candidate), check=False, env=env)
        if result.returncode:
            raise UpdateError("The installer failed. Review its output above and retry lls update.")
        version = subprocess.check_output(
            [
                sys.executable,
                "-E",
                "-c",
                "import sys; sys.path.pop(0); import importlib.metadata; "
                "sys.stdout.write(importlib.metadata.version('llamastudio'))",
            ],
            text=True,
            timeout=10,
        )
        if Version(version.strip()) != candidate:
            raise UpdateError("The installed version did not change as expected. Retry lls update.")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise UpdateError(f"Could not complete the upgrade: {exc}") from exc


def is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def run_update(
    console: Console, *, automatic: bool = False, check_only: bool = False, yes: bool = False
) -> bool:
    """Return True after an install; the caller must exit instead of running old loaded code."""
    interactive = is_interactive_terminal()
    if automatic and (not interactive or os.environ.get("CI")):
        return False
    try:
        installed = installation()
        candidate, state = check_release(force=not automatic)
    except UpdateError:
        if automatic:
            logger.debug("Automatic update check unavailable", exc_info=True)
            return False
        raise
    if candidate is None or candidate <= installed.version:
        if not automatic:
            console.print(f"LlamaStudio {installed.version}: no newer compatible release on PyPI.")
        return False
    if (
        automatic
        and state.get("prompted_version") == str(candidate)
        and _recent(state.get("prompted_at"), PROMPT_INTERVAL)
    ):
        return False
    console.print(f"[cyan]LlamaStudio update available: {installed.version} → {candidate}[/cyan]")
    if installed.blocked_reason:
        console.print(escape(installed.blocked_reason))
        if not automatic and not check_only:
            raise UpdateError(installed.blocked_reason)
    elif not check_only:
        if not interactive and not yes:
            console.print("Run lls update in a terminal, or lls update --yes to install.")
            return False
        # Mark declined/failed prompts too so every command does not ask again.
        state.update(prompted_version=str(candidate), prompted_at=time.time())
        _write_cache(state)
        if not yes and not click.confirm(f"Install LlamaStudio {candidate} now?", default=False):
            return False
        console.print(f"Installing LlamaStudio {candidate} in this Python environment…")
        install_upgrade(installed, candidate)
        state.update(latest=str(candidate), prompted_version=None, prompted_at=None)
        _write_cache(state)
        console.print(
            f"[green]Installed LlamaStudio {candidate}.[/green] Run your command again to use it.\n"
            "An already running backend keeps its current version until you restart it."
        )
        return True
    if automatic:
        state.update(prompted_version=str(candidate), prompted_at=time.time())
        _write_cache(state)
    return False
