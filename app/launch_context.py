from __future__ import annotations

import os
import sys

ANY_HOSTS = {"0.0.0.0", "::", ""}
FALSE_VALUES = {"0", "false", "no", "off"}
TRUE_VALUES = {"1", "true", "yes", "on"}


def local_connect_host(bind_host: str) -> str:
    """Return the local address clients should use to probe a bound app host."""
    return "127.0.0.1" if bind_host in ANY_HOSTS else bind_host


def browser_host(bind_host: str) -> str:
    """Return a browser-friendly host for the configured bind host."""
    configured = os.environ.get("LLAMASTUDIO_BROWSER_HOST")
    if configured:
        return configured
    if bind_host in ANY_HOSTS:
        # Browsers only expose microphone APIs to secure contexts. Loopback HTTP
        # is trusted; an arbitrary LAN hostname over HTTP is not.
        return "127.0.0.1"
    return bind_host


def app_url(bind_host: str, port: int, view: str | None = None) -> str:
    base_url = os.environ.get("LLAMASTUDIO_BROWSER_URL")
    if not base_url:
        base_url = f"http://{browser_host(bind_host)}:{port}"
    base_url = base_url.rstrip("/")
    return f"{base_url}/?view={view}" if view else base_url


def should_open_browser() -> bool:
    configured = os.environ.get("LLAMASTUDIO_OPEN_BROWSER")
    if configured:
        return configured.strip().lower() in TRUE_VALUES

    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False

    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False

    browser = os.environ.get("BROWSER", "").strip().lower()
    return browser not in FALSE_VALUES
