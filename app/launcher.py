"""
LLamaStudio application launcher.

This module is importable from an installed wheel, so console scripts can start
the FastAPI app without relying on a repository-root start.py file.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from .config import settings
from .config_store import config_loader
from .logger import logger, setup_logging
from .main import app
from .model_manager import scan_models

PID_FILE = Path(settings.CONFIG_DIR) / "app.pid"


def app_url(view: str | None = None) -> str:
    url = f"http://127.0.0.1:{settings.APP_PORT}"
    return f"{url}/?view={view}" if view else url


def select_launch_view() -> str:
    return config_loader.get_launch_view(
        model_loaded=False,
        models_available=bool(scan_models()),
        consume_first_launch=True,
    )


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def pid_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ")
    except Exception:
        return ""


def is_llamastudio_pid(pid: int) -> bool:
    cmdline = pid_cmdline(pid)
    module_marker = "app.launcher"
    script_marker = "start.py"
    return (
        module_marker in cmdline
        or script_marker in cmdline
        or ("llamastudio" in cmdline.lower() and "python" in cmdline)
    )


def port_in_use() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((settings.APP_HOST, settings.APP_PORT)) == 0


def pids_listening_on_app_port() -> list[int]:
    commands = [
        ["lsof", "-t", f"-iTCP:{settings.APP_PORT}", "-sTCP:LISTEN"],
        ["fuser", f"{settings.APP_PORT}/tcp"],
    ]
    for cmd in commands:
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            continue
        pids = []
        for token in output.split():
            if token.isdigit():
                pids.append(int(token))
        if pids:
            return pids
    return []


def existing_app_pids() -> list[int]:
    pids = set()
    try:
        pid = int(PID_FILE.read_text().strip())
        if pid != os.getpid() and pid_is_running(pid) and is_llamastudio_pid(pid):
            pids.add(pid)
    except Exception:
        pass

    for pid in pids_listening_on_app_port():
        if pid != os.getpid() and is_llamastudio_pid(pid):
            pids.add(pid)

    return sorted(pids)


def wait_for_port_clear(timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not port_in_use():
            return True
        time.sleep(0.2)
    return not port_in_use()


def stop_existing_app() -> bool:
    pids = existing_app_pids()
    if not pids:
        return False

    logger.info(f"[startup] Restarting existing LLamaStudio process(es): {pids}")
    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)

    deadline = time.time() + 8
    while time.time() < deadline:
        if all(not pid_is_running(pid) for pid in pids) and not port_in_use():
            return True
        time.sleep(0.2)

    for pid in pids:
        if pid_is_running(pid):
            logger.warning(f"[startup] Force killing unresponsive LLamaStudio process {pid}")
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)

    return wait_for_port_clear()


def write_pid_file() -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def remove_pid_file() -> None:
    try:
        if PID_FILE.exists() and PID_FILE.read_text().strip() == str(os.getpid()):
            PID_FILE.unlink()
    except Exception:
        pass


def open_browser() -> None:
    """Open the browser after a short delay to let the server start."""
    time.sleep(2)
    url = app_url(select_launch_view())
    logger.info(f"[startup] Opening browser to {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        logger.error(f"[startup] Could not open browser: {e}")
        logger.info(f"[startup] Open manually: {url}")


def main() -> None:
    global logger
    import threading

    import uvicorn

    logger = setup_logging()
    config_loader.initialize_for_launch(Path.cwd())
    stopped_existing = stop_existing_app()
    if port_in_use():
        logger.error(
            f"[startup] Port {settings.APP_PORT} is already in use by a non-LLamaStudio process. "
            f"Open {app_url()} manually or free the port."
        )
        sys.exit(1)

    write_pid_file()
    threading.Thread(target=open_browser, daemon=True).start()

    logger.info(f"[LLamaStudio] Starting on http://{settings.APP_HOST}:{settings.APP_PORT}")
    if stopped_existing:
        logger.info("[LLamaStudio] Previous instance was stopped; launching refreshed app")
    logger.info("[LLamaStudio] llama-server will start automatically")

    try:
        uvicorn.run(
            app,
            host=settings.APP_HOST,
            port=settings.APP_PORT,
            log_level="info",
        )
    finally:
        remove_pid_file()


if __name__ == "__main__":
    main()
