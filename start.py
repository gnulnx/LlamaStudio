#!/usr/bin/env python3
"""
LLamaStudio - Startup script.

This script:
1. Starts the FastAPI app (which auto-starts llama-server on startup)
2. Opens a browser window to the chat interface
3. On exit, shuts down both the FastAPI app and llama-server
"""
import signal
import sys
import time
import webbrowser
import socket
import subprocess
from pathlib import Path

# Add the project root to the path
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Initialize logging before any app imports
from app.logger import setup_logging
logger = setup_logging()

from app.main import app
from app.config import settings

PID_FILE = Path.home() / ".config" / "llamastudio" / "app.pid"


def app_url() -> str:
    return f"http://127.0.0.1:{settings.APP_PORT}"


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
    return str(Path(__file__).resolve()) in cmdline


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
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    deadline = time.time() + 8
    while time.time() < deadline:
        if all(not pid_is_running(pid) for pid in pids) and not port_in_use():
            return True
        time.sleep(0.2)

    for pid in pids:
        if pid_is_running(pid):
            logger.warning(f"[startup] Force killing unresponsive LLamaStudio process {pid}")
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    return wait_for_port_clear()


def write_pid_file():
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def remove_pid_file():
    try:
        if PID_FILE.exists() and PID_FILE.read_text().strip() == str(os.getpid()):
            PID_FILE.unlink()
    except Exception:
        pass


def open_browser():
    """Open the browser after a short delay to let the server start."""
    time.sleep(2)
    url = app_url()
    logger.info(f"[startup] Opening browser to {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        logger.error(f"[startup] Could not open browser: {e}")
        logger.info(f"[startup] Open manually: {url}")

if __name__ == "__main__":
    import uvicorn

    stopped_existing = stop_existing_app()
    if port_in_use():
        logger.error(
            f"[startup] Port {settings.APP_PORT} is already in use by a non-LLamaStudio process. "
            f"Open {app_url()} manually or free the port."
        )
        sys.exit(1)

    write_pid_file()

    # Open browser in a separate thread
    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    logger.info(f"[LLamaStudio] Starting on http://{settings.APP_HOST}:{settings.APP_PORT}")
    if stopped_existing:
        logger.info("[LLamaStudio] Previous instance was stopped; launching refreshed app")
    logger.info(f"[LLamaStudio] llama-server will start automatically")

    # Run uvicorn
    try:
        uvicorn.run(
            app,
            host=settings.APP_HOST,
            port=settings.APP_PORT,
            log_level="info",
        )
    finally:
        remove_pid_file()
