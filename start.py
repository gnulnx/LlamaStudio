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

# Add the project root to the path
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Initialize logging before any app imports
from app.logger import setup_logging
logger = setup_logging()

from app.main import app
from app.config import settings

def open_browser():
    """Open the browser after a short delay to let the server start."""
    time.sleep(2)
    url = f"http://127.0.0.1:{settings.APP_PORT}"
    logger.info(f"[startup] Opening browser to {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        logger.error(f"[startup] Could not open browser: {e}")
        logger.info(f"[startup] Open manually: {url}")

if __name__ == "__main__":
    import uvicorn

    # Open browser in a separate thread
    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    logger.info(f"[LLamaStudio] Starting on http://127.0.0.1:{settings.APP_PORT}")
    logger.info(f"[LLamaStudio] llama-server will start automatically")

    # Run uvicorn
    uvicorn.run(
        app,
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        log_level="info",
    )
