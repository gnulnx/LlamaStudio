import logging
import sys
from pathlib import Path
from .config import settings

def setup_logging():
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    app_log_path = log_dir / "app.log"

    # Define a clean formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    file_handler = logging.FileHandler(app_log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Root Logger
    root_logger = logging.getLogger()
    
    # Clear root handlers to avoid duplicates
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

    # Redirect third-party loggers we care about to both file and console
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "llamastudio"):
        l = logging.getLogger(logger_name)
        l.handlers = []
        l.addHandler(file_handler)
        l.addHandler(console_handler)
        l.setLevel(logging.INFO)
        l.propagate = False

    app_logger = logging.getLogger("llamastudio")
    app_logger.info("Application logging initialized. File: %s", app_log_path)
    return app_logger

# Export pre-initialized logger for easy importing
logger = logging.getLogger("llamastudio")
