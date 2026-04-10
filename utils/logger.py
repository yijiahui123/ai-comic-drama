"""Unified logging module with colored console output and file recording."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# ANSI color codes
_COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",
}

LOG_DIR = Path("logs")


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes to log level names."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, _COLORS["RESET"])
        reset = _COLORS["RESET"]
        record.levelname = f"{color}{record.levelname:<8}{reset}"
        return super().format(record)


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Return a named logger with colored console output and optional file handler.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        log_file: Optional filename inside the ``logs/`` directory.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — return existing instance.
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler (colored)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_fmt = ColoredFormatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler (plain text)
    if log_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOG_DIR / log_file
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_pipeline_logger(project_id: str) -> logging.Logger:
    """Return a logger that writes to a project-specific log file.

    Args:
        project_id: Unique identifier for the current pipeline run.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f"pipeline_{project_id}_{timestamp}.log"
    return get_logger(f"pipeline.{project_id}", log_file=log_file)
