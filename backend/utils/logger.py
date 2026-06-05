"""
utils/logger.py
───────────────
Configures a structured, rotating file + console logger for the app.
Import `logger` anywhere in the project.
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def configure_logger(log_level: str = "DEBUG", log_file: str = "logs/app.log") -> logging.Logger:
    """
    Set up and return the root application logger.

    Parameters
    ----------
    log_level : str
        Logging level string (DEBUG / INFO / WARNING / ERROR / CRITICAL).
    log_file : str
        Path to the rotating log file.
    """
    # Ensure the logs directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.DEBUG)

    # ── Root logger ──────────────────────────────────────────────────────────
    root_logger = logging.getLogger("ai_research")
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers if called multiple times
    if root_logger.handlers:
        return root_logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # ── Rotating file handler (10 MB × 5 backups) ────────────────────────────
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    return root_logger


# Module-level logger for direct imports
logger = logging.getLogger("ai_research")
