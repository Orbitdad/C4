from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

_ACTION_LOGGER_NAME = "jarvis.actions"


def configure_logging(log_file: Path) -> None:
    """
    Configure application-wide logging.

    - Console handler for human-readable logs.
    - Rotating file handler for persistent logs.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def get_action_logger() -> logging.Logger:
    return logging.getLogger(_ACTION_LOGGER_NAME)


def log_action(action_type: str, details: Dict[str, Any]) -> None:
    """
    Log a structured action record for transparency and auditing.
    """
    logger = get_action_logger()
    logger.info("%s %s", action_type, details)

