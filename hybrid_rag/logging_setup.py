"""Centralized logging setup for the Enterprise Global RAG system.

Loads rotating-file and console handlers from ``config/logging_config.json``,
creating a sensible default configuration if the file does not exist yet.
"""

import json
import logging
import logging.config
from pathlib import Path

# Project-root-relative paths so logging works regardless of the caller's CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "logging_config.json"
LOG_DIR = PROJECT_ROOT / "logs"

# Guard to avoid re-applying dictConfig on every module import
_CONFIGURED = False

# Default dictionary configuration written to disk when config/logging_config.json is missing
DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "console_clean": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "console_clean",
            "stream": "ext://sys.stdout",
        },
        "file_rotating": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": (LOG_DIR / "app.log").as_posix(),
            "maxBytes": 5242880,  # 5 MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "standard",
            "filename": (LOG_DIR / "errors.log").as_posix(),
            "maxBytes": 5242880,  # 5 MB
            "backupCount": 3,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file_rotating", "error_file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "urllib3": {"level": "WARNING", "propagate": True},
        "httpcore": {"level": "WARNING", "propagate": True},
        "httpx": {"level": "WARNING", "propagate": True},
        "openai": {"level": "WARNING", "propagate": True},
        "chromadb": {"level": "WARNING", "propagate": True},
        "PIL": {"level": "WARNING", "propagate": True},
        "watchdog": {"level": "WARNING", "propagate": True},
    },
}


def setup_logging(config_path: Path = CONFIG_FILE, force: bool = False) -> None:
    """Initializes logging once from ``config_path`` (creating defaults if missing)."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if config_path.is_file():
        # Load external JSON logging config
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError:
            logging.basicConfig(level=logging.INFO)
            logging.error(
                f"Malformed logging config in '{config_path}'. Reverting to basicConfig.",
                exc_info=True,
            )
            _CONFIGURED = True
            return
    else:
        # Persist defaults when no config file exists yet
        config = DEFAULT_LOGGING_CONFIG
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    # Resolve relative handler filenames against PROJECT_ROOT so logs always land in /logs
    for handler in config.get("handlers", {}).values():
        filename = handler.get("filename")
        if filename and not Path(filename).is_absolute():
            handler["filename"] = (PROJECT_ROOT / filename).as_posix()

    logging.config.dictConfig(config)
    _CONFIGURED = True
