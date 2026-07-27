"""
BioSuite Ultra — Centralized Logging Configuration.

Provides structured logging for all modules with configurable
handlers, formatters, log levels, and rotation.

Usage:
    from biosuite.core.log import get_logger
    logger = get_logger(__name__)
    logger.info("Analysis started")
    logger.warning("Using builtin fallback")
    logger.error("File not found: %s", filepath)
"""
import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional

VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")

_loggers = {}
_configured = False

# Log level from environment
_env_log_level = os.environ.get("BIOSUITE_LOG_LEVEL", "INFO").upper()
NUMPY_LOG_LEVEL = os.environ.get("BIOSUITE_NUMPY_LOG_LEVEL", "WARNING")


class ColorFormatter(logging.Formatter):
    """Colored console formatter for terminal output."""
    COLORS = {
        5: '\033[90m',    # VERBOSE: gray
        10: '\033[36m',   # DEBUG: cyan
        20: '\033[92m',   # INFO: green
        30: '\033[93m',   # WARNING: yellow
        40: '\033[91m',   # ERROR: red
        50: '\033[1;91m', # CRITICAL: bold red
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelno, '')
        level = f"{color}{record.levelname:<8}{self.RESET}"
        msg = f"{color}{record.msg}{self.RESET}"
        ts = datetime.now().strftime('%H:%M:%S')
        name = record.name if hasattr(record, 'name') else ''
        return f"{ts} {level} [{name}] {msg}"


class JsonFormatter(logging.Formatter):
    """JSON structured formatter for log aggregation."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, 'correlation_id'):
            log_entry["correlation_id"] = record.correlation_id
        return json.dumps(log_entry, ensure_ascii=False)


def _setup_root():
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger('biosuite')
    root.setLevel(VERBOSE)

    # Console handler with colors
    console = logging.StreamHandler(sys.stderr)
    level_name = _env_log_level
    console.setLevel(getattr(logging, level_name, logging.INFO))
    console.setFormatter(ColorFormatter())
    root.addHandler(console)

    # File handler with rotation (10MB, keep 5 backups)
    try:
        log_dir = os.path.join(os.path.expanduser('~'), '.biosuite', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "biosuite.log")
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10_000_000, backupCount=5, encoding='utf-8'
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s [%(name)s] %(funcName)s:%(lineno)d %(message)s'
        ))
        root.addHandler(fh)
    except Exception:
        pass  # File logging is optional

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)

    # NumPy log level
    try:
        logging.getLogger("numpy").setLevel(getattr(logging, NUMPY_LOG_LEVEL, logging.WARNING))
    except Exception:
        pass


def get_logger(name: str = None) -> "logging.Logger":
    """Get a logger for a module.

    Args:
        name: Module name (e.g., 'biosuite.core.sequence').
              If None, returns the root biosuite logger.

    Returns:
        logging.Logger instance.
    """
    if name is None:
        name = 'biosuite'
    elif not name.startswith('biosuite'):
        name = f'biosuite.{name}'

    if name not in _loggers:
        _setup_root()
        _loggers[name] = logging.getLogger(name)

    return _loggers[name]


def log_performance(func_name: str, elapsed_ms: float, details: str = "") -> None:
    """Log performance metrics for an analysis step."""
    msg = f"{func_name} completed in {elapsed_ms:.1f}ms"
    if details:
        msg += f" ({details})"
    get_logger('biosuite.performance').info(msg)


def log_warning(message: str, module: str = None) -> None:
    """Log a warning message."""
    get_logger(module or 'biosuite').warning(message)


def log_error(message: str, exc: Exception = None, module: str = None) -> None:
    """Log an error with optional exception info."""
    logger = get_logger(module or 'biosuite')
    if exc:
        logger.error("%s: %s", message, exc, exc_info=True)
    else:
        logger.error(message)


def log_step(module: str, function: str, status: str = "started", details: str = "") -> None:
    """Log an analysis step (replaces print-based step logging)."""
    logger = get_logger(f'biosuite.{module}')
    msg = f"{function}: {status}"
    if details:
        msg += f" ({details})"
    if status == "started":
        logger.debug(msg)
    elif status == "completed":
        logger.info(msg)
    elif status == "failed":
        logger.error(msg)
    else:
        logger.info(msg)
