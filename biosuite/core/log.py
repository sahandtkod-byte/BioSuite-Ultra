"""
BioSuite Ultra — Centralized Logging Configuration.

Provides structured logging for all modules with configurable
handlers, formatters, and log levels. Replaces print() with
proper Python logging throughout the codebase.

Usage:
    from biosuite.core.log import get_logger
    logger = get_logger(__name__)
    logger.info("Analysis started")
    logger.warning("Using builtin fallback")
    logger.error("File not found: %s", filepath)
"""
import logging
import os
import sys
from datetime import datetime

VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")

_loggers = {}
_configured = False


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


def _setup_root():
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger('biosuite')
    root.setLevel(VERBOSE)

    # Console handler with colors
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG)
    console.setFormatter(ColorFormatter())
    root.addHandler(console)

    # File handler (optional, to ~/.biosuite/logs/)
    try:
        log_dir = os.path.join(os.path.expanduser('~'), '.biosuite', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"biosuite_{datetime.now().strftime('%Y%m%d')}.log")
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s [%(name)s] %(funcName)s:%(lineno)d %(message)s'
        ))
        root.addHandler(fh)
    except Exception:
        pass  # File logging is optional


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
