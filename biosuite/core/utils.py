"""Utility functions: config, session, safe input, file loading, theme, helpers."""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from biosuite.core.log import get_logger

logger = get_logger(__name__)


class PerformanceWarning(UserWarning):
    """Warning emitted when a pure-Python fallback is used instead of an external tool."""
    pass


class FormatWarning(UserWarning):
    """Warning for file format issues or deprecated usage."""
    pass


# ── Threading Utilities ──────────────────────────────────────────────────────

def run_in_background(func: Any, *args: Any, callback: Optional[Any] = None, **kwargs: Any) -> threading.Thread:
    """Run a function in a background thread with an optional callback.

    Args:
        func: function to run.
        *args: positional arguments.
        callback: called with the result when the function returns.  **It is
            called from the worker thread**, not the main one — a GUI callback
            must hand the value to the Tk event loop itself (`widget.after`)
            rather than touching widgets here.
        **kwargs: keyword arguments.

    Returns:
        The started, daemon :class:`threading.Thread`.

    When *func* raises, the traceback is logged and the callback is called with
    ``None``: a caller that must tell "failed" from "succeeded with no data"
    should raise nothing and return a result object, or read the exception off the
    thread's own logging.  There is no timeout here — this starts a thread and
    hands it back; a bounded wait is the caller's `Thread.join(timeout)` or an
    executor.
    """
    def _report(value: Any) -> None:
        """Hand the outcome to the callback without letting its own error kill
        the worker — the task result is already decided by then."""
        if not callback:
            return
        try:
            callback(value)
        except Exception as exc:
            logger.error("Background callback %s failed: %s: %s",
                         getattr(callback, '__name__', repr(callback)),
                         type(exc).__name__, exc, exc_info=True)

    def _wrapper() -> Any:
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            logger.error("Background task %s failed: %s: %s",
                         getattr(func, '__name__', repr(func)),
                         type(exc).__name__, exc, exc_info=True)
            _report(None)
            return
        _report(result)

    thread = threading.Thread(target=_wrapper, daemon=True)
    thread.start()
    return thread


def run_with_progress(func: Any, progress_callback: Optional[Any] = None, *args: Any, **kwargs: Any) -> Any:
    """Run a function with progress updates.

    Args:
        func: function to run (must accept progress_callback kwarg).
        progress_callback: function to call with progress (0.0 to 1.0).
        *args: positional arguments.
        **kwargs: keyword arguments.

    Returns:
        Result of func.
    """
    return func(*args, progress_callback=progress_callback, **kwargs)


class CachedResult:
    """Cache function results to avoid recomputation.

    Thread-safe: one lock guards the cache and is held while the wrapped function
    runs, so concurrent callers of a key compute once (F2-5).  Unrelated keys
    queue behind a slow call; use an executor for fan-out.
    """

    def __init__(self, func: Any, maxsize: int = 128, ttl: Optional[float] = None) -> None:
        """Initialize CachedResult.

        Args:
            func: Function to cache.
            maxsize: Maximum number of cached entries; an integer of at least 1.
            ttl: Time-to-live in seconds. Entries older than this are evicted
                 on access. None means no expiration.

        Raises:
            ValueError: if *maxsize* is not a positive integer, or *ttl* is not
                ``None`` or a positive finite number.  A ``ttl`` of 0 or less made
                every entry expire the instant it was stored, so the cache looked
                wired up and recomputed everything.
        """
        if isinstance(maxsize, bool) or not isinstance(maxsize, numbers.Integral) \
                or maxsize < 1:
            raise ValueError(f"CachedResult maxsize must be an integer of at least 1, "
                             f"got {maxsize!r}.")
        if ttl is not None:
            if isinstance(ttl, bool) or not isinstance(ttl, numbers.Real) \
                    or not math.isfinite(ttl) or ttl <= 0:
                raise ValueError(f"CachedResult ttl must be None or a positive finite "
                                 f"number of seconds, got {ttl!r}.")
        self.func = func
        self.cache = {}
        self.maxsize = int(maxsize)
        self.access_order = []
        self.ttl = ttl  # seconds, or None for no expiration
        # RLock, not Lock: the wrapped function may itself call this cache.
        self._lock = threading.RLock()

    @staticmethod
    def _key_part(value: Any) -> Any:
        """Return a collision-resistant, hashable representation of *value*.

        ``str(value)`` is not usable as a cache key for scientific data: numpy
        abbreviates large arrays with an ellipsis, so two completely different
        10,000-element arrays stringified identically and the cache returned
        one call's result for the other's input.  Arrays are hashed over their
        raw bytes together with dtype and shape.
        """
        buffer = getattr(value, "tobytes", None)
        if buffer is not None and hasattr(value, "dtype") and hasattr(value, "shape"):
            try:
                import numpy as _np
                contiguous = _np.ascontiguousarray(value)
                digest = hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest()
                return ("ndarray", str(contiguous.dtype), contiguous.shape, digest)
            except Exception:  # pragma: no cover - defensive
                pass
        if isinstance(value, (bytes, bytearray, memoryview)):
            return ("bytes", hashlib.blake2b(bytes(value), digest_size=16).hexdigest())
        if isinstance(value, (str, int, float, bool, type(None))):
            return (type(value).__name__, value)
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, tuple(CachedResult._key_part(v) for v in value))
        if isinstance(value, dict):
            return ("dict", tuple(sorted(
                (str(k), CachedResult._key_part(v)) for k, v in value.items())))
        if isinstance(value, (set, frozenset)):
            return ("set", tuple(sorted(str(CachedResult._key_part(v)) for v in value)))
        to_dict = getattr(value, "to_dict", None)          # pandas objects
        if callable(to_dict):
            try:
                return ("to_dict", CachedResult._key_part(to_dict()))
            except Exception:  # pragma: no cover - defensive
                pass
        return ("repr", type(value).__name__, repr(value))

    @classmethod
    def make_key(cls, args: tuple, kwargs: dict) -> tuple:
        """Build the cache key for a call."""
        return (tuple(cls._key_part(a) for a in args),
                tuple(sorted((k, cls._key_part(v)) for k, v in kwargs.items())))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function, using cache if available."""
        key = self.make_key(args, kwargs)
        with self._lock:
            now = time.monotonic()
            if key in self.cache:
                result, created_at = self.cache[key]
                if self.ttl is not None and (now - created_at) > self.ttl:
                    # Entry expired — evict it
                    del self.cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
                else:
                    # Genuine LRU: a hit refreshes recency.
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.access_order.append(key)
                    return result
            # Auto-clean all expired entries on access
            self._evict_expired(now)
            result = self.func(*args, **kwargs)
            while len(self.cache) >= self.maxsize and self.access_order:
                oldest = self.access_order.pop(0)
                self.cache.pop(oldest, None)
            if len(self.cache) >= self.maxsize:
                # access_order exhausted but entries remain (should not happen);
                # drop the oldest insertion rather than raising IndexError.
                self.cache.pop(next(iter(self.cache)), None)
            self.cache[key] = (result, now)
            self.access_order.append(key)
            return result

    def is_expired(self, key: tuple) -> bool:
        """Check if a specific cache entry has expired.

        Args:
            key: A key built by :meth:`make_key` — a tuple, not the string a
                caller might guess at; a key that is not in the cache counts as
                expired, which is how a wrong-typed key reads as "nothing stored".

        Returns:
            True if the entry doesn't exist or has expired; False otherwise.
        """
        with self._lock:
            if key not in self.cache:
                return True
            if self.ttl is None:
                return False
            _, created_at = self.cache[key]
            return (time.monotonic() - created_at) > self.ttl

    def _evict_expired(self, now: Optional[float] = None) -> None:
        """Remove all expired entries from the cache."""
        with self._lock:
            if self.ttl is None or not self.cache:
                return
            if now is None:
                now = time.monotonic()
            expired_keys = [
                key for key, (_, created_at) in self.cache.items()
                if (now - created_at) > self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self.cache)

def get_app_dir() -> str:
    """Get the application directory, works with PyInstaller."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

APP_DIR = get_app_dir()

def center_window(win: Any, width: int, height: int) -> None:
    """Center a Tk window on the primary display with margins.

    Args:
        win: Tk/Toplevel window instance.
        width: Desired window width in px.
        height: Desired window height in px.
    """
    win.update_idletasks()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry(f'{width}x{height}+{x}+{y}')

_interrupted = False
_interaction_counter = 0

DEFAULT_CONFIG = {
    "theme": "dark-green",
    "default_dpi": 180,
    "save_format": "png",
    "interactive": False,
    "downsample_threshold": 5000,
    "quiet": False,
    "api_keys": {
        "ncbi_email": "",
        "ncbi_api_key": "",
        "uniprot_email": "",
        "kegg_email": "",
        "alphafold_email": "",
    }
}

def user_config_dir() -> str:
    """Directory holding per-user BioSuite state.

    Resolution order:

    1. ``BIOSUITE_CONFIG_DIR`` (explicit override, used by tests and by
       containers that mount a writable volume),
    2. ``%APPDATA%\\BioSuite`` on Windows / ``$XDG_CONFIG_HOME/biosuite`` or
       ``~/.config/biosuite`` elsewhere.

    The configuration deliberately no longer lives next to the source tree:
    ``api_keys`` written by the GUI/CLI (and by the test-suite) ended up inside
    the *git-tracked* ``biosuite_config.json``, which is how test residue was
    committed and how a real user key would be published by ``git add -A``.
    """
    override = os.environ.get("BIOSUITE_CONFIG_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "BioSuite")
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "biosuite")


def get_config_file() -> str:
    """Absolute path of the active config file (re-read on every call)."""
    return os.path.join(user_config_dir(), "biosuite_config.json")


def get_session_file() -> str:
    """Absolute path of the active session file (re-read on every call)."""
    return os.path.join(user_config_dir(), "biosuite_session.json")


#: Legacy location kept for one-time read-only migration of existing installs.
LEGACY_CONFIG_FILE = os.path.join(APP_DIR, "biosuite_config.json")

def load_config() -> dict:
    """Load user config from the user configuration directory.

    Falls back to a read-only migration from the historical in-repository
    location so existing installations keep their settings.

    A file that is not a JSON *object* — a list, a bare string, ``null``, a
    truncated write — is reported and skipped rather than merged, and the
    defaults are used.  This runs at import time (``config = load_config()``),
    so a value that reached ``{**DEFAULT_CONFIG, **value}`` and raised took the
    whole package down with it.

    Returns:
        dict: Parsed configuration merged over :data:`DEFAULT_CONFIG`.
    """
    for path in (get_config_file(), LEGACY_CONFIG_FILE):
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding="utf-8") as f:
                stored = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.warning("Could not read %s (%s: %s); trying the next location.",
                           path, type(exc).__name__, exc)
            continue
        if not isinstance(stored, dict):
            logger.warning("%s holds a JSON %s, not an object; ignoring it and using "
                           "the defaults.", path, type(stored).__name__)
            continue
        return {**DEFAULT_CONFIG, **stored}
    return DEFAULT_CONFIG.copy()

def save_config(cfg: Dict[str, Any]) -> None:
    """Persist the given config dict to the user configuration directory.

    Writes atomically (temp file + replace) and reports failures instead of
    silently discarding them, so a user never believes an API key was saved
    when it was not.

    Raises:
        OSError: if the configuration could not be written.
    """
    path = get_config_file()
    directory = os.path.dirname(path)
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".biosuite_config", dir=directory)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(cfg, f, indent=2)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        try:
            os.chmod(path, 0o600)      # the file may hold API keys
        except OSError as exc:
            # Not fatal, but the user must know their key file is readable by
            # others rather than be told the save succeeded outright.
            logger.warning(
                "Saved %s but could not restrict its permissions to 0600 (%s). "
                "If it contains an API key, tighten the permissions manually.",
                path, exc)
    except OSError as exc:
        logger.warning("Could not save configuration to %s: %s", path, exc)
        raise

def load_session() -> dict :
    """Load last saved session (recent files, last tab) if present."""
    session_file = get_session_file()
    if os.path.exists(session_file):
        try:
            with open(session_file, 'r', encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}

def save_session(session_data: Dict[str, Any]) -> None:
    """Save session state dict to disk for next-launch restore."""
    try:
        os.makedirs(user_config_dir(), exist_ok=True)
        with open(get_session_file(), 'w', encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
    except OSError as exc:
        # An explicit save that silently does nothing leaves the user believing
        # their session was stored. Still does not raise, so it cannot block
        # GUI shutdown.
        logger.warning("Could not save the session to %s: %s",
                       get_session_file(), exc)

def autosave_session(session_data: Optional[Dict[str, Any]] = None) -> None:
    """Best-effort autosave that silently ignores write errors
    so GUI shutdown is never blocked by I/O failures.
    """
    if session_data is None:
        session_data = session
    save_session(session_data)

config = load_config()
session = load_session()


# ── API Key Management ──────────────────────────────────────────────────────

def get_api_key(service: str) -> Optional[str]:
    """Get API key for a service (ncbi, uniprot, kegg, alphafold, etc.)."""
    keys = config.get('api_keys', {})
    return keys.get(service, '')

def set_api_key(service: str, key: str) -> None:
    """Save API key for a service."""
    if 'api_keys' not in config:
        config['api_keys'] = {}
    config['api_keys'][service] = key
    save_config(config)

def prompt_api_key(service: str, description: str = "") -> Optional[str]:
    """Prompt user to enter API key if not already set."""
    existing = get_api_key(service)
    if existing:
        return existing
    if config.get('quiet', False):
        return ''
    import sys
    if not sys.stdin.isatty():
        # GUI / CI / piped stdin: an interactive input() would hang forever
        # or die with EOFError — silently skip instead.
        logger.info(f"\n  API key for {service} not set (non-interactive session).")
        return ''
    logger.info(f"\n  API key required for: {service}")
    if description:
        logger.info(f"  {description}")
    key = input("  Enter API key (or blank to skip): ").strip()
    if key:
        set_api_key(service, key)
        logger.info(f"  API key saved for {service}.")
    return key

#: Theme names `set_theme` knows: the two dark palettes the GUI offers plus the
#: three the CLI's theme menu writes into the config.
DARK_THEMES = ('dark', 'dark-green', 'dark-purple')
LIGHT_THEMES = ('light', 'light-blue')
KNOWN_THEMES = frozenset(DARK_THEMES + LIGHT_THEMES)


def set_theme(choice: str) -> None:
    """Apply a named theme to the running application.

    Args:
        choice: one of :data:`DARK_THEMES` or :data:`LIGHT_THEMES`.  Anything
            else is reported with the names that do exist and changes nothing:
            measured before, an unrecognised name silently got the light palette,
            so a typo in the config ('light_cyan') read as a deliberate choice.
    """
    if not isinstance(choice, str) or (choice not in DARK_THEMES
                                       and choice not in LIGHT_THEMES):
        logger.warning("Unknown theme %r; the plots keep their current colours. "
                       "Known themes: %s.", choice, ', '.join(sorted(KNOWN_THEMES)))
        return
    import matplotlib.pyplot as plt

    is_dark = choice in DARK_THEMES
    if not is_dark:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                plt.style.use('default')
        plt.rcParams['figure.facecolor'] = '#e6f7fa'
        plt.rcParams['axes.facecolor'] = '#f0fcfd'
        plt.rcParams['text.color'] = '#006064'
        plt.rcParams['axes.labelcolor'] = '#006064'
        plt.rcParams['xtick.color'] = '#006064'
        plt.rcParams['ytick.color'] = '#006064'
        plt.rcParams['axes.edgecolor'] = '#80ced6'
        plt.rcParams['grid.color'] = '#b2ebf2'
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['font.size'] = 11
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['axes.titlesize'] = 13
    else:
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            try:
                plt.style.use('seaborn-darkgrid')
            except OSError:
                plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#121212'
        plt.rcParams['axes.facecolor'] = '#1a1a1a'
        plt.rcParams['text.color'] = '#00ff99'
        plt.rcParams['axes.labelcolor'] = '#00ff99'
        plt.rcParams['xtick.color'] = '#00ff99'
        plt.rcParams['ytick.color'] = '#00ff99'
        plt.rcParams['axes.edgecolor'] = '#2c2c2c'
        plt.rcParams['grid.color'] = '#333333'
        plt.rcParams['grid.alpha'] = 0.4
        plt.rcParams['font.size'] = 11
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['axes.titlesize'] = 13

def _session_default(key: Optional[str], cast, current):
    """A stored session value to offer instead of *current*, or None.

    The session file is JSON a user can edit and an older build can write, so a
    value that is not a number is ignored rather than returned by a function that
    promises a float or an int.
    """
    if not key or key not in session:
        return None
    stored = session[key]
    if isinstance(stored, bool) or not isinstance(stored, numbers.Real) \
            or not math.isfinite(stored):
        logger.info("Stored value for '%s' is %r, not a finite number; using the "
                    "default instead.", key, stored)
        return None
    return cast(stored)


def _stored_list(key: Optional[str]):
    """The list stored under *key*, or None when nothing usable is stored there.

    A number or a string under a list key is reported and ignored.  The old code
    fed whatever it found to `",".join(str(x) for x in ...)`, which raised
    `TypeError: 'int' object is not iterable` out of the prompt helper, and a
    stored string such as `"1,2"` was shown one character at a time as
    `1,,,2`.
    """
    if not key or key not in session:
        return None
    stored = session[key]
    if not isinstance(stored, (list, tuple)):
        logger.info("Stored value for '%s' is %r, not a list; no previous answer "
                    "is shown.", key, stored)
        return None
    return stored


def safe_float_input(prompt: str, default: float, key: Optional[str] = None) -> float:
    """Prompt for a float with validation and optional persistence key.

    Returns:
        float: Parsed value or the supplied default on invalid input.  A value
            that is not finite (`inf`, `-inf`, `nan`, `1e999`) is rejected the
            same way as a typo — measured before, all four were returned as
            answers, and a threshold of NaN silently turns every comparison on
            the far side of it False.  A refused answer is not stored: the
            session keeps the value it already had.  An empty line gives back
            *default* untouched, so a caller that passes `1` still gets `1` and
            not `1.0` — plotting titles showed the change.  Like
            `safe_int_input`, an empty line counts as an answer and is stored
            under *key*.
    """
    global _interaction_counter
    if config.get('quiet', False):
        return default
    stored = _session_default(key, float, default)
    if stored is not None:
        default = stored
    try:
        val = input(prompt).strip()
        if val == '':
            val = default
        else:
            val = float(val)
            if not math.isfinite(val):
                logger.info(f"Not a finite number. Using default: {default}")
                return default
        if key:
            session[key] = val
            _interaction_counter += 1
            if _interaction_counter % 10 == 0:
                autosave_session(session)
        return val
    except (ValueError, TypeError):
        logger.info(f"Invalid. Using default: {default}")
        return default

def safe_int_input(prompt: str, default: int, key: Optional[str] = None) -> int:
    """Prompt for an int with validation and optional persistence key."""
    global _interaction_counter
    if config.get('quiet', False):
        return default
    stored = _session_default(key, int, default)
    if stored is not None:
        default = stored
    try:
        val = input(prompt).strip()
        if val == '':
            val = default
        else:
            val = int(val)
        if key:
            session[key] = val
            _interaction_counter += 1
            if _interaction_counter % 10 == 0:
                autosave_session(session)
        return val
    except (ValueError, TypeError):
        logger.info(f"Invalid. Using default: {default}")
        return default

def safe_list_input(prompt: str, cast: Any = float, key: Optional[str] = None) -> Optional[List[Any]]:
    """Prompt once for a comma-separated list and cast every item.

    Returns:
        list or None: the cast items, or ``None`` when the line was blank (and no
        list is stored under *key*) or any item failed to cast.  The caller's
        default is *its* decision — this function has none to fall back on, and it
        does not re-prompt, both of which the earlier docstring claimed.
    """
    global _interaction_counter
    if config.get('quiet', False):
        return None
    stored = _stored_list(key)
    if stored is not None:
        prompt = f"{prompt} (last: {','.join(str(x) for x in stored)}): "
    else:
        prompt = f"{prompt}: "
    raw = input(prompt).strip()
    if not raw:
        if stored is not None:
            return list(stored)
        logger.info("No list entered.")
        return None
    items = []
    for piece in raw.split(','):
        try:
            items.append(cast(piece.strip()))
        except (ValueError, TypeError):
            logger.info("'%s' is not a valid %s; no list was returned.",
                        piece.strip(), getattr(cast, '__name__', str(cast)))
            return None
    if key:
        session[key] = items
        _interaction_counter += 1
        if _interaction_counter % 10 == 0:
            autosave_session(session)
    return items

def load_dataframe_safe(filepath: str) -> pd.DataFrame:
    """Read CSV/TSV/Excel into a DataFrame without raising.

    Returns:
        pandas.DataFrame or None: Data, or None with error logged.
    """
    if not os.path.exists(filepath):
        logger.info(f"File not found: {filepath}")
        return None
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath)
        elif ext == '.json':
            df = pd.read_json(filepath)
        elif ext == '.txt':
            df = pd.read_csv(filepath, sep=None, engine='python')
        elif ext == '.parquet':
            df = pd.read_parquet(filepath)
        elif ext in ['.h5', '.hdf5']:
            df = pd.read_hdf(filepath)
        else:
            df = pd.read_csv(filepath, sep=None, engine='python')
        if df.empty:
            logger.info("File is empty.")
            return None
        return df
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return None


def load_dataframe_safe_interactive(filepath: str) -> pd.DataFrame:
    """Load dataframe and optionally show summary stats (interactive use only)."""
    df = load_dataframe_safe(filepath)
    if df is not None and not config.get('quiet', False):
        try:
            show_stats = input("Show data summary (describe)? (y/n): ").strip().lower()
            if show_stats == 'y':
                logger.info(df.describe())
        except EOFError:
            pass
    return df

def maybe_downsample(x: List[Any], y: List[Any], max_points: Optional[int] = None) -> Tuple[List[Any], List[Any]]:
    """Uniformly downsample long series for responsive plotting.

    Keeps the first and last points and picks the rest at an even stride, so
    the visible shape of the curve is preserved and the result is identical on
    every run.

    The previous implementation used an unseeded ``np.random.choice`` without
    ``sort``: it returned the samples in random order (scrambling any line
    plot), dropped the endpoints, produced a different picture on every run,
    and raised ``TypeError`` when given plain Python lists despite the
    ``List[Any]`` signature.

    Args:
        x: x values (list, tuple, ndarray or Series).
        y: y values, same length as *x*.
        max_points: maximum number of points to keep.

    Returns:
        The downsampled ``(x, y)`` pair, in the same order as the input and of
        the same type family as the input.

    Raises:
        ValueError: if *x* and *y* have different lengths, or *max_points* is not
            an integer of at least 2.  With no *max_points* the configured
            ``downsample_threshold`` is used, and one that is not an integer
            (``null`` in the config file, say) is reported and replaced by 5000
            rather than crashing every plot.
    """
    if len(x) != len(y):
        raise ValueError(
            f"x and y must be the same length, got {len(x)} and {len(y)}")
    if max_points is None:
        max_points = config.get('downsample_threshold', 5000)
        if isinstance(max_points, bool) or not isinstance(max_points, numbers.Integral):
            logger.info("Config 'downsample_threshold' is %r, not an integer; "
                        "using 5000 points.", max_points)
            max_points = 5000
    if isinstance(max_points, bool) or not isinstance(max_points, numbers.Integral):
        raise ValueError(f"max_points must be an integer of at least 2, got "
                         f"{max_points!r} ({type(max_points).__name__}); a fractional "
                         f"value would be truncated, not rounded.")
    max_points = int(max_points)
    if max_points < 2:
        raise ValueError(f"max_points must be at least 2, got {max_points}")
    n = len(x)
    if n <= max_points:
        return x, y
    # Evenly spaced indices including both endpoints; deterministic.
    indices = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    logger.info(f"Downsampled from {n} to {len(indices)} points.")
    return _take(x, indices), _take(y, indices)


def _take(values, indices):
    """Index *values* by position, keeping the type the caller passed.

    A pandas object must be sliced with `iloc` — `series[[0, 1]]` is label-based and
    would raise on any frame whose index is not 0..n-1 — while a list or tuple is
    rebuilt through its own constructor.
    """
    if isinstance(values, np.ndarray):
        return values[indices]
    if hasattr(values, "iloc"):                      # pandas Series / DataFrame
        return values.iloc[indices]
    if isinstance(values, (list, tuple)):
        return type(values)(values[i] for i in indices)
    return np.asarray(values)[indices]


def apply_glass_ax(ax: Any) -> None:
    """Style an Axes with the dark glass look used across BioSuite plots."""
    from matplotlib.patches import FancyBboxPatch

    try:
        rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                              transform=ax.transAxes, clip_on=False,
                              facecolor='none', edgecolor=ax.spines['top'].get_edgecolor(),
                              linewidth=0.8, alpha=0.3)
        ax.add_patch(rect)
    # KeyError: polar axes have no 'top'/'right' spines (circos plots).
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        # Cosmetic only, but a programming error in a caller used to vanish here
        # with no trace; DEBUG keeps it findable without spraying every plot.
        logger.debug("Glass frame not applied to this Axes (%s: %s)",
                     type(exc).__name__, exc)

def ask_save_plot(default_name: str, save_format: str, dpi: int, pdf: Optional[str] = None, story: Optional[List[str]] = None) -> Optional[str]:
    """Save the current figure, then close it — in every branch, cancellation
    and save failure included.  These plots are written to disk and never shown,
    so an unclosed figure is pinned for the life of the process (F2-3).  A caller
    that wants to display a figure keeps the Figure the plot function returns
    instead of routing it through here.

    Returns:
        str or None: Chosen file path, None when cancelled.
    """
    import matplotlib.pyplot as plt

    fig = plt.gcf()
    try:
        if pdf is not None:
            try:
                plt.tight_layout()
                pdf.savefig()
                logger.info(f"   Added to PDF: {default_name}")
                if story is not None:
                    story.append(f"![{default_name}]({default_name}.png)")
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("Could not add %s to the PDF report: %s",
                               default_name, exc)
            return
        try:
            save = input("Save this plot? (y/n): ").strip().lower()
        except EOFError:
            save = 'n'
        if save == 'y':
            name = input(f"   Filename (default {default_name}.{save_format}): ").strip()
            if not name:
                name = default_name
            # A user who types "volume.png" means volume.png, not volume.png.png.
            target = name if name.lower().endswith(f".{save_format.lower()}") \
                else f"{name}.{save_format}"
            try:
                plt.tight_layout()
                plt.savefig(target, dpi=dpi,
                            bbox_inches='tight', facecolor=plt.rcParams['figure.facecolor'])
                logger.info(f"   Saved as {target}")
            except OSError as exc:
                logger.error("Could not write %s: %s: %s", target, type(exc).__name__, exc)
    finally:
        plt.close(fig)

def report_boxplot_stats(data: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Dict[str, float]]:
    """Compute boxplot summary stats per group for report panels.

    Returns:
        dict keyed by group, each holding ``median``, ``iqr`` and ``outliers``.
        Groups whose values are all missing are left out, and a frame missing one
        of the two columns is reported rather than raising ``KeyError`` out of a
        plot function.
    """
    missing = {group_col, value_col} - set(data.columns)
    if missing:
        logger.info("Boxplot statistics needs the columns %s; the data has %s.",
                    sorted(missing), list(data.columns))
        return {}
    stats_dict = {}
    for group in data[group_col].unique():
        vals = data[data[group_col]==group][value_col].dropna()
        if len(vals) > 0:
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3 - q1
            outliers = vals[(vals < q1 - 1.5*iqr) | (vals > q3 + 1.5*iqr)]
            stats_dict[group] = {
                'median': vals.median(),
                'iqr': iqr,
                'outliers': len(outliers)
            }
    if not stats_dict:
        logger.info("\nBoxplot Statistics: no group had a value to summarise.")
        return stats_dict
    logger.info("\nBoxplot Statistics:")
    for group, st in stats_dict.items():
        logger.info(f"   {group}: median={st['median']:.2f}, IQR={st['iqr']:.2f}, outliers={st['outliers']}")
    return stats_dict

def report_scatter_stats(x: list, y: list) -> Optional[dict]:
    """Pearson correlation and its p-value, for the scatter report.

    Returns:
        ``{'r': …, 'p': …}`` when the correlation can be computed, otherwise
        ``None`` with the reason logged: two series of different lengths, fewer
        than two points, or a constant series whose coefficient is not defined.
    """
    from scipy import stats as sp_stats
    if len(x) != len(y):
        logger.info("Correlation needs equal-length series, got %d and %d values.",
                    len(x), len(y))
        return None
    if len(x) < 2:
        logger.info("Insufficient data for correlation.")
        return None
    corr, pval = sp_stats.pearsonr(x, y)
    if not (math.isfinite(corr) and math.isfinite(pval)):
        # A constant series makes scipy answer NaN (with its own warning); printing
        # "r = nan" as a result is how a flat column turns into a fake statistic.
        logger.info("Pearson correlation is not defined for these series "
                    "(one of them is constant or holds no finite value).")
        return None
    logger.info(f"\nScatter Statistics: Pearson r = {corr:.3f}, p-value = {pval:.4f}")
    return {'r': corr, 'p': pval}

def report_volcano_stats(lfc: Any, pvals: Any, fc_thresh: float, p_thresh: float) -> Dict[str, int]:
    """Volcano-plot summary: up/down counts at the given cutoffs.

    Raises:
        ValueError: if the two series differ in length or are not 1-D — measured
            before, a length mismatch was broadcast into a plausible-looking
            ``{'up': 0, 'down': 0}`` answer instead of an error.
    """
    lfc = np.asarray(lfc, dtype=float)
    pvals = np.asarray(pvals, dtype=float)
    if lfc.ndim != 1 or pvals.ndim != 1:
        raise ValueError(f"report_volcano_stats needs 1-D fold-change and p-value "
                         f"series, got shapes {lfc.shape} and {pvals.shape}.")
    if len(lfc) != len(pvals):
        raise ValueError(f"report_volcano_stats got {len(lfc)} log2 fold changes but "
                         f"{len(pvals)} p-values; every gene needs both.")
    sig = (np.abs(lfc) >= fc_thresh) & (pvals < p_thresh)
    up = sig & (lfc > 0)
    down = sig & (lfc < 0)
    logger.info("\nVolcano Statistics:")
    logger.info(f"   Up-regulated genes: {np.sum(up)}")
    logger.info(f"   Down-regulated genes: {np.sum(down)}")
    logger.info(f"   Total significant: {np.sum(sig)}")
    return {'up': int(np.sum(up)), 'down': int(np.sum(down)), 'total': int(np.sum(sig))}

def report_pca_stats(pca: Any) -> Dict[str, Any]:
    """Explained-variance ratios from a fitted PCA object.

    Returns as many ``var1``/``var2`` entries as the fit has components; measured
    before, a one-component PCA raised ``IndexError: list index out of range``
    from inside the plot.
    """
    ratios = list(getattr(pca, 'explained_variance_ratio_', []))
    if not ratios:
        logger.info("\nPCA Statistics: the fitted PCA reports no variance ratios.")
        return {}
    logger.info("\nPCA Statistics:")
    for i, var in enumerate(ratios[:2]):
        logger.info(f"   PC{i+1} explains {var*100:.2f}% of variance")
    return {f'var{i+1}': value for i, value in enumerate(ratios[:2])}

def report_manhattan_stats(df: pd.DataFrame, threshold: float = 5e-8) -> Dict[str, int]:
    """Manhattan summary: significant SNPs at *threshold* (a p-value) and the top hit.

    Returns ``{}`` and logs what is missing when the frame cannot answer —
    measured before, a frame without a ``p`` column returned ``None`` from a
    function annotated ``-> Dict[str, int]``, and one without ``chrom``/``pos``
    raised ``KeyError`` while printing the report line.
    """
    missing = {'p', 'chrom', 'pos'} - set(df.columns)
    if missing:
        logger.info("Manhattan statistics needs the columns %s; the data has %s.",
                    sorted(missing), list(df.columns))
        return {}
    sig = df[df['p'] > -np.log10(threshold)]
    if sig.empty:
        logger.info("\nNo SNPs reached genome-wide significance.")
        return {'significant': 0, 'total': int(len(df))}
    top = sig.loc[sig['p'].idxmax()]
    logger.info("\nManhattan Statistics:")
    logger.info(f"   Significant SNPs: {len(sig)}")
    logger.info(f"   Top SNP: {top['chrom']}:{int(top['pos'])} with -log10(p)={top['p']:.2f}")
    return {'significant': int(len(sig)), 'total': int(len(df))}

def add_ttest_to_boxplot(data: pd.DataFrame, group_col: str, value_col: str, ax: Any) -> None:
    """Overlay a significance annotation on a boxplot Axes.

    Two groups get a Student's t-test, three or more a one-way ANOVA.  A test that
    cannot be answered is logged and leaves the Axes untouched: measured before, a
    single group raised ``TypeError: At least two samples are required; got 1.``,
    and a group of nothing but missing values drew ``t-test p = nan`` onto the
    figure as if it were a result.
    """
    from scipy import stats as sp_stats
    groups = list(data[group_col].unique())
    if len(groups) < 2:
        logger.info("No significance test for %d group; the boxplot is drawn without one.",
                    len(groups))
        return
    groups_data = [data[data[group_col] == g][value_col].dropna().to_numpy(dtype=float)
                   for g in groups]
    too_small = [len(vals) for vals in groups_data if len(vals) < 2]
    if too_small:
        logger.info("No significance test: %d of %d groups has fewer than 2 usable values.",
                    len(too_small), len(groups_data))
        return
    if len(groups_data) == 2:
        label, pval = 't-test', sp_stats.ttest_ind(*groups_data).pvalue
    else:
        label, pval = 'ANOVA', sp_stats.f_oneway(*groups_data).pvalue
    if not math.isfinite(pval):
        logger.info("%s did not produce a finite p-value; nothing annotated.", label)
        return
    ax.text(0.5, 1.02, f'{label} p = {pval:.4f}', ha='center',
            transform=ax.transAxes, fontsize=9)

def add_regression_eq(x: List[float], y: List[float], ax: Any) -> None:
    """Annotate the regression equation and R² onto a scatter Axes.

    A fit that cannot be computed — fewer than two points, identical x values, or
    a result that is not finite — is logged and leaves the figure without an
    annotation, rather than printing ``y = nanx + nan`` as a fitted line.
    """
    from scipy import stats as sp_stats
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        logger.info("Regression needs equal-length series, got %d and %d values.",
                    len(x), len(y))
        return
    usable = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[usable], y[usable]
    if len(x) < 2:
        logger.info("Insufficient data for a regression line.")
        return
    if np.unique(x).size < 2:
        logger.info("Regression needs at least two distinct x values; all %d are the same.",
                    len(x))
        return
    fit = sp_stats.linregress(x, y)
    values = (fit.slope, fit.intercept, fit.rvalue, fit.pvalue)
    if not all(math.isfinite(v) for v in values):
        logger.info("Regression did not produce finite coefficients; nothing annotated.")
        return
    eq = (f'y = {fit.slope:.3f}x + {fit.intercept:.3f}\n'
          f'R² = {fit.rvalue**2:.3f}, p = {fit.pvalue:.4f}')
    ax.text(0.05, 0.95, eq, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7))


# ── Shared Constants & Helpers ───────────────────────────────────────────────

GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

STOP_CODONS = {'TAA', 'TAG', 'TGA'}


def has_tool(name: str) -> bool:
    """Whether an external command-line tool can be run.

    Presence is decided by :func:`shutil.which`, which respects ``PATH`` (and
    ``PATHEXT`` on Windows) without starting anything.  The ``--version`` probe
    then only tells us the program actually started: a non-zero exit is still a
    program that exists (several bioinformatics tools print their usage line and
    exit ``1``/``2`` for ``--version``), while a tool that cannot be started at
    all — missing, or on ``PATH`` without the execute bit — is reported absent.
    Measured before: the probe's ``returncode == 0`` test called ``bwa`` with a
    usage-exiting build "not installed", and a ``PermissionError`` from the same
    call escaped and took down every module that asks this question.

    A probe that hangs for the whole timeout is reported as absent, so the caller
    falls back to the built-in engine instead of waiting on it again.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"has_tool needs a command name, got {name!r}.")
    if shutil.which(name) is None:
        return False
    try:
        subprocess.run([name, '--version'], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return False
    except OSError:
        # Not startable after all (deleted since the PATH lookup, no execute bit).
        return False
    return True


def smart_open(path: str, mode: str = "rt", **kwargs):
    """Open a possibly compressed file transparently — one point of truth for input.

    ``.gz`` and ``.bz2`` were detected by the format sniffers but never
    decompressed anywhere, so real NGS files failed to parse; text mode now also
    pins UTF-8 and consumes a leading BOM, which stops Windows' legacy code page
    from silently mangling non-ASCII headers.
    """
    import bz2
    import gzip

    if "b" not in mode:
        kwargs.setdefault("encoding", "utf-8-sig")
    lowered = str(path).lower()
    if lowered.endswith(".gz"):
        return gzip.open(path, mode, **kwargs)
    if lowered.endswith(".bz2"):
        return bz2.open(path, mode, **kwargs)
    return open(path, mode, **kwargs)


def read_fasta_simple(filepath: str) -> List[Tuple[str, str]]:
    """Read a FASTA file into (id, sequence) tuples without Biopython.

    The identifier is the first token of the header line, matching what
    ``Bio.SeqIO`` reports as ``record.id`` — callers that need the entire header
    use ``sequence.read_fasta`` instead.  A ``>`` line carrying no name is still a
    record.  A missing file raises rather than returning an empty list: a mistyped
    path must not look like a legitimately empty dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    sequences = []
    name, buf = None, []
    with smart_open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    sequences.append((name, ''.join(buf)))
                tokens = line[1:].split()
                name = tokens[0] if tokens else ''
                buf = []
            elif line:
                buf.append(line)
    if name is not None:
        sequences.append((name, ''.join(buf)))
    return sequences

# ── Shared Bioinformatics Utilities ────────────────────────────────────────────

def reverse_complement_dna(seq: str) -> str:
    """Compute the reverse complement of a DNA sequence.

    This is the canonical implementation shared across all modules.
    Each base is replaced by its complement (A<->T, C<->G, N->N) and the
    result is reversed.

    Args:
        seq: DNA sequence string (case preserved in output).

    Returns:
        Reverse complemented sequence string.

    This function used to carry its own translation table that handled only
    ACGTN, so it silently disagreed with sequence.reverse_complement: it left
    every IUPAC ambiguity code (R, Y, S, W, K, M, B, D, H, V) uncomplemented
    and happily reversed non-DNA junk, returning '321ZYX' for 'XYZ123'.
    It now delegates, so there is exactly one implementation.
    """
    from .sequence import reverse_complement
    return reverse_complement(seq)


# ── Restriction Enzyme Database ────────────────────────────────────────────────
# Comprehensive database of restriction enzymes with recognition sites
# and cut positions. Used by orf_finder.py, cloning.py, and other modules.
# Format: enzyme_name -> (recognition_site, cut_position_from_start_of_site)
#
# The cut falls after that many bases of the site on the top strand, so the
# overhang length is len(site) - 2*cut_position. Only the enzymes listed in
# ENZYME_CUT_TYPES have positions checked against REBASE; every other position is
# a historical placeholder and must not be presented as a measured value.

RESTRICTION_ENZYMES = {
    # ── A ──────────────────────────────────────────────────────────────────
    'AatII':     ('GACGTC', 1),
    'Acc65I':    ('GGTACC', 1),
    'AflII':     ('CTTAAG', 1),
    'AflIII':    ('ACRYGT', 1),
    'AgeI':      ('ACCGGT', 1),
    'AluI':      ('AGCT', 2),
    'ApaI':      ('GGGCCC', 1),
    'ApaLI':     ('GTGCAC', 1),
    'ApoI':      ('RAATTY', 1),
    'AscI':      ('GGCGCGCC', 2),
    'AvaI':      ('CYCGRG', 1),
    'AvaII':     ('GGWCC', 1),
    'AvrII':     ('CCTAGG', 1),
    # ── B ──────────────────────────────────────────────────────────────────
    'BaeI':      ('ACNNNNGTAYC', 1),
    'BamHI':     ('GGATCC', 1),
    'BanI':      ('GGYRCC', 1),
    'BanII':     ('GRGCYC', 1),
    'BbsI':      ('GAAGAC', 2),
    'BcgI':      ('CGANNNNNNTGC', 2),
    'BciVI':     ('GTATCC', 1),
    'BclI':      ('TGATCA', 1),
    'BfaI':      ('CTAG', 1),
    'BfuAI':     ('ACCTGC', 1),
    'BglI':      ('GCCNNNNNGGC', 2),
    'BglII':     ('AGATCT', 1),
    'BmtI':      ('GCTAGC', 1),
    'BsaI':      ('GGTCTC', 1),
    'BsaHI':     ('GRCGYC', 1),
    'BsaWI':     ('WCCGGW', 1),
    'BseRI':     ('GAGGAG', 1),
    'BsgI':      ('GTGCAG', 1),
    'BsiWI':     ('CGTACG', 1),
    'BslI':      ('CCNNNNNNNGG', 2),
    'BsmAI':     ('GTCTC', 1),
    'BsmFI':     ('GGGAC', 1),
    'BsoBI':     ('CYCGRG', 1),
    'Bsp1286I':  ('GDGCHC', 1),
    'BspEI':     ('TCCGGA', 1),
    'BspHI':     ('TCATGA', 1),
    'BspMI':     ('ACCTGC', 1),
    'BsrFI':     ('RCCGGY', 1),
    'BsrGI':     ('TGTACA', 1),
    'BstBI':     ('TTCGAA', 1),
    'BstEI':     ('GGTNACC', 1),
    'BstNI':     ('CCWGG', 1),
    'BstUI':     ('CGCG', 2),
    'BstXI':     ('CCANNNNNNTGG', 2),
    'BstZ17I':   ('GTATAC', 1),
    'BsuRI':     ('GGCC', 2),
    'BtgI':      ('CCRYGG', 1),
    'BtgZI':     ('GCGATG', 1),
    'BtsI':      ('GCAGTG', 1),
    'BveI':      ('ACCTGC', 1),
    # ── C ──────────────────────────────────────────────────────────────────
    'Cac8I':     ('GCNNGC', 1),
    'ClaI':      ('ATCGAT', 1),
    'CviAII':    ('CATG', 1),
    'CviJI':     ('RGCY', 1),
    # ── D ──────────────────────────────────────────────────────────────────
    'DdeI':      ('CTNAG', 1),
    'DpnI':      ('GATC', 1),
    'DraI':      ('TTTAAA', 1),
    'DraIII':    ('CACNNNGTG', 1),
    'DrdI':      ('GACNNNNNNGTC', 1),
    # ── E ──────────────────────────────────────────────────────────────────
    'EaeI':      ('YGGCCR', 1),
    'EagI':      ('CGGCCG', 1),
    'EarI':      ('CTCTTC', 1),
    'EciI':      ('GGCGGA', 1),
    'EcoNI':     ('CCTNNNNNAGG', 1),
    'EcoO109I':  ('RGGNCCY', 1),
    'EcoRI':     ('GAATTC', 1),
    'EcoRV':     ('GATATC', 3),
    'EcoT22I':   ('ATGCAT', 1),
    # ── F ──────────────────────────────────────────────────────────────────
    'FatI':      ('CATG', 1),
    'FauI':      ('CCCGC', 2),
    'Fnu4HI':    ('GCNGC', 1),
    'FokI':      ('GGATG', 1),
    'FseI':      ('GGCCGGCC', 2),
    'FspI':      ('TGCGCA', 1),
    # ── H ──────────────────────────────────────────────────────────────────
    'HaeII':     ('RGCGCY', 1),
    'HaeIII':    ('GGCC', 2),
    'HgaI':      ('GACGC', 1),
    'HhaI':      ('GCGC', 1),
    'HindIII':   ('AAGCTT', 1),
    'HinfI':     ('GANTC', 1),
    'HpaI':      ('GTTAAC', 1),
    'HpaII':     ('CCGG', 1),
    'Hpy188I':   ('TCNGA', 1),
    'Hpy188III': ('TCNNGA', 1),
    'HpyAV':     ('CCTTC', 1),
    'HpyCH4III': ('ACNGT', 1),
    'HpyCH4IV':  ('ACGT', 1),
    # ── K ──────────────────────────────────────────────────────────────────
    'KasI':      ('GGCGCC', 1),
    'KpnI':      ('GGTACC', 5),
    # ── M ──────────────────────────────────────────────────────────────────
    'MboI':      ('GATC', 1),
    'MboII':     ('GAAGA', 1),
    'MfeI':      ('CAATTG', 1),
    'MluI':      ('ACGCGT', 1),
    'MluCI':     ('AATT', 1),
    'MmeI':      ('TCCRAC', 1),
    'MnlI':      ('CCTC', 1),
    'MseI':      ('TTAA', 1),
    'MslI':      ('CAYNNNNRTG', 1),
    'MspA1I':    ('CMGCKG', 1),
    'MspI':      ('CCGG', 1),
    'MwoI':      ('GCNNNNNNNGC', 1),
    # ── N ──────────────────────────────────────────────────────────────────
    'NaeI':      ('GCCGGC', 1),
    'NarI':      ('GGCGCC', 1),
    'NcoI':      ('CCATGG', 1),
    'NdeI':      ('CATATG', 1),
    'NgoMIV':    ('GCCGGC', 1),
    'NheI':      ('GCTAGC', 1),
    'NlaIII':    ('CATG', 1),
    'NlaIV':     ('GGNNCC', 1),
    'NotI':      ('GCGGCCGC', 2),
    'NruI':      ('TCGCGA', 1),
    'NsiI':      ('ATGCAT', 1),
    'NspI':      ('RCATGY', 1),
    'NspBIII':   ('CMGCKG', 1),
    # ── P ──────────────────────────────────────────────────────────────────
    'PacI':      ('TTAATTAA', 2),
    'PaeR7I':    ('CTCGAG', 1),
    'PciI':      ('ACATGT', 1),
    'PmeI':      ('GTTTAAAC', 2),
    'PmlI':      ('CACGTG', 1),
    'PpuMI':     ('RGGWCCY', 1),
    'PstI':      ('CTGCAG', 5),
    'PvuI':      ('CGATCG', 1),
    'PvuII':     ('CAGCTG', 1),
    # ── R ──────────────────────────────────────────────────────────────────
    'RsaI':      ('GTAC', 2),
    'RsrII':     ('CGGWCCG', 1),
    # ── S ──────────────────────────────────────────────────────────────────
    'SacI':      ('GAGCTC', 5),
    'SacII':     ('CCGCGG', 1),
    'SalI':      ('GTCGAC', 1),
    'SapI':      ('GCTCTTC', 1),
    'Sau3AI':    ('GATC', 1),
    'Sau96I':    ('GGNCC', 1),
    'SbfI':      ('CCTGCAGG', 2),
    'ScaI':      ('AGTACT', 1),
    'ScrFI':     ('CCNGG', 1),
    'SexAI':     ('ACCWGGT', 1),
    'SfaNI':     ('GCATC', 1),
    'SfiI':      ('GGCCNNNNNGGCC', 2),
    'SfoI':      ('GGCGCC', 1),
    'SgrAI':     ('CRCCGGYG', 1),
    'SmaI':      ('CCCGGG', 3),
    'SmlI':      ('CTYRAG', 1),
    'SnaBI':     ('TACGTA', 1),
    'SpeI':      ('ACTAGT', 1),
    'SphI':      ('GCATGC', 5),
    'SplI':      ('CGTACG', 1),
    'SrfI':      ('GCCCGGGC', 1),
    'Sse8387I':  ('CCTGCAGG', 1),
    'SspI':      ('AATATT', 1),
    'StuI':      ('AGGCCT', 1),
    'StyI':      ('CCWWGG', 1),
    'StyD4I':    ('CCNGG', 1),
    # ── T ──────────────────────────────────────────────────────────────────
    'TaiI':      ('ACGT', 1),
    'TaqI':      ('TCGA', 1),
    'TfiI':      ('GAWTC', 1),
    'TseI':      ('GCWGC', 1),
    'Tsp45I':    ('GTSAC', 1),
    'Tsp509I':   ('AATT', 1),
    'TsrI':      ('GGNCC', 1),
    'Tth111I':   ('GACNNNGTC', 1),
    # ── X ──────────────────────────────────────────────────────────────────
    'XbaI':      ('TCTAGA', 1),
    'XcmI':      ('CCANNNNNNNNNTGG', 2),
    'XhoI':      ('CTCGAG', 1),
    'XmaI':      ('CCCGGG', 1),
    'XmaCI':     ('CCGG', 1),
    'XmnI':      ('GAANNNNTTC', 1),
    # ── Z ──────────────────────────────────────────────────────────────────
    'ZraI':      ('GACGTC', 1),
}

# Convenience: enzyme sites only (no cut positions) for modules that only
# need recognition sequences (e.g. cloning.py digestion simulation).
RESTRICTION_ENZYMES_SITES = {name: site for name, (site, _) in RESTRICTION_ENZYMES.items()}

# Overhang geometry verified against REBASE, for the enzymes whose cut positions
# above are also checked.  Anything absent from this dict reports a geometry that
# was derived from an unchecked placeholder, and says so.
ENZYME_CUT_TYPES = {
    'EcoRI':   '5prime',
    'BamHI':   '5prime',
    'HindIII': '5prime',
    'SmaI':    'blunt',
    'EcoRV':   'blunt',
    'PstI':    '3prime',
    'KpnI':    '3prime',
    'SacI':    '3prime',
    'SphI':    '3prime',
}


def enzyme_cut_type(name: str) -> str:
    """Return 'blunt' / '5prime' / '3prime' for *name*, or 'unverified'.

    'unverified' means the stored cut position for this enzyme has never been
    checked against REBASE, so any geometry derived from it is a guess.
    """
    return ENZYME_CUT_TYPES.get(name, "unverified")


def secure_temp_path(suffix: str = "", prefix: str = "biosuite_") -> str:
    """Reserve a unique temporary path safely and return it.

    tempfile.mktemp() only *predicts* a free name; between the prediction and
    the open, anything on a shared /tmp can create that path - typically a
    symlink to a file the process may write. mkstemp() creates the file
    atomically with mode 0600 and hands back a descriptor, closing the race.

    The path is returned already existing and empty, so callers must not treat
    mere existence as proof a tool wrote output - use wrote_output().
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    return path


def wrote_output(path: str) -> bool:
    """True if `path` is a file that holds at least one byte.

    Success checks written as os.path.exists(path) became tautologies once
    secure_temp_path started reserving the path by creating it. This is also
    strictly stronger than the original check: an external tool that exits 0
    without writing anything is now caught rather than passed to a parser, and a
    directory left at that path counts as nothing written (`getsize()` of a
    directory is platform-defined — 0 here — so `exists()` plus a size comparison
    had no rule behind it).

    A *path* that is not a string is a caller bug and raises, rather than being
    answered with a false that blames the external tool for it.
    """
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        # A reserved name is not a claim on the filesystem: another process can
        # delete the file between the two stats, and a temp sweep does exactly
        # that. "No output" is the honest answer in that race.
        return False


# ── Installation inventory ───────────────────────────────────────────────────

def count_analysis_modules() -> int:
    """Number of analysis modules this installation ships: ``biosuite.core`` plus
    its ``workflow`` subpackage, counted from the package itself.

    Every user-visible module count (``/health``, the CLI report, the documentation)
    has to come from here.  The CLI report carried a literal that stopped matching
    the package, which is how a count turns into a lie.
    """
    from pathlib import Path

    import biosuite.core as core_pkg

    root = Path(core_pkg.__file__).parent
    modules = {p.stem for p in root.glob("*.py") if p.stem != "__init__"}
    modules |= {f"workflow.{p.stem}" for p in (root / "workflow").glob("*.py")
                if p.stem != "__init__"}
    return len(modules)


def count_test_files() -> int:
    """``test_*.py`` files under ``./tests``, or 0 outside a source checkout.

    Resolved from the working directory because that is where the CLI is run from,
    and an installed wheel legitimately has no tests to count.
    """
    from pathlib import Path

    root = Path.cwd() / "tests"
    return len(sorted(root.rglob("test_*.py"))) if root.is_dir() else 0
