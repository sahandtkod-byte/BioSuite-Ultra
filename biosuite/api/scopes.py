"""Opt-in scoped API keys for the BioSuite Ultra API.

Scoped keys are enabled *only* when ``BIOSUITE_SCOPED_KEYS`` is set to a
non-empty configuration string of the form::

    BIOSUITE_SCOPED_KEYS="key1:read;key2:execute,read;key3:admin"

Each ``;``-separated entry binds one key to a comma-separated list of scopes
from :data:`SCOPE_LEVELS`.  Scopes inherit downward — ``admin`` grants
``execute`` and ``read``, ``execute`` grants ``read`` — so a stored entry always
holds the *effective* scope set.

When the variable is absent or empty no scoped keys exist and the legacy
single-key full-access behaviour is preserved exactly.  A malformed
configuration raises :class:`ScopesConfigurationError`, which :mod:`biosuite.api.auth`
lets propagate at import time: the process refuses to start instead of serving
traffic with a half-understood key table (fail closed).

Error messages describe the offending entry by *position*, never by value, so
no credential can leak into a log, an exception or an HTTP response.
"""
from __future__ import annotations

import hmac
import os
from typing import Dict, FrozenSet, Iterable, Optional

from ..core.log import get_logger

logger = get_logger(__name__)

__all__ = ["ENV_VAR", "SCOPE_LEVELS", "ScopesConfigurationError",
           "key_scope", "load_scoped_keys", "parse_scoped_keys",
           "scope_satisfies"]

#: Environment variable holding the opt-in scoped-key configuration.
ENV_VAR = "BIOSUITE_SCOPED_KEYS"

#: Declared scopes, ordered from weakest to strongest.
SCOPE_LEVELS = {"read": 0, "execute": 1, "admin": 2}

_ENTRY_SEPARATOR = ";"
_SCOPE_SEPARATOR = ":"
_SCOPE_LIST_SEPARATOR = ","


class ScopesConfigurationError(ValueError):
    """The scoped-key configuration is malformed; the process must not serve."""


def expand_scopes(names: Iterable[str]) -> FrozenSet[str]:
    """Return the effective scope set for *names*, including inherited scopes."""
    granted = set(names)
    if "execute" in granted:
        granted.add("read")
    if "admin" in granted:
        granted |= {"execute", "read"}
    return frozenset(granted)


def parse_scoped_keys(raw: Optional[str]) -> Dict[str, FrozenSet[str]]:
    """Parse ``BIOSUITE_SCOPED_KEYS`` into ``{key: effective scopes}``.

    Raises :class:`ScopesConfigurationError` on any malformed entry.  Absent or
    blank input means "no scoped keys configured" and returns an empty mapping.
    """
    if raw is None or not raw.strip():
        return {}
    parsed: Dict[str, FrozenSet[str]] = {}
    for position, entry in enumerate(raw.split(_ENTRY_SEPARATOR), start=1):
        if not entry.strip():
            raise ScopesConfigurationError(
                f"{ENV_VAR} entry {position} is empty")
        if _SCOPE_SEPARATOR not in entry:
            raise ScopesConfigurationError(
                f"{ENV_VAR} entry {position} is missing the "
                f"'{_SCOPE_SEPARATOR}' scope separator")
        key, _, scope_part = entry.partition(_SCOPE_SEPARATOR)
        key = key.strip()
        if not key:
            raise ScopesConfigurationError(
                f"{ENV_VAR} entry {position} has an empty key")
        names = [name.strip().lower()
                 for name in scope_part.split(_SCOPE_LIST_SEPARATOR)]
        if any(not name for name in names):
            raise ScopesConfigurationError(
                f"{ENV_VAR} entry {position} has an empty scope")
        unknown = sorted(set(names) - set(SCOPE_LEVELS))
        if unknown:
            raise ScopesConfigurationError(
                f"{ENV_VAR} entry {position} has unknown scope(s): "
                + ", ".join(unknown))
        for existing in parsed:
            if hmac.compare_digest(key, existing):
                raise ScopesConfigurationError(
                    f"{ENV_VAR} entry {position} duplicates an earlier key")
        parsed[key] = expand_scopes(names)
    return parsed


def load_scoped_keys() -> Dict[str, FrozenSet[str]]:
    """Read the environment configuration; malformed input fails closed."""
    try:
        return parse_scoped_keys(os.environ.get(ENV_VAR))
    except ScopesConfigurationError:
        logger.error("%s is malformed — refusing to start (fail closed). "
                     "Fix or unset the variable and try again.", ENV_VAR)
        raise


def key_scope(key: Optional[str],
              scoped_keys: Dict[str, FrozenSet[str]]) -> Optional[str]:
    """Return the strongest granted scope for *key*, or ``None``.

    Every configured key is compared with :func:`hmac.compare_digest`, so the
    lookup is constant-time in each comparison.
    """
    if not key:
        return None
    candidate = str(key)
    strongest: Optional[str] = None
    for configured, scopes in scoped_keys.items():
        if hmac.compare_digest(candidate, configured):
            level = max(SCOPE_LEVELS[name] for name in scopes)
            name = next(n for n, l in SCOPE_LEVELS.items() if l == level)
            if strongest is None or SCOPE_LEVELS[name] > SCOPE_LEVELS[strongest]:
                strongest = name
    return strongest


def scope_satisfies(granted: Optional[str], required: str) -> bool:
    """True when *granted* is at least as strong as the *required* scope."""
    if granted is None or required not in SCOPE_LEVELS:
        return False
    if granted not in SCOPE_LEVELS:
        return False
    return SCOPE_LEVELS[granted] >= SCOPE_LEVELS[required]
