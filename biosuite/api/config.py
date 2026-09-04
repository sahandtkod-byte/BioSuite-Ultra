"""Runtime security configuration for the BioSuite Ultra API.

Security policy
---------------
The API must never fall back to a *known* credential.  Two modes exist:

``BIOSUITE_DEV_MODE=1``
    Development mode.  Well-known placeholder credentials are used when the
    corresponding environment variable is absent, and a loud warning is
    emitted.  Intended for local experimentation only.

default (production)
    Any credential that is not supplied through the environment is replaced by
    a **random per-process value**, which makes offline token forgery and
    default-password login impossible.  Admin login is disabled entirely unless
    ``BIOSUITE_ADMIN_PASSWORD`` (or ``BIOSUITE_ADMIN_PASSWORD_HASH``) is set.
    :func:`validate_runtime_config` reports the missing variables so a server
    entry point can refuse to start instead of serving traffic insecurely.
"""
from __future__ import annotations

import binascii
import hashlib
import hmac
import logging
import os
import secrets
from typing import List, Optional

logger = logging.getLogger(__name__)

# Well-known placeholders.  These are ONLY ever active in explicit dev mode and
# are listed here so that :func:`validate_runtime_config` can reject them if an
# operator copies them into a production environment.
DEV_API_KEY = "changeme-dev-key"
DEV_JWT_SECRET = "changeme-dev-secret"
DEV_ADMIN_PASSWORD = "changeme-dev-password"
_KNOWN_WEAK = frozenset({DEV_API_KEY, DEV_JWT_SECRET, DEV_ADMIN_PASSWORD, "", "changeme"})

_PBKDF2_ROUNDS = 240_000


def dev_mode() -> bool:
    """True when ``BIOSUITE_DEV_MODE`` is set to a truthy value."""
    return os.environ.get("BIOSUITE_DEV_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_secret(env_var: str, dev_default: str, purpose: str) -> tuple[str, bool]:
    """Resolve a secret from the environment.

    Returns ``(value, is_configured)``.  When the variable is unset the value is
    the dev placeholder in dev mode, otherwise a fresh random string that no
    attacker can predict.  ``is_configured`` is False in both fallback cases.
    """
    value = os.environ.get(env_var)
    if value:
        return value, True
    if dev_mode():
        logger.warning(
            "%s is not set — using the DEVELOPMENT placeholder because "
            "BIOSUITE_DEV_MODE is enabled. Never do this in production.", env_var)
        return dev_default, False
    logger.warning(
        "%s is not set — generated an ephemeral random %s for this process. "
        "Set %s to serve traffic (tokens will not survive a restart).",
        env_var, purpose, env_var)
    return secrets.token_urlsafe(48), False


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Return ``pbkdf2_sha256$rounds$salt$hash`` for *password*."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS)
    return "pbkdf2_sha256${}${}${}".format(
        _PBKDF2_ROUNDS, binascii.hexlify(salt).decode(), binascii.hexlify(digest).decode())


def verify_password(password: str, encoded: Optional[str]) -> bool:
    """Constant-time verification of *password* against :func:`hash_password` output."""
    if not encoded or not password:
        return False
    try:
        scheme, rounds, salt_hex, hash_hex = encoded.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"),
            binascii.unhexlify(salt_hex), int(rounds))
    except (ValueError, binascii.Error):
        return False
    return hmac.compare_digest(binascii.hexlify(digest).decode(), hash_hex)


def resolve_admin_password_hash() -> tuple[Optional[str], bool]:
    """Resolve the admin password hash.  ``(None, False)`` disables admin login."""
    encoded = os.environ.get("BIOSUITE_ADMIN_PASSWORD_HASH")
    if encoded:
        return encoded, True
    plain = os.environ.get("BIOSUITE_ADMIN_PASSWORD")
    if plain:
        return hash_password(plain), True
    if dev_mode():
        logger.warning("BIOSUITE_ADMIN_PASSWORD is not set — using the DEVELOPMENT "
                       "placeholder password because BIOSUITE_DEV_MODE is enabled.")
        return hash_password(DEV_ADMIN_PASSWORD), False
    logger.warning("BIOSUITE_ADMIN_PASSWORD is not set — admin login is DISABLED.")
    return None, False


def validate_runtime_config() -> List[str]:
    """Return a list of human-readable production configuration problems.

    An empty list means the process is safe to expose.  Callers that bind a
    socket (see ``biosuite.api.server``) must refuse to start when it is not
    empty and dev mode is off.
    """
    problems: List[str] = []
    checks = (
        ("BIOSUITE_API_KEY", "API key"),
        ("BIOSUITE_JWT_SECRET", "JWT signing secret"),
    )
    for env_var, label in checks:
        value = os.environ.get(env_var)
        if not value:
            problems.append(f"{env_var} is not set ({label} required to serve traffic).")
        elif value in _KNOWN_WEAK:
            problems.append(f"{env_var} is set to a well-known placeholder value.")
        elif len(value) < 16:
            problems.append(f"{env_var} is shorter than 16 characters.")
    admin_pw = os.environ.get("BIOSUITE_ADMIN_PASSWORD")
    if admin_pw and admin_pw in _KNOWN_WEAK:
        problems.append("BIOSUITE_ADMIN_PASSWORD is set to a well-known placeholder value.")
    if admin_pw and len(admin_pw) < 12:
        problems.append("BIOSUITE_ADMIN_PASSWORD is shorter than 12 characters.")
    if not admin_pw and not os.environ.get("BIOSUITE_ADMIN_PASSWORD_HASH"):
        problems.append(
            "Neither BIOSUITE_ADMIN_PASSWORD nor BIOSUITE_ADMIN_PASSWORD_HASH is set "
            "(admin routes will be unavailable).")
    return problems


def format_config_error(problems: List[str]) -> str:
    """Render :func:`validate_runtime_config` output as an actionable message."""
    bullets = "\n".join(f"  - {p}" for p in problems)
    return (
        "Refusing to start: insecure production configuration.\n"
        f"{bullets}\n\n"
        "Fix by exporting strong values, for example:\n"
        "  export BIOSUITE_API_KEY=\"$(python -c 'import secrets;print(secrets.token_urlsafe(32))')\"\n"
        "  export BIOSUITE_JWT_SECRET=\"$(python -c 'import secrets;print(secrets.token_urlsafe(48))')\"\n"
        "  export BIOSUITE_ADMIN_PASSWORD='<a strong passphrase>'\n"
        "Or set BIOSUITE_DEV_MODE=1 for local development only."
    )
