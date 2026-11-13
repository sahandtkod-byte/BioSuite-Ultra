"""Credential resolution and the production configuration gate.

``biosuite.api.config`` is the module that decides whether the API is allowed to serve
traffic: a missing secret must become a random one, never a published placeholder, and
``validate_runtime_config`` has to name every problem it finds.  Each rule is tested
here as a rule — with the environment driven directly — because a test suite that only
exercises the happy path cannot tell a safe fallback from an insecure one.
"""
import logging

import pytest

from biosuite.api import config


@pytest.fixture
def clean_env(monkeypatch):
    """An environment with none of the API's credentials set."""
    for variable in ("BIOSUITE_API_KEY", "BIOSUITE_JWT_SECRET", "BIOSUITE_DEV_MODE",
                     "BIOSUITE_ADMIN_PASSWORD", "BIOSUITE_ADMIN_PASSWORD_HASH"):
        monkeypatch.delenv(variable, raising=False)
    return monkeypatch


# ── dev mode ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " on "])
def test_dev_mode_accepts_the_documented_spellings(clean_env, value):
    clean_env.setenv("BIOSUITE_DEV_MODE", value)
    assert config.dev_mode() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_dev_mode_is_off_by_default_and_for_falsy_values(clean_env, value):
    clean_env.setenv("BIOSUITE_DEV_MODE", value)
    assert config.dev_mode() is False


def test_dev_mode_defaults_to_off_when_unset(clean_env):
    assert config.dev_mode() is False


# ── resolve_secret ───────────────────────────────────────────────────────────

def test_a_supplied_secret_is_used_verbatim_and_reported_as_configured(clean_env):
    clean_env.setenv("BIOSUITE_API_KEY", "a-real-secret-value")

    value, configured = config.resolve_secret("BIOSUITE_API_KEY", config.DEV_API_KEY, "API key")

    assert (value, configured) == ("a-real-secret-value", True)


def test_dev_mode_falls_back_to_the_placeholder_with_a_warning(clean_env, caplog):
    clean_env.setenv("BIOSUITE_DEV_MODE", "1")

    with caplog.at_level(logging.WARNING, logger="biosuite.api.config"):
        value, configured = config.resolve_secret(
            "BIOSUITE_JWT_SECRET", config.DEV_JWT_SECRET, "JWT signing secret")

    assert not configured
    assert value == config.DEV_JWT_SECRET
    assert "BIOSUITE_DEV_MODE" in caplog.text


def test_production_generates_an_unpredictable_secret_instead_of_a_default(clean_env):
    first, configured = config.resolve_secret("BIOSUITE_JWT_SECRET", config.DEV_JWT_SECRET,
                                              "JWT signing secret")
    second, _ = config.resolve_secret("BIOSUITE_JWT_SECRET", config.DEV_JWT_SECRET,
                                      "JWT signing secret")

    assert configured is False
    assert first != second != config.DEV_JWT_SECRET
    assert len(first) >= 48


def test_the_generated_secret_is_announced_as_ephemeral(clean_env, caplog):
    with caplog.at_level(logging.WARNING, logger="biosuite.api.config"):
        config.resolve_secret("BIOSUITE_API_KEY", config.DEV_API_KEY, "API key")

    assert "ephemeral" in caplog.text
    assert "BIOSUITE_API_KEY" in caplog.text


# ── hash_password / verify_password ──────────────────────────────────────────

def test_a_hash_carries_its_scheme_rounds_and_salt():
    encoded = config.hash_password("correct horse", salt=b"0123456789abcdef")

    scheme, rounds, salt_hex, digest_hex = encoded.split("$")
    assert scheme == "pbkdf2_sha256"
    assert int(rounds) == config._PBKDF2_ROUNDS
    assert salt_hex == b"0123456789abcdef".hex()
    assert len(digest_hex) == 64                       # sha256, hex-encoded


def test_the_same_password_and_salt_hash_identically(clean_env):
    """A salt supplied twice must reproduce the digest, or verification can never work."""
    salt = b"\x01" * 16
    assert config.hash_password("pw", salt=salt) == config.hash_password("pw", salt=salt)


def test_different_salts_hash_the_same_password_differently():
    assert config.hash_password("pw") != config.hash_password("pw")


@pytest.mark.parametrize(("password", "encoded", "expected"), [
    ("right", None, False),                                        # no hash configured
    ("", "pbkdf2_sha256$1$00$00", False),                          # empty password
    ("wrong", "pbkdf2_sha256$1$00$00", False),                     # mismatch
    ("x", "bcrypt$1$00$00", False),                                # unknown scheme
    ("x", "no-dollar-signs", False),                               # cannot unpack
    ("x", "pbkdf2_sha256$notanint$00$00", False),                  # bad rounds
    ("x", "pbkdf2_sha256$1$zzzz$00", False),                       # non-hex salt
])
def test_verify_password_rejects_everything_that_is_not_a_valid_hash(password, encoded,
                                                                     expected):
    assert config.verify_password(password, encoded) is expected


def test_a_password_verifies_against_its_own_hash():
    encoded = config.hash_password("s3cr3t-passwd")

    assert config.verify_password("s3cr3t-passwd", encoded) is True
    assert config.verify_password("s3cr3t-passwD", encoded) is False


# ── resolve_admin_password_hash ──────────────────────────────────────────────

def test_an_explicit_hash_takes_precedence_over_a_plain_password(clean_env):
    encoded = config.hash_password("from-hash")
    clean_env.setenv("BIOSUITE_ADMIN_PASSWORD_HASH", encoded)
    clean_env.setenv("BIOSUITE_ADMIN_PASSWORD", "ignored")

    resolved, configured = config.resolve_admin_password_hash()

    assert resolved == encoded
    assert configured is True


def test_a_plain_password_is_hashed_on_the_way_through(clean_env):
    clean_env.setenv("BIOSUITE_ADMIN_PASSWORD", "long-enough-password")

    resolved, configured = config.resolve_admin_password_hash()

    assert configured is True
    assert config.verify_password("long-enough-password", resolved) is True


def test_dev_mode_enables_login_with_the_placeholder_and_says_so(clean_env, caplog):
    clean_env.setenv("BIOSUITE_DEV_MODE", "1")

    with caplog.at_level(logging.WARNING, logger="biosuite.api.config"):
        resolved, configured = config.resolve_admin_password_hash()

    assert resolved is not None
    assert configured is False
    assert config.verify_password(config.DEV_ADMIN_PASSWORD, resolved) is True
    assert "DEVELOPMENT placeholder" in caplog.text


def test_without_any_admin_password_login_is_disabled(clean_env, caplog):
    with caplog.at_level(logging.WARNING, logger="biosuite.api.config"):
        resolved, configured = config.resolve_admin_password_hash()

    assert (resolved, configured) == (None, False)
    assert "DISABLED" in caplog.text


# ── validate_runtime_config ──────────────────────────────────────────────────

GOOD = {"BIOSUITE_API_KEY": "0123456789abcdef0123456789abcdef",
        "BIOSUITE_JWT_SECRET": "fedcba9876543210fedcba9876543210",
        "BIOSUITE_ADMIN_PASSWORD": "a-long-admin-password"}


def test_a_complete_configuration_reports_no_problems(clean_env):
    for variable, value in GOOD.items():
        clean_env.setenv(variable, value)

    assert config.validate_runtime_config() == []


def test_missing_secrets_are_named_individually(clean_env):
    problems = config.validate_runtime_config()

    assert "BIOSUITE_API_KEY is not set" in problems[0]
    assert "BIOSUITE_JWT_SECRET is not set" in problems[1]
    assert any("admin" in p.lower() for p in problems)


@pytest.mark.parametrize("variable", ["BIOSUITE_API_KEY", "BIOSUITE_JWT_SECRET"])
def test_a_well_known_placeholder_is_refused(clean_env, variable):
    for key, value in GOOD.items():
        clean_env.setenv(key, value)
    clean_env.setenv(variable, "changeme")

    problems = config.validate_runtime_config()

    assert any(variable in p and "well-known placeholder" in p for p in problems)


@pytest.mark.parametrize("variable", ["BIOSUITE_API_KEY", "BIOSUITE_JWT_SECRET"])
def test_a_short_secret_is_refused(clean_env, variable):
    for key, value in GOOD.items():
        clean_env.setenv(key, value)
    clean_env.setenv(variable, "short")

    problems = config.validate_runtime_config()

    assert any(variable in p and "shorter than 16 characters" in p for p in problems)


def test_a_well_known_admin_password_is_refused(clean_env):
    for key, value in GOOD.items():
        clean_env.setenv(key, value)
    clean_env.setenv("BIOSUITE_ADMIN_PASSWORD", config.DEV_ADMIN_PASSWORD)

    problems = config.validate_runtime_config()

    assert any("BIOSUITE_ADMIN_PASSWORD" in p and "well-known placeholder" in p
               for p in problems)


def test_a_short_admin_password_is_refused(clean_env):
    for key, value in GOOD.items():
        clean_env.setenv(key, value)
    clean_env.setenv("BIOSUITE_ADMIN_PASSWORD", "shorty")

    problems = config.validate_runtime_config()

    assert any("BIOSUITE_ADMIN_PASSWORD" in p and "shorter than 12 characters" in p
               for p in problems)


def test_a_password_hash_counts_as_an_admin_password(clean_env):
    """An operator who stores only a hash must not be told to set a password."""
    for key, value in GOOD.items():
        if key != "BIOSUITE_ADMIN_PASSWORD":
            clean_env.setenv(key, value)
    clean_env.setenv("BIOSUITE_ADMIN_PASSWORD_HASH", config.hash_password("some-password"))

    assert config.validate_runtime_config() == []


def test_the_refusal_message_lists_every_problem_and_a_way_to_fix_them(clean_env):
    """Whoever starts the server should not have to read the source to continue."""
    problems = config.validate_runtime_config()      # nothing is set in this env
    assert problems, "an empty environment must look unsafe to the validator"

    message = config.format_config_error(problems)

    assert "Refusing to start" in message
    for problem in problems:
        assert f"  - {problem}" in message
    assert "export BIOSUITE_API_KEY=" in message
    assert "BIOSUITE_DEV_MODE=1" in message
