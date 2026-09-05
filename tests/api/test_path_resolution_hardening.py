"""Regression tests for the hardened data-directory path resolver.

These cover behaviour that the previous "resolve then compare" implementation
got wrong, and that CodeQL flagged as four High severity path-injection flows
(``biosuite/api/__init__.py`` lines 116/120/577/976):

* a dotfile such as ``.env`` sitting in the data directory was inside the
  confinement boundary, so the endpoint returned 200 for it and disclosed the
  absolute server path in its error message (measured: ``HTTP 200`` with
  ``Error reading unknown file: ... /tmp/<dir>/.env``). Nothing but the file
  reader's own format check stopped the contents being returned;
* a NUL byte was not rejected by the resolver at all; it fell through to the
  file reader, which surfaced it as a 422 ("embedded null byte") rather than
  the resolver refusing the path up front;
* double-encoded traversal (``..%252f``) survived the single decode that
  Starlette performs on query parameters;
* an absolute path was accepted as long as it happened to land inside the
  root, which is not what the docstring promised.

The resolver now validates the raw string against an allowlist *before* any
filesystem call, and then resolves each component against the real directory
listing so untrusted text is never concatenated into a path.
"""
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("BIOSUITE_API_KEY", "test-key-not-a-real-secret")
os.environ.setdefault("BIOSUITE_JWT_SECRET", "test-jwt-secret-not-a-real-secret")
os.environ.setdefault("BIOSUITE_ADMIN_PASSWORD", "test-admin-password")

from biosuite.api import HTTPException, resolve_user_path  # noqa: E402
from biosuite.api import app  # noqa: E402

HEADERS = {"X-API-Key": os.environ["BIOSUITE_API_KEY"]}


@pytest.fixture
def client():
    return TestClient(app)


# ── the resolver itself ─────────────────────────────────────────────────────

def _expect_status(user_path, status, tmp_path, monkeypatch):
    monkeypatch.setenv("BIOSUITE_DATA_DIR", str(tmp_path))
    with pytest.raises(HTTPException) as excinfo:
        resolve_user_path(user_path)
    assert excinfo.value.status_code == status, user_path


@pytest.mark.parametrize("payload", [
    ".env",                       # dotfile: previously readable
    ".ssh",
    "~",
    "~/.ssh/id_rsa",
    "..",
    "../etc/passwd",
    "/etc/passwd",
    "/proc/self/environ",
    "\\\\server\\share",          # UNC
    "C:\\Windows\\win.ini",       # drive-letter absolute
    "a/../../etc/passwd",
    "sub/../../../etc/shadow",
])
def test_traversal_and_dotfiles_are_rejected_with_400(payload, tmp_path,
                                                      monkeypatch):
    _expect_status(payload, 400, tmp_path, monkeypatch)


def test_dotfile_present_in_the_data_dir_is_still_refused(tmp_path, monkeypatch):
    """Measured old behaviour: HTTP 200, because .env is inside the root."""
    (tmp_path / ".env").write_text("BIOSUITE_JWT_SECRET=super-secret\n")
    _expect_status(".env", 400, tmp_path, monkeypatch)


def test_nul_byte_is_rejected_by_the_resolver(tmp_path, monkeypatch):
    """Measured old behaviour: no resolver check, 422 from the file reader."""
    _expect_status("sample\x00.fasta", 400, tmp_path, monkeypatch)


def test_double_encoded_traversal_is_rejected(tmp_path, monkeypatch):
    """Starlette decodes once; the resolver must unwrap the rest itself."""
    _expect_status("..%252f..%252fetc%252fpasswd", 400, tmp_path, monkeypatch)
    _expect_status("%2e%2e/%2e%2e/etc/passwd", 400, tmp_path, monkeypatch)


def test_absolute_path_inside_the_root_is_also_refused(tmp_path, monkeypatch):
    """Documented contract is 'relative, inside the root' - enforce it."""
    (tmp_path / "sample.fasta").write_text(">s\nACGT\n")
    _expect_status(str(tmp_path / "sample.fasta"), 400, tmp_path, monkeypatch)


def test_excessive_depth_is_rejected(tmp_path, monkeypatch):
    _expect_status("/".join(["a"] * 64), 400, tmp_path, monkeypatch)


def test_missing_file_is_404_not_500(tmp_path, monkeypatch):
    _expect_status("nope.fasta", 404, tmp_path, monkeypatch)


def test_a_file_that_exists_resolves_to_itself(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOSUITE_DATA_DIR", str(tmp_path))
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    target = nested / "sample.fasta"
    target.write_text(">s\nACGT\n")
    resolved = resolve_user_path("sub/deep/sample.fasta")
    assert resolved == target.resolve()


def test_symlink_escaping_the_root_is_refused(tmp_path, monkeypatch):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    try:
        (data_dir / "link.txt").symlink_to(outside)
    except OSError:                       # pragma: no cover - platform dependent
        pytest.skip("symlinks unavailable")
    _expect_status("link.txt", 400, data_dir, monkeypatch)


def test_symlink_staying_inside_the_root_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOSUITE_DATA_DIR", str(tmp_path))
    real = tmp_path / "real.fasta"
    real.write_text(">s\nACGT\n")
    try:
        (tmp_path / "alias.fasta").symlink_to(real)
    except OSError:                       # pragma: no cover - platform dependent
        pytest.skip("symlinks unavailable")
    assert resolve_user_path("alias.fasta") == real.resolve()


def test_untrusted_text_is_never_joined_into_a_path():
    """Structural guard for the CodeQL finding.

    The resolver must not build a path out of the caller's string. Anything of
    the form ``root / user_path`` or ``os.path.join(root, user_path)``
    reintroduces the taint flow that CodeQL reported.
    """
    import ast
    import inspect

    import biosuite.api as api_mod

    source = inspect.getsource(api_mod.resolve_user_path)
    tree = ast.parse(source)
    joined = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            joined.append(ast.unparse(node))
        if isinstance(node, ast.Call):
            func = ast.unparse(node.func)
            if func.endswith("join") and any(
                    "component" in ast.unparse(a) or "user_path" in ast.unparse(a)
                    for a in node.args):
                joined.append(ast.unparse(node))
    assert not joined, f"user data concatenated into a path: {joined}"
    assert "scandir" in source, "resolver no longer matches real directory entries"


# ── end-to-end through the endpoints CodeQL flagged ─────────────────────────

@pytest.mark.parametrize("endpoint", ["/api/v1/file/read",
                                      "/api/v1/file/detect-format"])
@pytest.mark.parametrize("payload", [".env", "sample\x00.fasta",
                                     "..%252f..%252fetc%252fpasswd",
                                     "/etc/passwd", "~/.ssh/id_rsa"])
def test_endpoints_never_leak_and_never_500(client, tmp_path, monkeypatch,
                                            endpoint, payload):
    monkeypatch.setenv("BIOSUITE_DATA_DIR", str(tmp_path))
    (tmp_path / ".env").write_text("BIOSUITE_JWT_SECRET=super-secret\n")
    response = client.post(endpoint, headers=HEADERS,
                           params={"file_path": payload})
    assert response.status_code in (400, 404), response.text
    assert response.status_code < 500
    assert "super-secret" not in response.text
    assert "root:" not in response.text
