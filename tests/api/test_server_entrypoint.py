"""The ``python -m biosuite.api.server`` command-line contract."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from biosuite.api import server


@pytest.fixture
def uvicorn_run(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    return calls


def test_cli_host_port_and_reload_are_honoured(uvicorn_run, monkeypatch):
    monkeypatch.setattr(server, "ensure_safe_to_serve", lambda: None)

    assert server.main(["--host", "0.0.0.0", "--port", "8465", "--no-reload"]) == 0

    (target, kwargs), = uvicorn_run
    assert target == ("biosuite.api:app",)
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8465
    assert kwargs["reload"] is False


def test_unknown_arguments_are_rejected_loudly(monkeypatch):
    monkeypatch.setattr(server, "ensure_safe_to_serve", lambda: None)

    with pytest.raises(SystemExit) as exc:
        server.main(["--does-not-exist"])

    assert exc.value.code == 2


def test_an_invalid_port_is_rejected(monkeypatch):
    monkeypatch.setattr(server, "ensure_safe_to_serve", lambda: None)

    with pytest.raises(SystemExit) as exc:
        server.main(["--port", "70000"])

    assert exc.value.code == 2


def test_a_missing_api_dependency_points_at_the_api_extra(monkeypatch):
    def missing(name, package=None):
        raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")

    monkeypatch.setattr(server.importlib, "import_module", missing)

    with pytest.raises(SystemExit) as exc:
        server._load_app()

    message = str(exc.value)
    assert "fastapi" in message
    assert "biosuite-ultra[api]" in message


def test_a_missing_server_dependency_points_at_the_api_extra(monkeypatch):
    def missing(name, package=None):
        raise ModuleNotFoundError("No module named 'uvicorn'", name="uvicorn")

    monkeypatch.setattr(server.importlib, "import_module", missing)

    with pytest.raises(SystemExit) as exc:
        server._load_uvicorn()

    message = str(exc.value)
    assert "uvicorn" in message
    assert "biosuite-ultra[api]" in message


def test_a_non_numeric_port_is_rejected(monkeypatch):
    monkeypatch.setattr(server, "ensure_safe_to_serve", lambda: None)

    with pytest.raises(SystemExit) as exc:
        server.main(["--port", "not-a-port"])

    assert exc.value.code == 2


def test_an_invalid_port_environment_is_rejected(monkeypatch, capsys):
    monkeypatch.setenv("BIOSUITE_API_PORT", "not-a-port")

    assert server.main([]) == 2
    assert "Invalid BIOSUITE_API_PORT" in capsys.readouterr().err


def test_the_module_entrypoint_prints_guidance_without_fastapi(tmp_path):
    """The real ``python -m`` path must not traceback before the guard runs."""
    (tmp_path / "fastapi.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'fastapi'\", name='fastapi')\n",
        encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(repo)])

    result = subprocess.run(
        [sys.executable, "-m", "biosuite.api.server"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "biosuite-ultra[api]" in result.stderr
    assert "Traceback" not in result.stderr
