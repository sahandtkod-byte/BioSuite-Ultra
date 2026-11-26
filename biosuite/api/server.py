"""
BioSuite Ultra API Server — Entry Point

Run with:
    python -m biosuite.api.server
    python -m biosuite.api.server --host 0.0.0.0 --port 8465
    # or
    uvicorn biosuite.api.server:app --host 127.0.0.1 --port 8000

``app`` is re-exported here so that the documented ASGI target resolves; it is
the very same application object as ``biosuite.api:app``.

Startup refuses to bind a socket when the process would serve traffic with
missing or well-known credentials.  Set ``BIOSUITE_DEV_MODE=1`` to override for
local development.

Open API docs at: http://localhost:8000/docs
"""
import argparse
import importlib
import os
import sys

from biosuite import __version__
from biosuite.core.log import get_logger

logger = get_logger(__name__)

__all__ = ["app", "main", "ensure_safe_to_serve"]


def _load_app():
    """Import the ASGI application with an actionable missing-extra message."""
    try:
        return importlib.import_module("biosuite.api").app
    except ModuleNotFoundError as exc:
        missing = exc.name or "an API dependency"
        raise SystemExit(
            "BioSuite API dependencies are not installed "
            f"(missing: {missing}).\n"
            "Install them with:\n"
            "  pip install \"biosuite-ultra[api]\"\n"
        ) from None


def _load_uvicorn():
    """Import uvicorn with the same missing-extra guidance as the ASGI app."""
    try:
        return importlib.import_module("uvicorn")
    except ModuleNotFoundError as exc:
        missing = exc.name or "uvicorn"
        raise SystemExit(
            "BioSuite API server dependencies are not installed "
            f"(missing: {missing}).\n"
            "Install them with:\n"
            "  pip install \"biosuite-ultra[api]\"\n"
        ) from None


app = _load_app()


def _api_config():
    # Imported lazily: importing ``biosuite.api.config`` executes the
    # ``biosuite.api`` package __init__, which imports fastapi.  Keeping this
    # out of the module header lets the package guard / ``_load_app`` turn a
    # missing API extra into an actionable message instead of a raw traceback.
    return importlib.import_module("biosuite.api.config")


def _validate_runtime_config():
    return _api_config().validate_runtime_config()


def _dev_mode():
    return _api_config().dev_mode()


def _format_config_error(problems):
    return _api_config().format_config_error(problems)


def ensure_safe_to_serve() -> None:
    """Abort startup when the runtime configuration is not production-safe."""
    problems = _validate_runtime_config()
    if not problems:
        return
    if _dev_mode():
        logger.warning("BIOSUITE_DEV_MODE is enabled — starting with an insecure "
                       "configuration:\n%s",
                       "\n".join(f"  - {p}" for p in problems))
        return
    raise SystemExit(_format_config_error(problems))


def _parse_port(value) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("port must be an integer") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _build_parser(default_host: str, default_port: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m biosuite.api.server",
        description="Serve the BioSuite Ultra REST API.",
    )
    parser.add_argument("--host", default=default_host,
                        help="bind address (default: BIOSUITE_API_HOST or 127.0.0.1)")
    parser.add_argument("--port", type=_parse_port, default=default_port,
                        help="listen port (default: BIOSUITE_API_PORT or 8000)")
    reload_group = parser.add_mutually_exclusive_group()
    reload_group.add_argument("--reload", dest="reload", action="store_true", default=None,
                              help="enable auto-reload (development only)")
    reload_group.add_argument("--no-reload", dest="reload", action="store_false",
                              help="disable auto-reload")
    return parser


def main(argv=None) -> int:
    default_host = os.environ.get("BIOSUITE_API_HOST", "127.0.0.1")
    try:
        default_port = _parse_port(os.environ.get("BIOSUITE_API_PORT", "8000"))
    except argparse.ArgumentTypeError as exc:
        print(f"Invalid BIOSUITE_API_PORT: {exc}", file=sys.stderr)
        return 2

    parser = _build_parser(default_host, default_port)
    # ``main()`` is also called by tests and by the CLI launcher.  Treat an
    # omitted argv as "no flags" there; the ``__main__`` path passes
    # ``sys.argv[1:]`` explicitly so command-line flags are honoured.
    args = parser.parse_args([] if argv is None else argv)

    ensure_safe_to_serve()

    if args.reload is None:
        reload_enabled = (_dev_mode() and os.environ.get("BIOSUITE_API_RELOAD", "1")
                          not in {"0", "false"})
    else:
        reload_enabled = args.reload

    uvicorn = _load_uvicorn()

    print("\n" + "=" * 60)
    print(f"  BioSuite Ultra v{__version__} | Pure-Python Bioinformatics Platform")
    print("  REST API server")
    print("=" * 60)
    print()
    print(f"  API Documentation: http://{args.host}:{args.port}/docs")
    print(f"  ReDoc Reference:   http://{args.host}:{args.port}/redoc")
    print(f"  Health Check:      http://{args.host}:{args.port}/health")
    print()
    print("  Bind address is --host or BIOSUITE_API_HOST (default: loopback only).")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    uvicorn.run(
        "biosuite.api:app",
        host=args.host,
        port=args.port,
        reload=reload_enabled,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
