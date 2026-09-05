"""
BioSuite Ultra API Server — Entry Point

Run with:
    python -m biosuite.api.server
    # or
    uvicorn biosuite.api.server:app --host 127.0.0.1 --port 8000

``app`` is re-exported here so that the documented ASGI target resolves; it is
the very same application object as ``biosuite.api:app``.

Startup refuses to bind a socket when the process would serve traffic with
missing or well-known credentials.  Set ``BIOSUITE_DEV_MODE=1`` to override for
local development.

Open API docs at: http://localhost:8000/docs
"""
import logging
import os
import sys

from biosuite import __version__
from biosuite.api import app  # noqa: F401  — the documented ASGI target
from biosuite.api.config import dev_mode, format_config_error, validate_runtime_config

logger = logging.getLogger(__name__)

__all__ = ["app", "main", "ensure_safe_to_serve"]


def ensure_safe_to_serve() -> None:
    """Abort startup when the runtime configuration is not production-safe."""
    problems = validate_runtime_config()
    if not problems:
        return
    if dev_mode():
        logger.warning("BIOSUITE_DEV_MODE is enabled — starting with an insecure "
                       "configuration:\n%s",
                       "\n".join(f"  - {p}" for p in problems))
        return
    raise SystemExit(format_config_error(problems))


def main(argv=None) -> int:
    import uvicorn

    ensure_safe_to_serve()

    host = os.environ.get("BIOSUITE_API_HOST", "127.0.0.1")
    port = int(os.environ.get("BIOSUITE_API_PORT", "8000"))
    reload_enabled = dev_mode() and os.environ.get("BIOSUITE_API_RELOAD", "1") not in {"0", "false"}

    print("\n" + "=" * 60)
    print(f"  BioSuite Ultra v{__version__} | Pure-Python Bioinformatics Platform")
    print("  REST API server")
    print("=" * 60)
    print()
    print(f"  API Documentation: http://{host}:{port}/docs")
    print(f"  ReDoc Reference:   http://{host}:{port}/redoc")
    print(f"  Health Check:      http://{host}:{port}/health")
    print()
    print("  Bind address is BIOSUITE_API_HOST (default: loopback only).")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    uvicorn.run(
        "biosuite.api:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
