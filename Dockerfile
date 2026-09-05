# BioSuite Ultra - Dockerfile
#
# Multi-stage build. The runtime image contains no build toolchain, runs as an
# unprivileged user and serves the API on loopback inside the container (the
# published port is what makes it reachable), so a misconfigured container is
# not silently exposed to the network with default credentials.

# ── Stage 1: build a wheel ──────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY biosuite ./biosuite

RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /dist

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Unprivileged runtime account. Running the API as root meant any RCE in a
# dependency was immediately root inside the container.
RUN useradd --create-home --uid 10001 biosuite

WORKDIR /app

COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl "biosuite-ultra[api]" \
    && rm -f /tmp/*.whl

# Writable locations for user config and mounted data.
ENV BIOSUITE_CONFIG_DIR=/home/biosuite/.config/biosuite \
    BIOSUITE_DATA_DIR=/data \
    BIOSUITE_API_HOST=0.0.0.0 \
    BIOSUITE_API_PORT=8000 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data /output "$BIOSUITE_CONFIG_DIR" \
    && chown -R biosuite:biosuite /data /output /home/biosuite

USER biosuite

EXPOSE 8000

# The server refuses to start unless BIOSUITE_API_KEY and BIOSUITE_JWT_SECRET
# are supplied, so no image ever ships with working default credentials.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; \
req=urllib.request.Request('http://127.0.0.1:8000/health', \
headers={'X-API-Key': os.environ.get('BIOSUITE_API_KEY','')}); \
urllib.request.urlopen(req, timeout=4)"

CMD ["python", "-m", "biosuite.api.server"]
