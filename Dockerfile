# BioSuite Ultra — Multi-stage Dockerfile
# API: FastAPI + uvicorn  |  GUI: PyQt6 + xvfb

# ── Base ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml setup.py MANIFEST.in ./
COPY biosuite/ biosuite/

RUN pip install --no-cache-dir -e ".[api]"

# ── API ───────────────────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["python", "-m", "biosuite.api.main"]

# ── GUI (headless) ───────────────────────────────────────────────────
FROM base AS gui

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb libxkbcommon-x11-0 libgl1-mesa-glx libegl1-mesa \
    libfontconfig1 libdbus-1-3 libxcb-xinerama0 libxcb-cursor0 \
    fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -e ".[gui]"

ENV QT_QPA_PLATFORM=offscreen \
    DISPLAY=:99

CMD Xvfb :99 -screen 0 1920x1080x24 &>/dev/null & \
    sleep 1 && python -m biosuite.gui.main_window
