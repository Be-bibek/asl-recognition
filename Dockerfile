# ── Stage 1: base ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for Pillow and OpenCV (if added later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: app ───────────────────────────────────────────────────────────
FROM base AS app

COPY . .

# Expose port for future FastAPI integration
EXPOSE 8000

# Default: run training. Override CMD to run inference / API.
CMD ["python", "train.py"]
