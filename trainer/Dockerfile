# ──────────────────────────────────────────────────────
#  Trainer image
#  • Only needs Python runtime + FAISS + transformers
#  • No FastAPI / Uvicorn in this image.
# ──────────────────────────────────────────────────────
FROM python:3.11-slim AS trainer

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install matplotlib

COPY src /app/src
COPY trainer /app/trainer
COPY serve/entrypoint.sh /app/entrypoint.sh

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# No ENTRYPOINT here; will specify at runtime