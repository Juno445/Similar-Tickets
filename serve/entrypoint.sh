#!/usr/bin/env bash
set -euo pipefail

# Ensure model cache directory has proper permissions
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/app/models}"
if [[ -d "$MODEL_CACHE_DIR" ]]; then
    echo "[entrypoint] Setting permissions for model cache directory: $MODEL_CACHE_DIR"
    find "$MODEL_CACHE_DIR" -type d -exec chmod 755 {} \; 2>/dev/null || true
    find "$MODEL_CACHE_DIR" -type f -exec chmod 644 {} \; 2>/dev/null || true
fi

echo "[entrypoint] waiting for initial snapshot …"
while [[ ! -f "${INDEX_PATH:-/shared}/current/meta.json" ]]; do
  sleep 2
done

echo "[entrypoint] snapshot found, starting FastAPI"
exec uvicorn src.webhook_server:app --host 0.0.0.0 --port 8000