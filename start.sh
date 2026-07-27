#!/usr/bin/env bash
# Start the FastAPI backend, wait for it to warm, then start Streamlit in the
# foreground so the container's lifecycle follows the UI process.
set -euo pipefail

echo "[ainseba] starting FastAPI on :8000"
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --log-level info &
API_PID=$!

# The API warms the chain (and cross-encoder) at startup; poll until ready.
echo "[ainseba] waiting for backend warm-up..."
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "[ainseba] backend ready after ${i}s"
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "[ainseba] FATAL: backend died during startup" >&2
    exit 1
  fi
  sleep 1
done

echo "[ainseba] starting Streamlit on :${STREAMLIT_SERVER_PORT:-7860}"
exec streamlit run frontend/app.py \
  --server.port "${STREAMLIT_SERVER_PORT:-7860}" \
  --server.address 0.0.0.0 \
  --server.headless true
