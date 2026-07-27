# AinSeba - single-container deployment (Hugging Face Docker Space, Fly.io, any Docker host)
# Runs FastAPI on 8000 (internal) and Streamlit on 7860 (public).
FROM python:3.11-slim

# HF Spaces runs as uid 1000 and needs a writable HOME for model caches.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    AINSEBA_API_URL=http://127.0.0.1:8000

WORKDIR $HOME/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && rm -rf /var/lib/apt/lists/*

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the cross-encoder at build time so the first request is fast
# and the running container needs no model download.
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" || \
    echo "Reranker prefetch skipped"

COPY --chown=user . .

USER user
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["bash", "start.sh"]
