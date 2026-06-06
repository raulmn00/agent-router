# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/router:/app/agents:/app/eval:/app/backend \
    ROUTER_MODEL_PATH=/app/router/model \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false

# System deps: curl to fetch the model at startup; libgomp1 for torch's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer).
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

# Copy package sources. The DistilBERT model dir is gitignored and fetched at
# startup by entrypoint.sh — but `eval/models/embed_router.joblib` (~25KB,
# fitted LogReg) is committed and travels in this layer.
COPY router/ /app/router/
COPY agents/ /app/agents/
COPY eval/ /app/eval/
COPY backend/ /app/backend/

# Strip the entrypoint script's CRLF if any, and make it executable.
RUN chmod +x /app/backend/entrypoint.sh

# Drop root — every layer above runs as root (apt, pip, chmod). From here on
# the process runs as a system user that owns /app so entrypoint.sh can write
# the fetched model into /app/router/model.
RUN useradd --system --uid 1000 --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app

EXPOSE 8000

# entrypoint.sh handles model fetching, then exec's the CMD.
ENTRYPOINT ["/app/backend/entrypoint.sh"]
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/backend"]
