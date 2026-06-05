# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/router:/app/agents:/app/backend \
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

# Copy package sources. The model directory is gitignored and won't be in the
# build context — it gets fetched at startup by entrypoint.sh (see below).
COPY router/ /app/router/
COPY agents/ /app/agents/
COPY backend/ /app/backend/

# Strip the entrypoint script's CRLF if any, and make it executable.
RUN chmod +x /app/backend/entrypoint.sh

EXPOSE 8000

# entrypoint.sh handles model fetching, then exec's the CMD.
ENTRYPOINT ["/app/backend/entrypoint.sh"]
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/backend"]
