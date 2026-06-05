#!/usr/bin/env bash
# Container entrypoint: ensure the DistilBERT router model is on disk, then
# hand off to the CMD (usually uvicorn).
#
# The 256 MB trained model is NOT in the git repo or the container image. We
# fetch it from a configurable URL on first start. For a portfolio deploy:
# tar up `router/model/`, attach to a GitHub Release, and set MODEL_RELEASE_URL
# to that release asset URL.
#
#   tar -czf model.tar.gz -C router model
#   gh release upload v0.1.0 model.tar.gz
#   export MODEL_RELEASE_URL=https://github.com/<you>/agent-router/releases/download/v0.1.0/model.tar.gz

set -euo pipefail

MODEL_DIR="${ROUTER_MODEL_PATH:-/app/router/model}"

if [ ! -f "${MODEL_DIR}/config.json" ]; then
    if [ -z "${MODEL_RELEASE_URL:-}" ]; then
        cat >&2 <<EOF
ERROR: router model not found at ${MODEL_DIR} and MODEL_RELEASE_URL is not set.

Either:
  1. Build the image with the model already present (mount or COPY router/model
     into the build context — note that .dockerignore currently excludes it), or
  2. Set MODEL_RELEASE_URL to a tarball produced by:
       tar -czf model.tar.gz -C router model
     The entrypoint will fetch it on container start.
EOF
        exit 1
    fi

    echo "[entrypoint] fetching model from \${MODEL_RELEASE_URL}"
    mkdir -p "$(dirname "${MODEL_DIR}")"
    curl --fail --silent --show-error --location "${MODEL_RELEASE_URL}" \
        | tar -xz -C "$(dirname "${MODEL_DIR}")"
    if [ ! -f "${MODEL_DIR}/config.json" ]; then
        echo "ERROR: model tarball did not contain ${MODEL_DIR}/config.json" >&2
        echo "       tar layout should be: model/{config.json,model.safetensors,tokenizer*}" >&2
        exit 1
    fi
    echo "[entrypoint] model ready at ${MODEL_DIR}"
fi

exec "$@"
