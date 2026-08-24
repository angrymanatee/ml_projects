#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
MLFLOW_DIR="${MLFLOW_DIR:-/Volumes/Wokyis/ML/mlflow}"
HOST="${1:-127.0.0.1}"
PORT="${2:-5000}"

exec uv run --project "${REPO_ROOT}" mlflow gc \
    --backend-store-uri "sqlite:///${MLFLOW_DIR}/mlflow.db" \
    --artifacts-destination "${MLFLOW_DIR}/artifacts" \
    --tracking-uri "http://${HOST}:${PORT}"
