#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
MLFLOW_DIR="${REPO_ROOT}/mlflow"
HOST="${1:-localhost}"
PORT="${2:-5000}"

mkdir -p "${MLFLOW_DIR}"

DB="${MLFLOW_DIR}/mlflow.db"
if [[ -f "${DB}" ]] && uv run --project "${REPO_ROOT}" python -c "
import sqlite3, sys
tables = {r[0] for r in sqlite3.connect('${DB}').execute(\"SELECT name FROM sqlite_master WHERE type='table'\")}
sys.exit(0 if 'experiments' in tables else 1)
"; then
    uv run --project "${REPO_ROOT}" mlflow db upgrade "sqlite:///${DB}"
fi

exec uv run --project "${REPO_ROOT}" mlflow server \
    --backend-store-uri "sqlite:///${MLFLOW_DIR}/mlflow.db" \
    --artifacts-destination "${MLFLOW_DIR}/artifacts" \
    --host "${HOST}" \
    --port "${PORT}"
