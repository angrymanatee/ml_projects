#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "==> black (format)"
uv run black .

echo "==> ruff (lint + fix)"
uv run ruff check --fix .

echo "==> pytest"
uv run pytest

echo "==> pre-commit (full suite)"
uv run pre-commit run --all-files

echo "==> all checks passed"
