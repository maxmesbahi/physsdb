#!/usr/bin/env bash
# Run the dashboard locally without Docker (for code iteration).
# Assumes you have a python venv or conda env with requirements.txt installed
# and the runs/ directory populated.
set -euo pipefail
cd "$(dirname "$0")/.."

export SDB_MODEL_ROOT="${SDB_MODEL_ROOT:-$(pwd)/runs}"
export SDB_DATA_ROOT="${SDB_DATA_ROOT:-$(pwd)/data}"
export GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}"

PORT="${SDB_PORT:-7860}"
HOST="${SDB_HOST:-0.0.0.0}"

echo "→ runs:  $SDB_MODEL_ROOT"
echo "→ data:  $SDB_DATA_ROOT"
echo "→ url :  http://$HOST:$PORT"
exec python -u app.py --host "$HOST" --port "$PORT"
