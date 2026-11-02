#!/usr/bin/env bash
# Day 18 · Phase G — stop the stack. Pass extra flags through, e.g. `stop.sh --volumes` to clean data.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found on PATH." >&2; exit 1; }

echo ">> Stopping stack (including optional side-cars) ..."
docker compose --profile proxy --profile dev down "$@"
echo ">> Stopped."
