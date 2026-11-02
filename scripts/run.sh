#!/usr/bin/env bash
# Day 18 · Phase G — start the stack (API + optional side-cars via profiles).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found on PATH." >&2; exit 1; }

# Seed .env from the template on first run (no secrets — see .env.example).
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo ">> Created .env from .env.example"
fi

# The trained artifacts + data are MOUNTED read-only (not baked into the image) — they must exist.
for d in artifacts data; do
  [ -d "$d" ] || { echo "ERROR: ./$d not found — trained artifacts/data must be present (volume mount)." >&2; exit 1; }
done
mkdir -p logs

PROFILE_ARGS=()
[ "${WITH_PROXY:-0}" = "1" ] && PROFILE_ARGS+=(--profile proxy)
[ "${WITH_MLFLOW:-0}" = "1" ] && PROFILE_ARGS+=(--profile dev)

echo ">> Starting stack: docker compose ${PROFILE_ARGS[*]:-} up -d"
docker compose "${PROFILE_ARGS[@]}" up -d
echo ">> Up. Check readiness:  curl -fs localhost:\${PORT:-8000}/ready"
