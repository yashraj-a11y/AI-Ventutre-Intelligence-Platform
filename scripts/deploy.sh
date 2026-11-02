#!/usr/bin/env bash
# Day 18 · Phase G — build + (re)start + wait for readiness. One command to (re)deploy.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found on PATH." >&2; exit 1; }

echo "== Deploy: build =="
./scripts/build.sh

echo "== Deploy: (re)start =="
docker compose up -d

PORT="${PORT:-8000}"
echo "== Deploy: wait for readiness on :${PORT}/ready (model load ~tens of seconds) =="
for _ in $(seq 1 60); do
  if curl -fs "http://localhost:${PORT}/ready" >/dev/null 2>&1; then
    echo ">> Ready ✅"; exit 0
  fi
  sleep 3
done
echo "ERROR: timed out waiting for readiness; recent api logs:" >&2
docker compose logs --tail 50 api >&2 || true
exit 1
