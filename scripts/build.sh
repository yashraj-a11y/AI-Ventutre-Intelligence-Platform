#!/usr/bin/env bash
# Day 18 · Phase G — build the AI Venture Intelligence Platform image.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found on PATH." >&2; exit 1; }

IMAGE="${IMAGE:-avip-api:latest}"
echo ">> Building ${IMAGE} (multi-stage; CPU-only torch; NLTK + SBERT baked) ..."
docker build -t "${IMAGE}" .
echo ">> Built ${IMAGE}. Size:"
docker image ls "${IMAGE}" --format '   {{.Repository}}:{{.Tag}}  {{.Size}}'
