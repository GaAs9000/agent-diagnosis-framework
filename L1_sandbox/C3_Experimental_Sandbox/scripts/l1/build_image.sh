#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v docker >/dev/null || { echo "Docker is required; no host fallback." >&2; exit 2; }
docker build -f docker/Dockerfile.l1-office -t l1-office:0.2.0 .
docker image inspect l1-office:0.2.0 --format '{{.Id}}'
