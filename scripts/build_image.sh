#!/usr/bin/env bash
# Build the linux/amd64 image. Dependencies are resolved and downloaded on the host
# (through whatever package index uv is configured with) and copied into the image.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?usage: scripts/build_image.sh <image:tag>}
PLATFORM=(--python-version 3.12 --python-platform x86_64-manylinux_2_28)

uv pip compile pyproject.toml "${PLATFORM[@]}" --exclude-newer 2026-10-03T00:00:00Z --no-header -q -o requirements.lock.txt
rm -rf build/site-packages
uv pip install "${PLATFORM[@]}" --target build/site-packages --no-deps -q -r requirements.lock.txt
find build/site-packages -name '__pycache__' -prune -exec rm -rf {} +
rm -rf build/site-packages/bin  # console scripts carry host-specific shebangs; the image uses python -m
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build --platform linux/amd64 -f server/Dockerfile -t "$TAG" .
