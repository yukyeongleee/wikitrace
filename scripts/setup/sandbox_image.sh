#!/usr/bin/env bash
# Build the terminal-agent sandbox image (node 20, python3, numpy).
#
# Pi runs inside this container, so every live run needs it. See
# docker/sandbox/Dockerfile. Offline tests never build or run it (FakeTraverseRunner).
set -euo pipefail

TAG="wikitrace-sandbox:node20"
root="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"

docker build -t "$TAG" "$root/docker/sandbox"
echo "built $TAG"
# The local image id pins this build: set it as `traverse.docker_image` in the config.
docker image inspect --format 'local image id: {{.Id}}' "$TAG"
