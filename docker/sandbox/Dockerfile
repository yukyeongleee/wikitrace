# Sandbox image for the Pi terminal agent.
#
# Pi runs INSIDE this container, so whatever the terminal agent can compute with is
# exactly what this image ships. node is there because Pi itself is a node CLI; python
# comes from Debian packages so the build needs no network beyond apt, and
# `python-is-python3` gives a bare `python`, which agents probe before `python3`.
#
# Build + tag (the code default, TraverseConfig.docker_image, IS this tag):
#   scripts/setup/sandbox_image.sh   # or: docker build -t wikitrace-sandbox:node20 docker/sandbox
# To pin a build, set `traverse.docker_image` in the config to the printed image id.
FROM node:20-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-numpy python-is-python3 \
    && rm -rf /var/lib/apt/lists/*
