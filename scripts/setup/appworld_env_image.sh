#!/usr/bin/env bash
# Build the AppWorld episode image: the install's own appworld and dependency set, plus our
# environment server. Its tag names the AppWorld version and a digest of the files it is built
# from (the server code, the Dockerfile, this script), so rebuild whenever one changes; an
# AppWorld run checks for the tag before its first episode and refuses to start without it.
set -euo pipefail

root="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
install="$(readlink -f "${APPWORLD_INSTALL:-$root/data/appworld}")"
site="$(echo "$install"/.venv/lib/python3.11/site-packages)"
[ -d "$site/appworld/apps" ] || { echo "no unpacked appworld under $site; run appworld.sh" >&2; exit 1; }

tag="$(cd "$root" && uv run --no-sync python -c \
  'import sys; from wikitrace.envs.appworld.session import image_tag; print(image_tag(sys.argv[1]))' \
  "$install")"

ctx="$(mktemp -d)"
trap 'rm -rf "$ctx"' EXIT
mkdir -p "$ctx/site" "$ctx/srv/envs"
cp "$root/docker/appworld-env/Dockerfile" "$ctx/"
uv pip freeze --color never --python "$install/.venv/bin/python" | grep -v '^appworld ' > "$ctx/requirements.txt"
cp -r "$site/appworld" "$site"/appworld-*.dist-info "$ctx/site/"
find "$ctx/site" -name __pycache__ -prune -exec rm -rf {} +
# The install record names a host path; the image has no use for it.
rm -f "$ctx"/site/appworld-*.dist-info/direct_url.json
cp "$root/src/wikitrace/harness/environment/server.py" \
   "$root/src/wikitrace/harness/environment/protocol.py" "$ctx/srv/"
# Beside server.py, whose dir is on sys.path, it would shadow the `appworld` package.
cp "$root/src/wikitrace/envs/appworld/envs/appworld.py" "$ctx/srv/envs/"

docker build -t "$tag" "$ctx"
echo "built $tag"
