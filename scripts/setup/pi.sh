#!/usr/bin/env bash
# Install the Pi coding agent the terminal harness drives: the jdf-prog/pi-mono fork at the
# pinned commit, with tools/pi-patches applied, built under tools/pi-mono. Needs Node 20 on PATH.
set -euo pipefail

REPO_URL="https://github.com/jdf-prog/pi-mono.git"
BRANCH="codex/context-management-ablation"
COMMIT="a6be5eb4cce278de31ac05792af3dfc0883215dc"

root="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
pi="$root/tools/pi-mono"
die() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

command -v node >/dev/null || die "node is not on PATH; Pi needs Node 20"
[ "$(node -p 'process.versions.node.split(".")[0]')" -ge 20 ] || die "Node $(node -v) found; Pi needs Node 20 or later"
command -v npm >/dev/null || die "npm is not on PATH"

if [ ! -d "$pi/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$pi"
fi
git -C "$pi" fetch --quiet origin "$BRANCH" || true
git -C "$pi" checkout --quiet "$COMMIT"
head="$(git -C "$pi" rev-parse HEAD)"
[ "$head" = "$COMMIT" ] || die "tools/pi-mono is at $head, not $COMMIT"

for patch in "$root"/tools/pi-patches/*.patch; do
  name="$(basename "$patch")"
  if git -C "$pi" apply -R --check "$patch" 2>/dev/null; then
    echo "already applied: $name"
  elif git -C "$pi" apply --check "$patch"; then
    git -C "$pi" apply "$patch"
    echo "applied: $name"
  else
    die "patch does not apply: $name"
  fi
done

(cd "$pi" && npm install)
for package in tui ai agent coding-agent; do
  (cd "$pi/packages/$package" && npm run build)
done

grep -q 'usage = { include: true }' "$pi/packages/ai/dist/providers/openai-completions.js" \
  || die "the built ai package lacks patch 0001 (authoritative OpenRouter cost)"
grep -q '"get_usage"' "$pi/packages/coding-agent/dist/modes/rpc/rpc-mode.js" \
  || die "the built coding-agent package lacks patch 0002 (get_usage RPC)"
echo "Pi built at $pi/packages/coding-agent/dist/cli.js"
