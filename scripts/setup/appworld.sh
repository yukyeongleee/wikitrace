#!/usr/bin/env bash
#
# Install AppWorld: the simulator package in its own venv, its unpacked app sources, and
# the benchmark data, into the layout the environment adapter assumes.
#
# Target layout (see src/wikitrace/envs/appworld/):
#   $DATA_ROOT/appworld/.venv/bin/python        # appworld 0.2.0.dev0, CPython 3.11
#   $DATA_ROOT/appworld/data/{tasks,api_docs,base_dbs,datasets}/   # `appworld download data`
#   $DATA_ROOT/appworld/.cache/tests/            # `appworld install` (package mode) unpacks here
#
# Usage:
#   scripts/setup/appworld.sh
#   DATA_ROOT=/mnt/data scripts/setup/appworld.sh
#
# Sources: the official GitHub repo at the commit the probe pinned (the 0.2.0 data format is
# not served by the PyPI release), whose app sources ship as git-lfs bundles, so `git-lfs` is
# required; and the data bundle (198 MB) from the maintainers' S3 bucket. The package resolves
# every path from APPWORLD_ROOT (data at $APPWORLD_ROOT/data), so every run exports it as this
# directory, and APPWORLD_CACHE keeps the unpacked tests out of $HOME.

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

BENCH="appworld"
DEST="$DATA_ROOT/$BENCH"
CACHE="$CACHE_ROOT/$BENCH"
VENV="$DEST/.venv"

APPWORLD_REPO_URL="${APPWORLD_REPO_URL:-https://github.com/StonyBrookNLP/appworld.git}"
APPWORLD_COMMIT="${APPWORLD_COMMIT:-42b5bcf3cd334fee33f0c37c02070a9f5807add5}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# uv's default cache lives under $HOME, which is quota-limited on the lab hosts.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}" UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export APPWORLD_ROOT="$DEST" APPWORLD_CACHE="$DEST/.cache"

SITE="$VENV/lib/python$PYTHON_VERSION/site-packages/appworld"

need uv
mkdir -p "$DEST" "$CACHE"

# Pinned clone; the LFS bundles are pulled explicitly so a pointer never reaches the wheel.
clone_repo() {
  local clone="$CACHE/repo"
  if [ ! -f "$clone/src/appworld/.source/apps.bundle" ]; then
    need git; need git-lfs
    rm -rf "$clone"
    git init -q "$clone"
    git -C "$clone" remote add origin "$APPWORLD_REPO_URL"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$clone" fetch -q --depth 1 origin "$APPWORLD_COMMIT"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$clone" checkout -q FETCH_HEAD
    git -C "$clone" lfs pull >&2
  fi
  [ "$(stat -c %s "$clone/src/appworld/.source/apps.bundle")" -gt 4096 ] \
    || die "apps.bundle is still an LFS pointer in $clone"
  echo "$clone"
}

# jinja2 is for the parity runner (scripts/experiments/reference), which renders the
# shipped instructions.txt as the baseline agent does.
if [ -f "$SITE/.source/apps.bundle" ] && "$VENV/bin/python" -c "import appworld, jinja2" 2>/dev/null; then
  log "venv already installed at $VENV — skipping"
else
  clone="$(clone_repo)"
  log "venv: $VENV (python $PYTHON_VERSION)"
  uv venv -q --clear --python "$PYTHON_VERSION" "$VENV"
  uv pip install -q --python "$VENV/bin/python" "$clone" jinja2
fi
assert_file "$SITE/.source/apps.bundle"

if [ -d "$SITE/apps/api_docs" ] && [ -d "$APPWORLD_CACHE/tests/package/apps" ]; then
  log "app sources already unpacked — skipping"
else
  log "appworld install (unpack app sources into the venv, tests into $APPWORLD_CACHE)"
  "$VENV/bin/appworld" install
fi
assert_file "$SITE/apps/api_docs"

# `download data` deletes and re-downloads an existing data/, so skip on a verified layout.
if [ -d "$DEST/data/tasks" ] && [ -d "$DEST/data/api_docs" ]; then
  log "data already installed under $DEST/data — skipping"
else
  log "appworld download data -> $DEST/data"
  "$VENV/bin/appworld" download data --root "$DEST"
fi
assert_file "$DEST/data/tasks"
assert_file "$DEST/data/api_docs"
assert_file "$DEST/data/datasets/train.txt"

cat >&2 <<EOF

[install] AppWorld ready under: $DEST  (data version $(cat "$DEST/data/version.txt"))
[install] Point the config at it (default already matches):
             dataset:
               appworld_root: $DEST
[install] Every process that imports appworld must see:
             APPWORLD_ROOT=$DEST
             APPWORLD_CACHE=$DEST/.cache
          The environment server runs under $VENV/bin/python.
[install] Suite ids: test-normal, test-challenge.
EOF
