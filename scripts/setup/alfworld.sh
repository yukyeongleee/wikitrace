#!/usr/bin/env bash
#
# Install ALFWorld (TextWorld-only path): the simulator package in its own venv plus
# the game data, into the layout the environment adapter assumes.
#
# Target layout (see src/wikitrace/envs/textgame/):
#   $DATA_ROOT/alfworld/.venv/bin/python        # alfworld 0.5.0 + textworld[pddl], CPython 3.11
#   $DATA_ROOT/alfworld/json_2.1.1/{train,valid_train,valid_seen,valid_unseen}/
#   $DATA_ROOT/alfworld/logic/{alfred.pddl,alfred.twl2}
#   $DATA_ROOT/alfworld/detectors/mrcnn_alfred_objects_sep13_004.pth   # unused, fetched by alfworld-download
#
# Usage:
#   scripts/setup/alfworld.sh
#   DATA_ROOT=/mnt/data scripts/setup/alfworld.sh
#
# Sources: the official GitHub repo at the commit the probe pinned (0.5.0 is not on PyPI),
# and the 0.2.2/0.4.2 release archives that `alfworld-download` fetches (2.3 GB on disk).
# The package reads $ALFWORLD_DATA at import; every run must export it as this directory.

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

BENCH="alfworld"
DEST="$DATA_ROOT/$BENCH"
CACHE="$CACHE_ROOT/$BENCH"
VENV="$DEST/.venv"

ALFWORLD_REPO_URL="${ALFWORLD_REPO_URL:-https://github.com/alfworld/alfworld.git}"
ALFWORLD_COMMIT="${ALFWORLD_COMMIT:-aaba6870f86c5be6a08a491f32a50b906227bc3e}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# uv's default cache lives under $HOME, which is quota-limited on the lab hosts.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}" UV_LINK_MODE="${UV_LINK_MODE:-copy}"

need uv
mkdir -p "$DEST" "$CACHE"

if ALFWORLD_DATA="$DEST" "$VENV/bin/python" -c "import alfworld, textworld" 2>/dev/null; then
  log "venv already installed at $VENV — skipping"
else
  log "venv: $VENV (python $PYTHON_VERSION)"
  uv venv -q --clear --python "$PYTHON_VERSION" "$VENV"
  uv pip install -q --python "$VENV/bin/python" "git+$ALFWORLD_REPO_URL@$ALFWORLD_COMMIT"
fi
assert_file "$VENV/bin/alfworld-download"

data_ready() {
  for p in json_2.1.1/train json_2.1.1/valid_seen json_2.1.1/valid_unseen \
           logic/alfred.pddl logic/alfred.twl2 detectors/mrcnn_alfred_objects_sep13_004.pth; do
    [ -e "$DEST/$p" ] || return 1
  done
}

if data_ready; then
  log "game data already installed under $DEST — skipping"
else
  # Partial downloads resume from $TMPDIR; the archives themselves are deleted after extraction.
  log "download game data (2.3 GB) -> $DEST"
  TMPDIR="$CACHE" ALFWORLD_DATA="$DEST" "$VENV/bin/alfworld-download" --data-dir "$DEST"
  data_ready || die "alfworld-download finished but the layout is incomplete under $DEST"
fi

cat >&2 <<EOF

[install] ALFWorld ready under: $DEST
[install] Point the config at it (default already matches):
             dataset:
               alfworld_root: $DEST
[install] Every process that imports alfworld must see:
             ALFWORLD_DATA=$DEST
          The environment server runs under $VENV/bin/python.
[install] Suite ids: valid-seen, valid-unseen.
EOF
