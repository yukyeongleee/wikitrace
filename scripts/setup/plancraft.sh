#!/usr/bin/env bash
#
# Install Plancraft: the simulator package in its own venv. The examples ship inside the
# package (plancraft/data/{train,val,test}.json), so there is no separate dataset.
#
# Target layout (see src/wikitrace/envs/textgame/):
#   $DATA_ROOT/plancraft/.venv/bin/python       # plancraft 0.4.9 + CPU torch, CPython 3.11
#
# Usage:
#   scripts/setup/plancraft.sh
#   DATA_ROOT=/mnt/data scripts/setup/plancraft.sh
#
# Sources: the official GitHub repo at the commit the design record pins, and the CPU
# wheels of torch/torchvision. The base install is not enough for the authors' evaluator:
# plancraft/models imports torch, einops and transformers unconditionally (the `full` extra).

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

BENCH="plancraft"
DEST="$DATA_ROOT/$BENCH"
VENV="$DEST/.venv"

PLANCRAFT_REPO_URL="${PLANCRAFT_REPO_URL:-https://github.com/gautierdag/plancraft}"
PLANCRAFT_COMMIT="${PLANCRAFT_COMMIT:-7617e9ff0c68e3158e3ba8ea260e6d83c454ba6f}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

# uv's default cache lives under $HOME, which is quota-limited on the lab hosts.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}" UV_LINK_MODE="${UV_LINK_MODE:-copy}"

need uv
mkdir -p "$DEST"

if "$VENV/bin/python" -c "import plancraft.environment, plancraft.evaluator, torch" 2>/dev/null; then
  log "venv already installed at $VENV — skipping"
else
  log "venv: $VENV (python $PYTHON_VERSION)"
  uv venv -q --clear --python "$PYTHON_VERSION" "$VENV"
  # torch first from the CPU index, so the package's own resolution does not pull a CUDA build.
  uv pip install -q --python "$VENV/bin/python" --index-url "$TORCH_INDEX_URL" \
    "torch==2.14.0" "torchvision==0.29.0"
  uv pip install -q --python "$VENV/bin/python" \
    "plancraft @ git+$PLANCRAFT_REPO_URL@$PLANCRAFT_COMMIT" \
    einops huggingface-hub hf-transfer "transformers==4.52.3"
fi
"$VENV/bin/python" -c "import plancraft.environment, plancraft.evaluator, torch" \
  || die "plancraft venv at $VENV does not import the simulator and evaluator"
assert_file "$VENV/lib/python$PYTHON_VERSION/site-packages/plancraft/data/test.json"

cat >&2 <<EOF

[install] Plancraft ready under: $DEST
[install] Point the config at it (default already matches):
             dataset:
               plancraft_root: $DEST
          The environment server runs under $VENV/bin/python.
[install] Suite ids: plancraft-seen, plancraft-unseen.
EOF
