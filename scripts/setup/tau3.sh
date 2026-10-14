#!/usr/bin/env bash
#
# Install tau3 (tau2-bench): the authors' package in its own venv, from a clone at the
# pinned commit, with every dependency at the version the clone's uv.lock pins. The retail
# data (db.json, tasks.json, policy.md) ships in the clone's data/ dir, which the package
# resolves from its source tree, so the install is editable.
#
# Target layout (see src/wikitrace/envs/tau3/):
#   $DATA_ROOT/tau3/repo                       # sierra-research/tau2-bench @ $TAU2_COMMIT
#   $DATA_ROOT/tau3/.venv/bin/python           # tau2 (no voice extra), CPython 3.12
#   $DATA_ROOT/tau3/.venv/.uv-lock.sha256      # the uv.lock the venv was synced from
#
# Usage:
#   scripts/setup/tau3.sh
#   DATA_ROOT=/mnt/data scripts/setup/tau3.sh

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

BENCH="tau3"
DEST="$DATA_ROOT/$BENCH"
VENV="$DEST/.venv"
CLONE="$DEST/repo"

TAU2_REPO_URL="${TAU2_REPO_URL:-https://github.com/sierra-research/tau2-bench}"
TAU2_COMMIT="${TAU2_COMMIT:-fc0055dc4e0a316c3f83133267fbd6faaa770992}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# uv's default cache lives under $HOME, which is quota-limited on the lab hosts.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}" UV_LINK_MODE="${UV_LINK_MODE:-copy}"

need uv
need git
mkdir -p "$DEST"

if [ "$(git -C "$CLONE" rev-parse HEAD 2>/dev/null || true)" = "$TAU2_COMMIT" ]; then
  log "clone already at $TAU2_COMMIT: $CLONE — skipping"
else
  log "clone: $TAU2_REPO_URL @ $TAU2_COMMIT -> $CLONE"
  rm -rf "$CLONE"
  git init -q "$CLONE"
  git -C "$CLONE" remote add origin "$TAU2_REPO_URL"
  # One commit, no history: the pinned tree is all the environment reads.
  git -C "$CLONE" fetch -q --depth 1 origin "$TAU2_COMMIT"
  git -C "$CLONE" checkout -q FETCH_HEAD
fi
[ "$(git -C "$CLONE" rev-parse HEAD)" = "$TAU2_COMMIT" ] || die "clone at $CLONE is not at $TAU2_COMMIT"

CHECK="import tau2.orchestrator.orchestrator, tau2.evaluator.evaluator; from tau2.registry import registry; registry.get_env_constructor('retail')()"
LOCK_MARK="$VENV/.uv-lock.sha256"
LOCK_SHA="$(sha256sum "$CLONE/uv.lock" | cut -d' ' -f1)"
if [ "$(cat "$LOCK_MARK" 2>/dev/null || true)" = "$LOCK_SHA" ] && "$VENV/bin/python" -c "$CHECK" >/dev/null 2>&1; then
  log "venv already synced from $CLONE/uv.lock at $VENV — skipping"
else
  log "venv: $VENV (python $PYTHON_VERSION), every dependency at the version $CLONE/uv.lock pins"
  # An exact sync: a package the lock does not name is removed, one at another version replaced.
  UV_PROJECT_ENVIRONMENT="$VENV" uv sync -q --frozen --no-dev --python "$PYTHON_VERSION" --project "$CLONE"
  echo "$LOCK_SHA" > "$LOCK_MARK"
fi
"$VENV/bin/python" -c "$CHECK" >/dev/null 2>&1 \
  || die "tau3 venv at $VENV does not import the retail environment and the evaluator"
assert_file "$CLONE/data/tau2/domains/retail/tasks.json"
assert_file "$CLONE/data/tau2/domains/retail/db.json"
assert_file "$CLONE/data/tau2/domains/retail/policy.md"
assert_file "$CLONE/data/tau2/user_simulator/simulation_guidelines.md"

cat >&2 <<EOF

[install] tau3 ready under: $DEST
[install] Point the config at it (default already matches):
             dataset:
               tau3_root: $DEST
          The environment server runs under $VENV/bin/python.
[install] Suite id: retail.
EOF
