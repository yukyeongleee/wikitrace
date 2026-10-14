# shellcheck shell=bash
# Shared helpers for benchmark install scripts.
#
# Conventions every scripts/setup/<benchmark>.sh follows:
#   - Target dir is $DATA_ROOT/<benchmark>  (DATA_ROOT defaults to <repo>/data).
#   - Idempotent: an already-present, verified layout is skipped, not re-downloaded.
#   - No credentials in the repo. Secrets (HF_TOKEN, service-account json) come from
#     the environment and are never echoed.
#   - Downloads go to a cache ($DATA_ROOT/.cache/<benchmark>) so a failed extract can
#     resume without re-fetching.
#
# Source this file from a benchmark script:
#   SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
#   . "$SCRIPT_DIR/_lib.sh"

set -euo pipefail

# Repo root = two levels up from scripts/setup/.
_LIB_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$_LIB_DIR/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
CACHE_ROOT="${CACHE_ROOT:-$DATA_ROOT/.cache}"

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"; }

# fetch <url> <dest-file>  — download unless dest already exists (non-empty).
fetch() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then
    log "cached: $(basename "$dest")"
    return 0
  fi
  log "download: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "$dest.part" "$url"
  else
    need wget
    wget -O "$dest.part" "$url"
  fi
  mv "$dest.part" "$dest"
}

# fetch_gdrive <file-id> <dest-file>  — Google Drive (large files); needs `gdown`.
# Some gdown versions exit 0 on a quota/permission miss, so assert the artifact.
fetch_gdrive() {
  local id="$1" dest="$2"
  [ -s "$dest" ] && { log "cached: $(basename "$dest")"; return 0; }
  need gdown
  mkdir -p "$(dirname "$dest")"
  log "gdrive: $id"
  gdown --id "$id" -O "$dest" || true
  [ -s "$dest" ] || die "gdown failed for $id (Drive quota? retry later): $dest"
}

# verify_sha256 <file> <expected>  — skip when expected is empty (unknown/unpinned).
verify_sha256() {
  local file="$1" want="$2"
  [ -z "$want" ] && return 0
  need sha256sum
  local got; got="$(sha256sum "$file" | awk '{print $1}')"
  [ "$got" = "$want" ] || die "checksum mismatch for $file: got $got want $want"
  log "verified sha256: $(basename "$file")"
}

# extract <archive> <dest-dir>  — zip / tar.gz / tgz.
extract() {
  local arc="$1" dest="$2"
  mkdir -p "$dest"
  case "$arc" in
    *.zip)            need unzip; unzip -q -o "$arc" -d "$dest" ;;
    *.tar.gz|*.tgz)   tar -xzf "$arc" -C "$dest" ;;
    *.tar)            tar -xf "$arc" -C "$dest" ;;
    *) die "unknown archive type: $arc" ;;
  esac
}

# assert_file <path>  — hard-fail if a required post-install artifact is missing.
assert_file() { [ -e "$1" ] || die "expected artifact missing: $1"; }
