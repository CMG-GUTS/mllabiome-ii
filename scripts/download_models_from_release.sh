#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-}"
RELEASE="${2:-models-v0.1.0}"
OUT_DIR="${3:-app/backend/production_models}"

if [[ -z "$REPO" ]]; then
  echo "Usage: scripts/download_models_from_release.sh OWNER/REPO [models-v0.1.0] [app/backend/production_models]" >&2
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI 'gh' is required: https://cli.github.com/" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
gh release download "$RELEASE" --repo "$REPO" --pattern "*_model_v*.zip" --dir "$OUT_DIR" --clobber
gh release download "$RELEASE" --repo "$REPO" --pattern "SHA256SUMS.txt" --dir "$OUT_DIR" --clobber || true
gh release download "$RELEASE" --repo "$REPO" --pattern "model_manifest.json" --dir "$OUT_DIR" --clobber || true

if [[ -f "$OUT_DIR/SHA256SUMS.txt" ]]; then
  (cd "$OUT_DIR" && shasum -a 256 -c SHA256SUMS.txt)
else
  echo "Downloaded model assets. No SHA256SUMS.txt was found to verify." >&2
fi
