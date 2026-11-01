#!/usr/bin/env bash
# Download official Microduck ONNX policies into work/local/scratch/policies.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/work/local/scratch/policies"
REPO="${POLICY_REPO:-pollen-robotics/microduck-policies}"
REV="${POLICY_VERSION:-main}"
BASE="https://huggingface.co/${REPO}/resolve/${REV}"

mkdir -p "$OUT"
files=(
  manifest.json
  alpha_walking.onnx
  alpha_stand.onnx
  alpha_sitstand.onnx
  alpha_ground_pick.onnx
  roulade.onnx
  ball_kick_left.onnx
  ball_kick_right.onnx
  velstand.onnx
)

echo "Downloading policies from ${BASE}"
for f in "${files[@]}"; do
  dest="$OUT/$f"
  if [[ -f "$dest" ]]; then
    echo "skip: $f"
    continue
  fi
  echo "get: $f"
  # Prefer huggingface-cli if available; else curl.
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO" "$f" --revision "$REV" --local-dir "$OUT" --local-dir-use-symlinks False
  else
    curl -fL --retry 3 -o "$dest" "${BASE}/$f" || {
      echo "WARN: failed to download $f (optional files may 404 on some revisions)"
      rm -f "$dest"
    }
  fi
done

ls -la "$OUT"
echo "OK: policies under $OUT"
