#!/usr/bin/env bash
# Fetch the released policy and world model for the browser demo (the web/ exports of kevinhenryz/realmimic on Hugging
# Face) into site/assets/policy/, and check them against the SHA-256 list below. The files are not committed.
#   tools/fetch_site_weights.sh          # from Hugging Face: the hf CLI if installed (uses its login while the
#                                        # repository is private), else plain HTTPS (HF_TOKEN is sent if set)
#   tools/fetch_site_weights.sh <dir>    # from a local copy of the repository's web/ folder
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=kevinhenryz/realmimic
OUT=site/assets/policy
FILES=(policy.onnx meta.json world_model.onnx world_model.json)
declare -A SHA256=(
  [policy.onnx]=6128231c008e377b85a641d6683046c549518a54b1bcd774c76f1073a29f7c0c
  [meta.json]=91831c9eec12c37442afeb93e8ff7f90c112e6bd7ed331796239ab00b3d11e96
  [world_model.onnx]=24d45752b00913b0a091e3408e1c8f47a2773c7bbda082e316ad32d2ab8682a6
  [world_model.json]=0f57d26983261531e6f067824d168aa20b331bf66fb0b3d1f86901d4e210be76
)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for f in "${FILES[@]}"; do
  if [ $# -ge 1 ]; then
    cp "$1/$f" "$tmp/$f"
  elif command -v hf > /dev/null; then
    hf download "$REPO" "web/$f" --local-dir "$tmp/hf" > /dev/null
    mv "$tmp/hf/web/$f" "$tmp/$f"
  else
    curl -sfL ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} "https://huggingface.co/$REPO/resolve/main/web/$f" -o "$tmp/$f"
  fi
  echo "${SHA256[$f]}  $tmp/$f" | sha256sum -c --quiet || { echo "$f: checksum does not match the released file" >&2; exit 1; }
done
mkdir -p "$OUT"
for f in "${FILES[@]}"; do mv "$tmp/$f" "$OUT/$f"; done
echo "$OUT: released policy and world model fetched and verified"
