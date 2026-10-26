#!/usr/bin/env bash
# Download the motion datasets into data/raw/, then run tools/prepare_data.sh.
#
#   tools/download_data.sh            # BONES-SEED (training + benchmark) and LAFAN1 (held-out evaluation)
#   tools/download_data.sh --no-lafan1
#
# BONES-SEED is gated: before the first run, open https://huggingface.co/datasets/bones-studio/seed while logged in,
# accept the BONES Motion Capture Dataset License Agreement, and log this machine in with `uv run hf auth login`.
# The data is never redistributed by this repository; each user downloads it under their own licence.
# Disk: the download is 23.5 GB; unpacked, converted and filtered, everything needs about 170 GB free.
# Uses the `hf` CLI that `uv sync` installs (huggingface_hub). Re-running skips files that are already complete.
set -euo pipefail
cd "$(dirname "$0")/.."

lafan1=1
for arg in "$@"; do
  case "$arg" in
    --no-lafan1) lafan1=0 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

hf() { env -u PYTHONPATH uv run hf "$@"; }

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Not logged in to Hugging Face. Run: uv run hf auth login" >&2
  echo "(create a read token at https://huggingface.co/settings/tokens)" >&2
  exit 1
fi

echo "== BONES-SEED, G1 retargeted clips (gated; a 23.5 GB download)"
if ! hf download bones-studio/seed g1.tar.gz --repo-type dataset --local-dir data/raw/bones_seed; then
  echo >&2
  echo "BONES-SEED download failed. If the error says access is restricted or gated:" >&2
  echo "  open https://huggingface.co/datasets/bones-studio/seed, accept the licence with the same account" >&2
  echo "  you logged in with, wait for access to be granted, then run this script again." >&2
  exit 1
fi

if ((lafan1)); then
  echo "== LAFAN1, G1 retargeted clips (held-out evaluation only)"
  hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset --include "g1/*" --local-dir data/raw/lafan1
fi

echo
echo "Done. Next: tools/prepare_data.sh   (converts, filters and builds the library and benchmark lists)"
