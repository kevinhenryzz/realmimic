#!/usr/bin/env bash
# Downloaded datasets -> converted 50 Hz npz clips -> paper-filtered BONES-SEED -> training library
# (data/library/bones_paper.pt, the input of `sm-train release`) and benchmark lists (data/bench/).
#
#   tools/prepare_data.sh [WORKERS]                  # or: sbatch -c 32 tools/prepare_data.sh 32
#
# Expects (see README, "Data"):
#   data/raw/bones_seed/g1.tar.gz      hf download bones-studio/seed g1.tar.gz --repo-type dataset (gated, academic licence)
#   data/raw/lafan1/g1/*.csv           optional, evaluation only:
#                                      hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset --include "g1/*"
# Steps already done are skipped, so the script can be re-run after an interruption.
#SBATCH --time=12:00:00
set -euo pipefail
export PYTHONUNBUFFERED=1  # live logs when output goes to a file (Slurm)
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"  # sbatch runs a spooled copy of this script
WORKERS=${1:-${SLURM_CPUS_PER_TASK:-16}}
run() { env -u PYTHONPATH uv run python "$@"; }

if [[ -d data/raw/lafan1/g1 && ! -d data/npz/lafan1 ]]; then
  run scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 \
    --input-fps 30 --workers "$WORKERS"
fi

if [[ ! -d data/raw/bones_seed/g1/csv ]]; then
  tar -xzf data/raw/bones_seed/g1.tar.gz -C data/raw/bones_seed
fi
# The converter skips clips whose npz already exists.
run scripts/data/csv_to_npz_batch.py --input-dir data/raw/bones_seed/g1/csv --output-dir data/npz/bones_seed \
  --input-fps 120 --workers "$WORKERS"
if [[ ! -f data/npz/bones_seed/manifest_filtered.csv ]]; then
  run scripts/data/filter_bones_seed.py --npz-dir data/npz/bones_seed --workers "$WORKERS"
fi

# The released model trains on the paper-filtered BONES-SEED clips only (LAFAN1 stays held out).
if [[ ! -f data/library/bones_paper.pt ]]; then
  run scripts/data/build_library.py --npz-dir data/npz/bones_seed --manifest data/npz/bones_seed/manifest_filtered.csv \
    --out data/library/bones_paper.pt --workers "$WORKERS"
fi
run tools/make_benchmark_sets.py data/npz/bones_seed/manifest_filtered.csv data/bench
# Compare with the md5 sums in the README ("Data").
md5sum data/npz/bones_seed/manifest_filtered.csv data/bench/full.txt data/bench/rand2000.txt data/bench/hard.txt
