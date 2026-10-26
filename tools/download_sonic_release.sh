#!/usr/bin/env bash
# Download NVIDIA SONIC's released policy (encoder, decoder, observation config) for evaluating it in the
# benchmark, and SONIC's kinematic planner for generating planner motion clips:
#   tools/download_sonic_release.sh [DIR]   (policy into DIR, default third_party/sonic_release;
#                                            planner into third_party/gear_sonic/planner_sonic.onnx)
#
# The weights are NVIDIA's, under the NVIDIA Open Model License (https://huggingface.co/nvidia/GEAR-SONIC); this
# repository does not redistribute them. Deploying through SONIC's runtime uses deploy/sonic/setup_sim.sh instead.
set -euo pipefail
cd "$(dirname "$0")/.."
dir="${1:-third_party/sonic_release}"
fetch() { env -u PYTHONPATH uv run hf download nvidia/GEAR-SONIC "$1" --local-dir "$2" >/dev/null && echo "downloaded $2/$1"; }
for f in model_encoder.onnx model_decoder.onnx observation_config.yaml; do fetch "$f" "$dir"; done
fetch planner_sonic.onnx third_party/gear_sonic
echo "Evaluate SONIC with: uv run python scripts/eval_benchmark.py --sonic-release $dir --robot g1 ..."
