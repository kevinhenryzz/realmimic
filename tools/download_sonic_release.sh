#!/usr/bin/env bash
# Download NVIDIA SONIC's released policy (encoder, decoder, observation config) for evaluating it in the
# benchmark: tools/download_sonic_release.sh [DIR]   (default third_party/sonic_release)
#
# The weights are NVIDIA's, under the NVIDIA Open Model License (https://huggingface.co/nvidia/GEAR-SONIC); this
# repository does not redistribute them. Evaluation only; deploying through SONIC's runtime uses deploy/sonic/.
set -euo pipefail
cd "$(dirname "$0")/.."
dir="${1:-third_party/sonic_release}"
for f in model_encoder.onnx model_decoder.onnx observation_config.yaml; do
  env -u PYTHONPATH uv run hf download nvidia/GEAR-SONIC "$f" --local-dir "$dir" >/dev/null
  echo "downloaded $dir/$f"
done
echo "Evaluate it with: uv run python scripts/eval_benchmark.py --sonic-release $dir --robot g1 ..."
