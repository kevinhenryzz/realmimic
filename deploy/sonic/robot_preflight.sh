#!/usr/bin/env bash
# Read-only preflight on the robot's onboard computer, run over SSH before building or deploying.
# It inspects; it never writes anywhere except stdout.
#
#   ssh <robot> 'bash -s' < deploy/sonic/robot_preflight.sh [EXISTING_SONIC_DIR]
#
# EXISTING_SONIC_DIR: an already-installed SONIC checkout to report on (left untouched).
set -uo pipefail

EXISTING="${1:-}"
section() { printf '\n== %s\n' "$*"; }

section "platform"
cat /etc/nv_tegra_release 2>/dev/null || echo "not a Jetson (no /etc/nv_tegra_release)"
lsb_release -ds 2>/dev/null; uname -rm
command -v nvcc >/dev/null && nvcc --version | tail -1 || ls -d /usr/local/cuda* 2>/dev/null
gcc --version | head -1; cmake --version | head -1; command -v just && just --version

section "TensorRT / ONNX Runtime candidates"
for d in "${TensorRT_ROOT:-}" /usr/lib/aarch64-linux-gnu "$HOME"/*/deps/TensorRT; do
  [[ -n "$d" && -e "$d" ]] && ls "$d"/lib*/libnvinfer.so.* "$d"/libnvinfer.so.* 2>/dev/null | head -2
done
ls -d /opt/onnxruntime 2>/dev/null && ls /opt/onnxruntime/lib | grep -E 'libonnxruntime\.so' | head -3

section "build headers"
for h in /usr/include/zmq.h /usr/include/msgpack.hpp /usr/include/eigen3/Eigen/Core \
         /usr/include/yaml-cpp/yaml.h /usr/include/nlohmann/json.hpp; do
  [[ -e "$h" ]] && echo "ok      $h" || echo "MISSING $h"
done

section "network"
ip -4 -br addr | grep -E '192\.168\.123\.' || echo "no 192.168.123.x address (robot DDS bus)"
ping -c1 -W1 github.com >/dev/null 2>&1 && echo "internet: yes" || echo "internet: no"

section "resources"
df -h "$HOME" | tail -1
free -g | head -2
nvpmodel -q 2>/dev/null | head -2

section "running low-level controllers (must be none during our tests)"
pgrep -af 'g1_deploy|lowcmd|deploy.sh' || echo "none"

if [[ -n "$EXISTING" ]]; then
  section "existing SONIC checkout (read-only): $EXISTING"
  git -C "$EXISTING" rev-parse HEAD
  git -C "$EXISTING" status --porcelain --untracked-files=no | head -20
  echo "($(git -C "$EXISTING" status --porcelain --untracked-files=no | wc -l) modified tracked files)"
fi
