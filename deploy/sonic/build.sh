#!/usr/bin/env bash
# Build SONIC's gear_sonic_deploy at the pinned commit, with realmimic's patches applied, in a
# directory of its own. Never touches an existing SONIC install.
#
#   deploy/sonic/build.sh [WORKDIR]          (default: ~/realmimic-sonic)
#
# Environment:
#   TensorRT_ROOT       TensorRT install root (x86 tarball, or the Jetson's existing TensorRT)
#   onnxruntime_ROOT    ONNX Runtime root (default: /opt/onnxruntime as SONIC expects)
#   CUDAToolkit_ROOT    optional, CUDA toolkit root
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source <(grep -E '^[A-Z_]+=' "$HERE/PIN")
WORKDIR="${1:-$HOME/realmimic-sonic}"
SRC="$WORKDIR/GR00T-WholeBodyControl"
DEPLOY="$SRC/gear_sonic_deploy"

die() { echo "error: $*" >&2; exit 1; }

# ------------------------------------------------------------------ prerequisites
missing=()
for header in /usr/include/zmq.h /usr/include/msgpack.hpp /usr/include/eigen3/Eigen/Core /usr/include/yaml-cpp/yaml.h /usr/include/nlohmann/json.hpp; do
  [[ -e "$header" ]] || missing+=("$header")
done
if ((${#missing[@]})); then
  echo "missing system headers: ${missing[*]}" >&2
  echo "install with: sudo apt-get install -y libzmq3-dev libmsgpack-dev libeigen3-dev libyaml-cpp-dev libgtest-dev nlohmann-json3-dev" >&2
  exit 1
fi
[[ -n "${TensorRT_ROOT:-}" && -e "$TensorRT_ROOT/include/NvInfer.h" ]] || die "set TensorRT_ROOT (NvInfer.h not found)"
export onnxruntime_ROOT="${onnxruntime_ROOT:-/opt/onnxruntime}"
[[ -e "$onnxruntime_ROOT/include/onnxruntime_cxx_api.h" ]] || die "onnxruntime not found at $onnxruntime_ROOT"
command -v just >/dev/null || die "'just' not on PATH"

# ------------------------------------------------------------------ source at the pinned commit
mkdir -p "$WORKDIR"
fresh=0
if [[ ! -d "$SRC/.git" ]]; then
  fresh=1
  git clone --filter=blob:none --no-checkout "$SONIC_REPO" "$SRC"
  # Deploy runtime + the Python package its MuJoCo sim-to-sim loop imports (root files come along).
  git -C "$SRC" sparse-checkout set gear_sonic_deploy gear_sonic install_scripts
fi
git -C "$SRC" fetch --quiet origin "$SONIC_COMMIT"
if ((!fresh)) && [[ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]]; then
  die "$SRC has local changes; refusing to reset it"
fi
git -C "$SRC" checkout --quiet --detach "$SONIC_COMMIT"

# realmimic patches, in order, as commits on a local branch (so `git log` shows them).
git -C "$SRC" checkout --quiet -B realmimic "$SONIC_COMMIT"
shopt -s nullglob
patches=("$HERE"/patches/*.patch)
if ((${#patches[@]})); then
  git -C "$SRC" -c user.name="realmimic build" -c user.email="build@realmimic.invalid" am --quiet "${patches[@]}"
fi
# Vendored binaries (unitree_sdk2 etc.) are Git LFS objects; fetch only those, not model files.
git -C "$SRC" lfs pull --include "gear_sonic_deploy/thirdparty/**"
echo "SONIC at $(git -C "$SRC" rev-parse --short HEAD) (${#patches[@]} realmimic patches on $SONIC_COMMIT)"

# Header-only cppzmq (not packaged on Ubuntu 22.04), pinned.
CPPZMQ="$WORKDIR/cppzmq"
[[ -d "$CPPZMQ" ]] || git clone --quiet --depth 1 --branch "$CPPZMQ_TAG" "$CPPZMQ_REPO" "$CPPZMQ"

# ------------------------------------------------------------------ build (SONIC's own recipe)
export HAS_ROS2=0
export CXXFLAGS="${CXXFLAGS:-} -I$CPPZMQ"
export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$onnxruntime_ROOT/lib:${LD_LIBRARY_PATH:-}"
(cd "$DEPLOY" && just build)
echo "built: $DEPLOY/target/release/g1_deploy_onnx_ref"
