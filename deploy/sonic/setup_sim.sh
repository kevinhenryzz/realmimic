#!/usr/bin/env bash
# Prepare SONIC's MuJoCo sim-to-sim environment next to a build.sh checkout. Never modifies the
# checkout's tracked files.
#
#   deploy/sonic/setup_sim.sh [WORKDIR]          (default: ~/realmimic-sonic)
#
# Creates, outside the checkout:
#   WORKDIR/.venv_sim             Python 3.10 venv with the packages SONIC's install_mujoco_sim.sh
#                                 installs for run_sim_loop.py (gear_sonic itself is used from the
#                                 checkout via PYTHONPATH, so torch is not pulled in)
#   WORKDIR/unitree_sdk2_python   SONIC's vendored Unitree SDK2 Python at the pinned commit
#   WORKDIR/reference/example     SONIC's example reference clips   (Git LFS objects, materialised
#   WORKDIR/sonic_g1_model        SONIC's stock G1 MJCF + meshes     here, not in the checkout)
# and, as untracked downloads inside the checkout (where the runtime expects them):
#   gear_sonic_deploy/policy/release/model_{encoder,decoder}.onnx   (HF nvidia/GEAR-SONIC)
#   gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx      (HF nvidia/GEAR-SONIC)
# HF's observation_config.yaml goes to WORKDIR/hf_release/ and is compared with the tracked one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source <(grep -E '^[A-Z_]+=' "$HERE/PIN")
WORKDIR="${1:-$HOME/realmimic-sonic}"
SRC="$WORKDIR/GR00T-WholeBodyControl"
DEPLOY="$SRC/gear_sonic_deploy"
VENV="$WORKDIR/.venv_sim"

die() { echo "error: $*" >&2; exit 1; }
[[ -d "$SRC/.git" ]] || die "$SRC is not a SONIC checkout; run deploy/sonic/build.sh first"
head="$(git -C "$SRC" rev-parse "$SONIC_COMMIT^{commit}")"
[[ "$(git -C "$SRC" merge-base HEAD "$head")" == "$head" ]] || die "$SRC is not based on $SONIC_COMMIT"

# ------------------------------------------------------------------ Unitree SDK2 Python (pinned)
SDK="$WORKDIR/unitree_sdk2_python"
if [[ ! -d "$SDK" ]]; then
  mkdir -p "$SDK"
  git -C "$SRC" archive "$SONIC_COMMIT" external_dependencies/unitree_sdk2_python \
    | tar -x -C "$SDK" --strip-components=2
fi

# ------------------------------------------------------------------ venv (SONIC: Python 3.10)
if [[ ! -x "$VENV/bin/python" ]]; then
  uv python install 3.10
  uv venv "$VENV" --python 3.10 --prompt gear_sonic_sim
fi
# gear_sonic[sim] + its base dependencies minus torch (not imported by the sim loop); mujoco pinned
# to realmimic's version so the MJCF we generate loads identically.
uv pip install --python "$VENV/bin/python" \
  "numpy==1.26.4" "scipy==1.15.3" joblib tqdm easydict loguru \
  "mujoco==3.16.0" tyro pin pyyaml pyzmq msgpack msgpack-numpy opencv-python \
  "cyclonedds==0.10.2" huggingface_hub -e "$SDK"

# ------------------------------------------------------------------ LFS trees, outside the checkout
# materialise TREE OUT: write the pinned commit's TREE to OUT, resolving Git LFS pointers.
materialise() {
  local tree="$1" out="$2" tmp="$2.partial" path dst
  [[ -d "$out" ]] && return
  rm -rf "$tmp"
  git -C "$SRC" ls-tree -r --name-only "$SONIC_COMMIT" "$tree" | while read -r path; do
    dst="$tmp/${path#"$tree"/}"
    mkdir -p "$(dirname "$dst")"
    git -C "$SRC" show "$SONIC_COMMIT:$path" >"$dst"
    if head -c 40 "$dst" | grep -q '^version https://git-lfs'; then
      git -C "$SRC" lfs smudge -- "$path" <"$dst" >"$dst.lfs" && mv "$dst.lfs" "$dst"
    fi
  done
  mv "$tmp" "$out"
}
materialise gear_sonic_deploy/reference/example "$WORKDIR/reference/example"
materialise gear_sonic/data/robot_model/model_data/g1 "$WORKDIR/sonic_g1_model"

# ------------------------------------------------------------------ SONIC release weights + planner
"$VENV/bin/python" - "$DEPLOY" "$WORKDIR/hf_release" <<'EOF'
import filecmp, shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

deploy, hf_dir = Path(sys.argv[1]), Path(sys.argv[2])
files = {
    "model_encoder.onnx": deploy / "policy/release/model_encoder.onnx",
    "model_decoder.onnx": deploy / "policy/release/model_decoder.onnx",
    "planner_sonic.onnx": deploy / "planner/target_vel/V2/planner_sonic.onnx",
    "observation_config.yaml": hf_dir / "observation_config.yaml",
}
for name, dst in files.items():
    if dst.exists():
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(hf_hub_download("nvidia/GEAR-SONIC", name), dst)
    print(f"downloaded {name} -> {dst}")
tracked = deploy / "policy/release/observation_config.yaml"
same = filecmp.cmp(tracked, files["observation_config.yaml"], shallow=False)
print(f"HF observation_config.yaml {'==' if same else '!='} tracked {tracked}")
EOF
echo "sim venv: $VENV"
