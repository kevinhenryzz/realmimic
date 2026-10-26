# realmimic: SuperMimic on one RTX 4090

**A whole-body motion tracker for the Unitree G1, trained on one RTX 4090 (24 GB) in 10.5 hours, and run through
NVIDIA's SONIC deploy runtime.**

realmimic reproduces SuperMimic ([Liao et al., 2026](https://arxiv.org/abs/2610.28480)): a humanoid motion tracker
trained through a learned, differentiable world model. We set out to train the full recipe on a single consumer GPU
and to measure it on the paper's own benchmark:
- **Training on one RTX 4090.** `uv run sm-train release` reproduces the released model.
- **The paper's benchmark** on all 129,491 motion clips, with NVIDIA SONIC's released policy scored in the same
  evaluator.
- **A path to the robot.** Your G1 as a config file (hands, payloads), deployed inside NVIDIA's SONIC runtime and
  planner, unforked, with safety patches, and tested in simulation against the real deploy program.

**Status.** v1.0: the reproduction, and deployment validated in simulation. Coming in v1.1: deploy weights and results
on a real Unitree G1. Released weights (CC BY-NC 4.0): [huggingface.co/kevinhenryz/realmimic](https://huggingface.co/kevinhenryz/realmimic).

**Get started:** [Quick start](#quick-start) · [Data](#data) · [Train and evaluate](#train-and-evaluate) · [Deploy](#deploy-sonic-runtime)

> Independent and unofficial; not affiliated with the SuperMimic authors, UC Berkeley, NVIDIA, Bones Studio or Unitree.

## Results

We use the paper's benchmark (its Appendix F): every clip of the BONES-SEED dataset that passes the paper's filter,
129,491 clips, each played once from its first frame. A clip fails if the robot falls: pelvis height off by more than
0.25 m or its tilt by more than 0.8 from the reference. Our policy and SONIC's released policy run in the same
evaluator (MuJoCo Warp, mjlab), each with its own motor gains. The paper's figures come from its own evaluator and are
quoted as reported. The tracking score is the average normalized tracking reward per step (1 = perfect).

| policy | training | tracking score | failed (of 129,491) |
|---|---|---|---|
| **realmimic, released model** | **one RTX 4090, 10.5 h, 86k iterations** | **0.9545** | **10** |
| realmimic, last three checkpoints (80k, 84k, 86k) | | 0.9536 – 0.9545 | 9, 8, 10 |
| realmimic, same recipe, second seed (not released) | one RTX 6000 Ada, 86k iterations | 0.9540 | 4 (last three 4, 5, 4) |
| SONIC release, in our evaluator | (NVIDIA) | 0.875 | 218 |
| SuperMimic, as reported (the paper's evaluator scores SONIC at 179) | one H100 80 GB, 3 h | (paper) | 25 |

None of these failures happens on a clip's last frame. Across the two seeds the final checkpoint fails 4 to 10 clips.
Re-running one checkpoint with a different number of parallel simulations moves the failure count by about ±10 (at
~200 failures), so differences under 20 are within measurement noise.

**Failures by motion type** (the paper's Table A.8 groups, recovered from motion-category names; group sizes are ours):

| motion type | clips, ours (paper) | SONIC, paper | SuperMimic 3 h, paper | SONIC release, ours | realmimic |
|---|---|---|---|---|---|
| side rolls | 22 (22) | 22 | 2 | 22 | 0 |
| flips (360 / 180 / 90) | 54 (44) | 44 | 1 | 54 | 0 |
| fire-in-the-hole turns | 20 (20) | 3 | 3 | 1 | 0 |
| crossed legs | 10 (10) | 9 | 0 | 9 | 2 |
| lying, resting and standing up | 102 (99) | 28 | 8 | 32 | 3 |
| broad jumps | 24 (24) | 15 | 0 | 16 | 0 |
| dance routine | 8 (8) | 8 | 0 | 8 | 0 |
| everything else | 129,251 (129,287) | 50 | 11 | 76 | 5 |
| total | 129,491 (129,514) | 179 | 25 | 218 | 10 |

**Motion it never saw (LAFAN1).** Whole clips (40, each 1 to 4.5 minutes): 13 and 14 failed over the two seeds,
tracking score 0.664 and 0.667 (SONIC release: 13, 0.600). The same clips cut into 882 ten-second segments: 35 and 51 failed
(4.0 and 5.8 %), tracking score 0.828 and 0.821 (SONIC release: 52 = 5.9 %, 0.735). Tracking is closer in both seeds, and the
failure count is in SONIC's range. Over whole clips the policy drifts in world position (its goal has no
root-position term), so a single slip over minutes fails a clip; the segments measure tracking itself.

## What we found

Two problems in the open reimplementation we started from decide the result at full data scale; the paper specifies
neither.

1. **Training rarely practised the hardest motions.** Training starts episodes in 1-second pieces of clips, preferring
   pieces it tracks badly. The inherited estimate of "badly" started every piece at the best possible score and moved
   5 % per visit, so with ~1 million pieces the rare acrobatic ones kept looking easy, while every clip's last second
   looked hard. Flips and side rolls got *half* their fair share of practice.
   - **Fix:** a running mean that counts visits and assumes the worst for unvisited pieces, with a continuation value
     at clip ends (`env.bin_estimator="count"`, the default).
   - **Effect:** on the 130 hardest clips, all 54 flips and 22 side rolls are learned by 6k iterations; with the old
     estimate, 5 to 12 flips still failed at 12k.
2. **Nothing in the training objective covered the end of a clip.** The learned-model rollouts stopped 32 frames
   before the end, so long runs learned to fall on the final frame. That costs almost nothing in the score but counts
   as a failure: one such run had 2,882 failures at 40k iterations, 2,802 of them on the last frame, and still more
   later (4,417 on the last frame at 48k).
   - **Fix:** past the end, the reference holds the last frame still and the rollouts run into it
     (`env.freeze_clip_end`, the default). The released model has no last-frame failures.

**What did not matter measurably.**
- **World-model coordinate frame.** The heading frame we inherited is singular when the pelvis points straight up or
  down, mid-flip; we use body-frame features instead (`world_model.state_spec.frame="body"`, the default). A full run
  with the heading frame reaches the same result (9 / 11 / 8 failures at 80k / 84k / 86k, tracking score 0.9540), within the
  spread between seeds: at this scale, the flips were decided by the sampling fix above.
- **An alternative shape for the tracking reward** (a linear tail on its kernels, `env.kernel_tail`): 11 failures,
  tracking score 0.9528, against 10 and 0.9545 with the paper's shape, which the released model uses.

## Deployment, in simulation

Before a policy touches a robot, we run it inside NVIDIA SONIC's actual deploy program, the binary that runs on the
G1, connected to a MuJoCo simulation of a G1 with BrainCo hands. Each session walks, turns, squats, kneels and stands up
through SONIC's planner, with pushes, sensor errors and 10 to 20 ms command delays (`deploy/sonic/sim2sim.py`). Our
released tracker is first fine-tuned for this robot. Means over five sessions:

| policy | falls | joint tracking error (rad) | body tilt error (rad) | ankle motor load (Nm) | waist motor load (Nm) |
|---|---|---|---|---|---|
| SONIC's released policy | 0 | 0.083 | 0.052 | 4.48 | 3.31 |
| realmimic, fine-tuned for the robot | 0 | 0.075 | 0.008 | 3.32 | 2.86 |

A second training seed gives no falls, 0.069 rad, 0.008 rad, 4.23 Nm and 3.34 Nm. Without the fine-tune, the
released tracker falls once under command delay on this robot model. The fine-tune recipe and deploy weights come with
v1.1, together with results on a real G1.

## Compute

The released model trains on one RTX 4090 in 10.5 hours. Developing and checking it took more; measured GPU-hours
(wall-clock hours × GPUs, from the training and evaluation logs):

| used for | hardware | GPU-hours |
|---|---|---|
| released model | one RTX 4090 | 10.5 |
| second full-scale run (same recipe, alternative reward shape; not released) | one RTX 4090 | 11.0 |
| other RTX 4090 runs: two early full-scale attempts, five robot fine-tunes | RTX 4090 | 12.4 |
| experiments that chose the recipe | RTX 6000 Ada 48.4, RTX 5000 Ada 4.9 | 53.3 |
| further robot fine-tunes, each with a second seed | RTX 6000 Ada | 23.1 |
| second seed (11.8) and a run with the heading frame (11.9) | one RTX 6000 Ada each | 23.7 |
| evaluations: full set and LAFAN1, including repeats | RTX 6000 Ada 38.4, RTX 5000 Ada 9.3 | 47.7 |
| data and environment setup | RTX 6000 Ada, RTX 5000 Ada | 1.7 |
| total | | ≈ 183 |

Not included: benchmarks run alongside training on the same RTX 4090 (not metered separately) and the simulation
deployment sessions (mostly CPU).

## Differences from the paper

| item | paper | here | why |
|---|---|---|---|
| policy batch | 4096 imagined rollouts | 2048 | fits the 4090's 24 GB next to the motion library |
| schedule | (paper Table A.5) | 86k iterations, learning-rate decay over the first 69k | one 4090 in 10.5 h |
| clips | 129,514 | 129,491 | the same filter on the current BONES-SEED release |
| practice sampling | unspecified | count-aware estimate, pessimistic for unvisited pieces | see *What we found* |
| clip ends | unspecified | last frame held, rollouts reach it | see *What we found* |
| world-model frame | unspecified | body frame | no singularity; no measured effect |

**Motor gains.** Training uses mjlab's default G1 model (hip pitch in the 7520-14 group, kp ≈ 40), the low-gain setup
the paper builds on; SONIC's runtime uses kp 99 on the same joints. The benchmark scores each policy under its own
gains, as in the paper's Appendix F.1.

## Quick start

**You need:** Linux, an NVIDIA GPU with 24 GB (training; evaluation works on smaller GPUs), NVIDIA driver 580 or newer
(older drivers: see `scripts/env_cuda12_driver.sh`), about 170 GB of free disk for the data,
[uv](https://docs.astral.sh/uv/getting-started/installation/), and a Hugging Face account.

```bash
git clone https://github.com/kevinhenryzz/realmimic && cd realmimic
uv sync --group dev                 # Python 3.12 and every dependency, pinned by uv.lock
uv run hf auth login                # once; a read token from https://huggingface.co/settings/tokens
tools/download_data.sh              # BONES-SEED (accept its licence on Hugging Face first) and LAFAN1
tools/prepare_data.sh               # convert, filter, build the training library and the benchmark lists
uv run sm-train release             # train the released model's recipe (one RTX 4090, about 10.5 h)
```

Then evaluate it as below. `uv run pytest -q tests` runs the test suite (GPU tests need CUDA and the converted LAFAN1
clips).

## Data

This repository contains no BONES-SEED motion data or metadata. Each user downloads it under their own licence:
- **BONES-SEED** (training and benchmark): open [its Hugging Face page](https://huggingface.co/datasets/bones-studio/seed)
  while logged in and accept the BONES Motion Capture Dataset License Agreement, then run `tools/download_data.sh`.
  The script stops with instructions if access has not been granted yet.
- **LAFAN1** (held-out evaluation, never used for training) is downloaded by the same script; skip it with `--no-lafan1`.

`tools/prepare_data.sh` converts the clips to 50 Hz, applies the paper's Appendix E filter, builds the training library
(`data/library/bones_paper.pt`, from BONES-SEED alone) and writes the benchmark lists (`data/bench/`). Motion groups are
regular expressions over motion-category tokens, equivalent to the groups of the paper's Table A.8. The script prints
the lists' md5 sums; compare them with ours:

| list | md5 |
|---|---|
| `manifest_filtered.csv` | `5bc2aae2efc6f986ab4cf7adbb4d8ffc` |
| `full.txt` | `e588d04ab59d948b86347088dff357f0` |
| `rand2000.txt` | `28dfbc14f02146cc05fe62c794d89f01` |
| `hard.txt` | `b931bb29b8c786bfdfa34b57c0832caf` |

## Train and evaluate

```bash
# the released model's recipe (one RTX 4090, ~10.5 h); any flag after the preset overrides it
uv run sm-train release --run-name release
# the Appendix F benchmark on a checkpoint (the full set in chunks)
uv run python scripts/eval_benchmark.py --ckpt logs/release/ckpt_latest.pt --npz-dir data/npz/bones_seed \
    --clip-list data/bench/full.txt --num-envs 4096 --chunk-size 8192 --out runs/benchmark/release_full.json
# failures per motion group, next to the paper's figures
uv run python tools/group_table.py "realmimic=runs/benchmark/release_full.json"
# held-out LAFAN1: whole clips, then the same clips in 10-second segments
uv run python scripts/eval_benchmark.py --ckpt logs/release/ckpt_latest.pt --npz-dir data/npz/lafan1 \
    --num-envs 512 --out runs/benchmark/release_lafan1.json
uv run python tools/segment_clips.py data/npz/lafan1 data/npz/lafan1_seg10 --seconds 10
uv run python scripts/eval_benchmark.py --ckpt logs/release/ckpt_latest.pt --npz-dir data/npz/lafan1_seg10 \
    --num-envs 1024 --out runs/benchmark/release_lafan1_seg10.json
# SONIC's released policy in the same evaluator (downloaded from NVIDIA, under its own licence)
tools/download_sonic_release.sh
uv run python scripts/eval_benchmark.py --sonic-release third_party/sonic_release --robot g1 \
    --npz-dir data/npz/bones_seed --clip-list data/bench/full.txt --num-envs 4096 --chunk-size 8192 \
    --out runs/benchmark/sonic_release_full.json
```

A full-set evaluation takes about half an hour on one RTX 6000 Ada; `--chunk-size 8192` keeps it within 32 GB.

## Deploy (SONIC runtime)

See [`deploy/sonic/README.md`](deploy/sonic/README.md). In short:
1. `deploy/sonic/build.sh` builds NVIDIA's runtime at a pinned commit, with our safety patches: a signal-safe stop
   with a damping hold, a policy-output guard, a LowState timeout and a motor-temperature trip.
2. `tools/export_sonic_bundle.py` exports a checkpoint as a SONIC decoder bundle for your robot spec.
3. `deploy/sonic/sim2sim.py` runs that exact runtime against MuJoCo, with planner sessions, perturbations and delay
   tests, before anything touches hardware.

The deploy fine-tune recipe (robot model, hardware-safe motion mix, load and posture terms) and its weights come with
v1.1.

The observation layout the runtime feeds the policy is documented in
[`docs/sonic-observation-contract.md`](docs/sonic-observation-contract.md).

## Browser demo

`site/` is a static page that runs the released policy live in the browser: MuJoCo compiled to WebAssembly at the
training physics settings, the policy in ONNX Runtime Web, drag the robot to push it. See
[`site/README.md`](site/README.md).

## Safety

This software commands a humanoid robot. Run it with the robot on a gantry first. Keep the runtime's stop path
(keyboard `O`, Ctrl-C) local to the robot, never across a relayed network. Never run it unattended. Under the
BONES-SEED licence, policies trained on its data may not be used in hazardous environments requiring fail-safe
performance.

## Credits and licences

- **Training core:** derived from [YanYuan98/Supermimic_mod](https://github.com/YanYuan98/Supermimic_mod), itself a
  fork of [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced), MIT licence
  (`LICENSES/`, `NOTICE`).
- **Method:** SuperMimic, Q. Liao et al., CoRL 2026, arXiv:2610.28480.
- **Runtime:** NVIDIA GR00T-WholeBodyControl / SONIC (Apache-2.0), used unforked through patches. The release policy
  and planner weights are under the NVIDIA Open Model License and are downloaded by script.
- **Data:** Training data includes Motion Data by Bones Studio ([https://bones.studio/](https://bones.studio/)). Use
  of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement. LAFAN1
  (CC BY-NC-ND 4.0) is used for evaluation, never for training, and is never redistributed.
- **Demo motions:** CMU Graphics Lab Motion Capture Database and 100STYLE (CC BY 4.0); see `site/` for per-clip
  credits.
- **Code:** Apache-2.0 (`LICENSE`, `NOTICE`). **Released weights:** CC BY-NC 4.0, in line with the non-commercial
  research scope of the BONES-SEED licence they were trained under.
