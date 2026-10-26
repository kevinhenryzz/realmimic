# SONIC runtime integration

realmimic runs its policy inside NVIDIA SONIC's deploy runtime (`gear_sonic_deploy`). We do not
fork it: `build.sh` checks out the commit in `PIN` into its own directory, applies `patches/` as
commits on a local `realmimic` branch, and builds with SONIC's own recipe (`just build`).

**Prerequisites** (versions are SONIC's pins):
- **TensorRT 10.7** for x86: the "TensorRT 10.7 GA for Linux x86_64 and CUDA 12.0 to 12.6 TAR Package" from
  [NVIDIA's TensorRT download page](https://developer.nvidia.com/tensorrt) (free NVIDIA developer account); unpack
  it anywhere. On a Jetson, use the TensorRT that JetPack installed (`/usr/include`, `/usr/lib/aarch64-linux-gnu`).
- **ONNX Runtime 1.16.3** (C++): `onnxruntime-linux-x64-1.16.3.tgz` (x86) or `onnxruntime-linux-aarch64-1.16.3.tgz`
  from [its GitHub release](https://github.com/microsoft/onnxruntime/releases/tag/v1.16.3); unpack it.
- **CUDA toolkit 12.x**, `git-lfs`, and [`just`](https://github.com/casey/just) (SONIC's build runner).
- System packages, listed below.

```bash
export TensorRT_ROOT=/path/to/TensorRT-10.7.0.23        # x86 tarball, or the Jetson's TensorRT
export onnxruntime_ROOT=/path/to/onnxruntime-1.16.3     # SONIC's pinned version (default /opt/onnxruntime)
deploy/sonic/build.sh ~/realmimic-sonic
```

System packages (Ubuntu 22.04): `libzmq3-dev libmsgpack-dev libeigen3-dev libyaml-cpp-dev
libgtest-dev nlohmann-json3-dev`. cppzmq (header-only) is fetched at a pinned tag by the script.

`PIN` must match the commit of the runtime installed on the robot, and the robot's runtime must be
built by `build.sh` too, so it carries the same patches.

## Safety patches

SONIC's runtime stops control on its own faults (lost LowState, joint velocity > 35 rad/s, failed
inference) and on the operator's `O` key, by sending a damping command (kp 0, kd 8) and exiting.
The patches add to that stop path; every new check ends in it. Unit tests of the decision logic are
in SONIC's gtest target: `target/release/run_tests --gtest_filter='SafetyChecks.*'` (run from
`gear_sonic_deploy`; SONIC's own FK test needs reference clips that are not checked out).

| patch | change |
|---|---|
| `0001` signal-safe stop | SIGINT / SIGTERM / SIGHUP request the same stop as `O` (the handler only sets a flag; the main loop acts). Every stop now holds the damping command: the control thread is stopped first, then the 500 Hz command writer keeps publishing damping for `--stop-damping-ms` before the final write and exit, so one lost DDS message cannot leave the last policy target standing. |
| `0002` policy guard | Optional check of the decoder's raw output (IsaacLab order, before action scale and default angles) every tick, before the motor command is written: NaN / Inf or \|a\| above the bound stops control and prints the joint index and value. The bound is on the raw action, not the joint target, which may legitimately exceed joint limits with low gains. NaN / Inf are detected from the bit pattern because the runtime builds with `-ffast-math`, under which `std::isnan` is folded to false. Also makes SONIC's LowState-age check configurable. |
| `0003` thermal guard | Optional thresholds on each motor's hotter temperature reading (either of its two sensors; SONIC labels them winding / driver, the Unitree SDK casing (85 C) / winding (120 C)), checked every control tick while waiting for start and in control: a warning printed once per motor, and a trip that stops control naming the motor. SONIC's own 90 °C audio / console warning is unchanged. |

| flag | default | effect |
|---|---|---|
| `--stop-damping-ms <ms>` | 1000 | damping hold on every stop (`O`, signals, safety checks); 0 disables the hold |
| `--enable-policy-guard` | off | policy guard with bound 20 |
| `--policy-action-limit <value>` | off | policy guard with this bound (implies `--enable-policy-guard`) |
| `--lowstate-timeout-ms <ms>` | 500 | stop when the newest LowState is older (SONIC's fixed 500 ms before) |
| `--motor-temp-warn <degC>` | off | print once per motor when it reaches the threshold |
| `--motor-temp-trip <degC>` | off | stop when any motor reaches the threshold |

Without flags the runtime behaves as SONIC's except for the two stop changes of `0001` (signals, damping
hold). A second Ctrl-C does not cut the hold short; Ctrl-\ (SIGQUIT) or `kill -9` still kill the
runtime at once, leaving the last command standing. Not added: a softer "hold idle" thermal level
(the planner's idle is commanded through the movement state that the input thread owns, so the
control loop overriding it would race the operator's input). Joint torques `tau_est` were already
logged by SONIC's StateLogger (`deploy_logs/motor_torque.csv`, hardware order, same rows as
`motor_temperature.csv`).

## Sim-to-sim through the deploy binary

`sim2sim.py` runs the exact deploy binary against a MuJoCo simulation of any RobotSpec, drives it
with a scripted operator session and writes a metrics report. One-time setup, after `build.sh`:

```bash
deploy/sonic/setup_sim.sh ~/realmimic-sonic
```

It creates SONIC's sim venv (`~/realmimic-sonic/.venv_sim`, Python 3.10), copies the pinned
`unitree_sdk2_python`, materialises SONIC's example clips and stock G1 model from Git LFS outside
the checkout, and downloads SONIC's release encoder/decoder and planner into the runtime's
untracked `policy/release/` and `planner/target_vel/V2/` folders. Tracked files are not touched.

```bash
export TensorRT_ROOT=... onnxruntime_ROOT=... CUDAToolkit_ROOT=/usr/local/cuda-12.4
# a policy on a robot spec (the bundle's launch.env flags are passed to the runtime)
uv run python deploy/sonic/sim2sim.py run --robot g1_brainco --bundle exports/<bundle> \
    --session planner --out runs/sim2sim/<name>
# SONIC's release policy on SONIC's stock model / on a robot spec with SONIC's default gains
uv run python deploy/sonic/sim2sim.py run --robot sonic_stock --bundle sonic_release --session planner
uv run python deploy/sonic/sim2sim.py run --robot g1_brainco --bundle sonic_release
# normal-mode clip playback instead of the planner session
uv run python deploy/sonic/sim2sim.py run ... --session clip --clip <clip under SONIC's reference/example>
uv run python deploy/sonic/sim2sim.py report runs/sim2sim/<name>     # recompute the reports
uv run python deploy/sonic/sim2sim.py summarize runs/sim2sim/*/      # markdown table
```

**Processes.** `sonic_sim.py` runs in SONIC's sim venv and reuses SONIC's own simulator code
(`DefaultEnv`: scene loading, motor→joint mapping, PD torque from `rt/lowcmd`, `rt/lowstate`
packing; `UnitreeSdk2Bridge`; `ElasticBand`). It replaces only `BaseSimulator`'s loop: headless,
real time, physics + PD at `--sim-dt` (1 ms), LowState at `--lowstate-dt` (2 ms, the robot's
500 Hz), robot spawned standing in `default_angles` and held by the band until the session drops
it. The runtime runs with `--input-type keyboard`; the session types keys on its stdin (the
keyboard handler reads stdin non-blocking, so a pipe works exactly like a terminal). DDS is on
`lo` with `--disable-crc-check` (the Python bridge does not compute CRCs).

**Robot model.** `realmimic.robots.sim2sim` writes the training entity (spec inertials, hands,
payloads, armature, collisions) with one `<motor>` per joint in hardware order and the training
effort limits as `ctrlrange`; `tests/realmimic/test_sim2sim_scene.py` pins the order SONIC's
`DefaultEnv` relies on. `--robot sonic_stock` uses SONIC's `scene_43dof.xml` unchanged.

**Sessions** (`--idle` sets the idle phases):
- `planner`: `]` start → Enter (planner) → drop band → idle → `1000` slow walk at 0.5 m/s → `w` 6 s
  → `r` stop → `s` 6 s → stop → `q` ×3 then `e` ×3, 1.5 s apart (turn in place ≈ +90° and back;
  presses sent in one burst only turn the planner's reference ≈ 0.45 rad) → `n`, `---` squat at
  0.5 m → `2` kneel on two legs → `1`, `===`, `p` rise → idle.
- `clip`: `]` start → drop band → `t` play the clip (normal mode) → idle.

A fall (pelvis < 0.35 m or tilt > 1 rad after the band drop) ends the session; `O` is always sent
at the end (damping held for `--stop-damping-ms`, then exit). Safety flags go in `--extra-flags`.

**Robustness** (`--perturb PROFILE`, `--seed N`): the simulator can degrade itself towards the real
robot, each profile one family of imperfections, `all` every one at once. Magnitudes are moderate
estimates of a G1's imperfections, not measurements (the 5 degree ankle-roll offset is a stress value):

| profile | what changes |
|---|---|
| `latency` | LowState published 5 ms old, LowCmd applied 5 ms late |
| `delay10`, `delay20` | LowCmd applied 10 / 20 ms late: pass/fail tests before hardware (not in `all`) |
| `imu` | 2 degree IMU mounting tilt, gyro bias up to 0.02 rad/s per axis, gyro noise 0.05 rad/s |
| `encoders` | joint calibration offsets up to 0.02 rad, left ankle roll 0.087 rad, joint velocity noise 0.05 rad/s |
| `weak_motors` | effort limits at 85 % |
| `payload` | 0.5 kg more on `torso_link` than modelled |
| `low_friction` | floor and feet sliding friction 0.5 |

The sensor errors enter the LowState the runtime reads, so the observation-log check still holds; the
draws and profile are stored in `sim.npz` and `session.json`, and `summarize` prints a `perturb` column.

**Outputs** in the run directory: `sim.npz` (sim state at 200 Hz: base pose/velocity, q, dq,
applied torque, LowCmd q/kp/kd, contacts), `deploy_logs/` (the runtime's StateLogger CSVs),
`target_motion.csv`, `policy_input.csv`, `session.json`, `deploy.log`, `sim.log`, and
- `metrics.json` (`realmimic.eval.sim2sim`): fall, per-phase pelvis height / tilt / speed / yaw,
  per-joint mean |τ|, RMS, p95 and time above 50 % of the effort limit for ankles, knees, hip
  pitch and waist (thermal proxy), joint / root-height / tilt tracking error against the logged
  reference, the gains the sim actually received, control-tick and runtime timing, sim lag;
- `policy_check.json` (our bundles): the exported ONNX rerun on every logged input against the action
  the runtime's TensorRT engine produced (next tick's `last_action`), so a wrong engine build or
  TensorRT version is caught, here and on the robot;
- `obs_check.json` (`realmimic.eval.sonic_obs_check`): the runtime's logged policy input against
  `realmimic.obs.sonic` recomputed from the runtime's own logged state (history terms in full,
  reference terms on frame 0; relative tolerance 1e-5 for the log's 6 significant digits);
- `release_check.json` (`--bundle sonic_release --session clip` only,
  `sonic_obs_check.release_policy_check`): SONIC's release policy rerun in torch
  (`realmimic.eval.sonic_release_policy`) on the logged states and clip: reconstructed frames vs the
  target-motion log, token vs the logged `token_state`, action vs the next tick's `last_action`.

**GPU.** The runtime builds TensorRT engines on first use (planner: 774 MB ONNX). `run` refuses to
start below `--gpu-min-free-mib` (2500) and kills the runtime if free memory drops below
`--gpu-abort-free-mib` (300). Policy engines are cached in `runs/sim2sim/.engines/<sha>/`, so
bundle directories are only read.

**Where the sim differs from the robot's low-level control** (all runs):
- PD: the motor drivers close `tau + kp (q* - q) + kd (dq* - dq)` at kHz rates; the sim
  recomputes it every `--sim-dt` from the latest LowCmd (SONIC's own loop: 200 Hz), clipped at a
  constant effort limit. No torque-speed curve, current limit, or thermal derating.
- Ankles and waist roll/pitch are parallel linkages on the robot (firmware maps joint-space PD to
  two motors in `mode_pr` PR); the sim actuates the joints directly.
- `motor_cmd.mode`, `mode_pr`, `mode_machine` and CRC are ignored; LowState temperatures and motor
  error flags are 0; IMU = exact pelvis pose/velocity (no noise, bias or latency); the
  accelerometer field carries `qacc` (no gravity) instead of specific force.
- LowState / LowCmd travel over `lo` with negligible latency. (The runtime's printed
  `LowState age` is taken after observation + inference, ≈ 9 ms later in the tick.)
- Start-up: the robot spawns standing in `default_angles`, held at that height by SONIC's elastic
  band, which is dropped 2 s after the planner starts (SONIC: hangs at 1 m, dropped by hand).
- Training model vs SONIC's stock model: the stock model clips hip roll at 88 Nm (robot and
  mjlab: 139), uses armature 0.01 everywhere, joint damping 0.05 and friction loss 0.2, the Euler
  integrator, mesh collisions and Dex3 hands; our scenes use mjlab's reflected inertias, no
  passive damping/friction, `implicitfast`, capsule collisions, welded hands.
