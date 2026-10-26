# SONIC observation contract

What the SONIC deploy runtime feeds the policy, term by term, and how this repository reproduces it.

The policy we export replaces SONIC's decoder. The SONIC C++ runtime builds the policy input vector
itself, so our policy has to consume exactly that vector. This page pins down what the runtime
computes for each term we plan to use. Every claim is taken from the SONIC source at the commit
running on the robot.

The torch reference implementation is `src/realmimic/obs/sonic.py`. Its tests are in
`tests/realmimic/test_sonic_obs.py`. `tools/check_sonic_constants.py <gear_sonic_deploy>` compares
the runtime constants and term dimensions of any SONIC checkout against it.

## 0. Source and citation keys

NVlabs/GR00T-WholeBodyControl @ `7f151314d4d1606544bf249d2a7a1cb754c64582`. Paths are relative to
`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/` unless noted.

| Key | File |
|---|---|
| `cpp` | `src/g1_deploy_onnx_ref.cpp` |
| `params` | `include/policy_parameters.hpp` |
| `logger.hpp` / `logger.cpp` | `include/state_logger.hpp` / `src/state_logger.cpp` |
| `math` | `include/math_utils.hpp` |
| `obscfg` | `include/observation_config.hpp` |
| `policy` | `include/control_policy.hpp` |
| `robot` | `include/robot_parameters.hpp` |
| `kbd` | `include/input_interface/keyboard_handler.hpp` |
| `planner` | `include/localmotion_kplanner.hpp` |
| `gains` | `src/motor_gain_scaling.cpp` |
| `ref.md` / `obs.md` | `docs/source/references/motion_reference.md` / `observation_config.md` (repo root) |
| `bridge.py` / `sim.py` | `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py` / `base_sim.py` (repo root) |

## 1. One control tick (50 Hz)

The program runs through three states: INIT (3 s ramp to `default_angles`), WAIT_FOR_CONTROL and
CONTROL (cpp:3834-3871). Each CONTROL tick runs these steps in order (cpp:3873-4114):

1. `GatherRobotStateToLogger` (cpp:3885 → 2842-2978) reads the latest `rt/lowstate`. It appends one
   StateLogger entry:
   - `base_quat` and `base_ang_vel`: the **pelvis IMU** (`ls->imu_state()`, cpp:2928-2929).
   - `body_q`, `body_dq`: joint state (cpp:2862-2864).
   - `last_action`: the action computed in the *previous* tick (cpp:2965).

   The torso IMU (`rt/secondary_imu`) is logged separately (cpp:2932-2934). No in-scope term reads it.
2. `UpdateHeadingState()` (cpp:3956 → 559-585) handles heading re-initialisation; see §3.9.
3. `GatherObservations()` (cpp:3960 → 2002-2017) zero-fills the buffer. It then runs every enabled
   term's gatherer at its offset.
4. `CreatePolicyCommand()` (cpp:3979 → 3137-3167) runs inference, stores the raw output as
   `last_action`, and writes the motor command (§6).
5. With `--policy-input-logfile`, the double observation buffer is written as one CSV row (cpp:4059-4067).
6. `CurrentFrameAdvancement()` (cpp:4069 → 3184-3436) advances the reference frame by 1 while playing.

So at tick *t* the history terms end with the state measured at *t*, and `last_actions` ends with
a<sub>t−1</sub>. The reference terms start at the frame that is current at *t*. That frame advances
after the action is computed.

## 2. Shared mechanics

### 2.1 History buffer (StateLogger)
- **One ring, written only in CONTROL.**
  - It is created once with capacity 10000, `dt = control_dt_ = 0.02` (cpp:2186, 2475-2476).
  - It is appended only in CONTROL (cpp:3885).
  - It is never cleared. Motion switches, planner toggles and heading re-inits all leave it alone.
    The program cannot return to WAIT_FOR_CONTROL.
- **Sampling.** `GatherHis*(…, num_frames, step)` asks for `GetLatest(num_frames, control_dt·step,
  newest_first=false)` (e.g. cpp:1457-1458).
  - `step` is a whole multiple of `dt`, so the runtime uses the stride path (logger.cpp:213-234).
    It takes entries newest, newest−step, ….
  - **step1 = 10 consecutive ticks.**
- **Frame order: oldest → newest.** The stride path collects newest-first, then reverses because
  `newest_first=false` (logger.cpp:232). The current tick is the **last** frame. Within a term, each
  frame's values are contiguous: `offset + f·dim` (e.g. cpp:1480).
- **Start-up padding.** If fewer entries exist than requested, all-zero entries are appended after
  the oldest real entry and then end up at the **front** after the reverse (logger.cpp:227-232).
  This affects the first 9 CONTROL ticks after `start`.
  - Zero entries (`makeZeroEntry_`, logger.cpp:474-505) have zero `base_quat` (0,0,0,0),
    `base_ang_vel`, `body_q`, `body_dq` and `last_action`.
  - Consequences per term: padded joint-position frames read 0, which is the *default pose*, because
    `body_q` is stored relative to it. Padded gravity frames read **(0, 0, +1)**: §3.5 applied to the
    zero quaternion gives `−v`.

### 2.2 Reference window
- **Source.** A gatherer reads `current_motion_` at `current_frame_`, under the motion mutex (cpp:3944-3966). The motion is either:
  - a CSV clip: `joint_pos.csv`, `joint_vel.csv`, `body_pos.csv`, `body_quat.csv`, all in IsaacLab
    joint order, quaternions wxyz, body 0 = pelvis (ref.md:45-93, 290-292);
  - or the planner's rollout (planner:456-516).
- **Frame index.** Frame *k* of an N-frame window is `current_frame_ + k·step`, clamped to
  `timesteps − 1` (cpp:440-447, 650-657, 763-770). It is the current frame for every *k* while
  `operator_state.play` is false. Joint velocities clamp the same way, but are written as **zeros**
  while paused (cpp:842-847, 875-881).
- **When playback starts.** A newly selected clip starts at frame 0, paused (cpp:2308-2311,
  kbd:381-406, 423-431). A finished reference clip stops, resets to frame 0 paused, and re-inits the
  heading (cpp:3413-3427). The planner instead holds its last frame (cpp:3338-3339).
- **Upper-body override.** When ZMQ/ROS2 streams upper-body targets, the 17 upper-body entries of
  the full-29 joint terms are replaced by the streamed values (cpp:784-797, 861-874). This does not
  happen with keyboard, gamepad or planner input.

## 3. Terms

All dimensions are from the registry (cpp:1708-1815). In the table below, "Ours" names the function
in `realmimic.obs.sonic`.

| Term | Dim | Gatherer (cpp) | Ours |
|---|---|---|---|
| `his_base_angular_velocity_10frame_step1` | 30 | 1813 → 1592-1611 | `his_base_angular_velocity` |
| `his_body_joint_positions_10frame_step1` | 290 | 1810 → 1454-1500 | `his_body_joint_positions` |
| `his_body_joint_velocities_10frame_step1` | 290 | 1811 → 1502-1545 | `his_body_joint_velocities` |
| `his_last_actions_10frame_step1` | 290 | 1812 → 1547-1590 | `his_last_actions` |
| `his_gravity_dir_10frame_step1` | 30 | 1814 → 1613-1634 | `his_gravity_dir` |
| `motion_joint_positions_10frame_step1` | 290 | 1751 → 740-816 | `motion_joint_positions` |
| `motion_joint_velocities_10frame_step1` | 290 | 1752 → 819-907 | `motion_joint_velocities` |
| `motion_root_z_position_10frame_step1` | 10 | 1735 → 427-453 | `motion_root_z_position` |
| `motion_anchor_orientation_10frame_step1` | 60 | 1738 → 616-695 (mode 0) | `motion_anchor_orientation(mode="base")` |
| `motion_anchor_orientation` | 6 | 1732 → same, 1 frame | same, F = 1 |
| `motion_anchor_orientation_heading[_10frame_step{1,5}]` | 6 / 60 | 1741-1743 (mode 1) | `mode="heading"` |
| `motion_anchor_orientation_refheading[_10frame_step{1,5}]` | 6 / 60 | 1746-1748 (mode 2) | `mode="refheading"` |

### 3.1 `his_base_angular_velocity_10frame_step1`
- **Source.** Pelvis IMU gyroscope from `rt/lowstate` (`ls->imu_state().gyroscope()`, cpp:2929). In
  SONIC's MuJoCo sim this is the floating-base `qvel[3:6]` (bridge.py:192, sim.py:352), the pelvis
  angular velocity in the pelvis frame.
- **Units.** rad/s, unscaled (cpp:1606-1608).
- **Layout.** `[ω_{t−9} xyz, …, ω_t xyz]`. Padding frames are 0.

### 3.2 `his_body_joint_positions_10frame_step1`
- **Source.** Motor encoders, stored as `body_q[i] = motor[mujoco_to_isaaclab[i]].q() −
  default_angles[mujoco_to_isaaclab[i]]` (cpp:2862-2863).
- **Order and offset.** IsaacLab order, **relative to `default_angles`**. Hardware motor order is
  MuJoCo order (params:9-12, cpp:2890-2899).
- **Units.** rad.
- **Layout.** 10 × 29, oldest first. Padding frames are 0, i.e. the default pose.

### 3.3 `his_body_joint_velocities_10frame_step1`
- **Source.** `body_dq[i] = motor[mujoco_to_isaaclab[i]].dq()` (cpp:2864): IsaacLab order, rad/s,
  no offset.
- **Safety stop.** Any |dq| > 35 stops control on the robot (cpp:2865-2869, skipped in sim).
- **Layout.** 10 × 29, oldest first. Padding frames are 0.

### 3.4 `his_last_actions_10frame_step1`
- **Value.** `last_action[i] = action_onnx[i]` (cpp:3157): the **raw network output**, in IsaacLab
  order, before scale and offset.
- **Timing.** It is logged at the start of the next tick (cpp:2965), so the frame for tick *t* holds
  a<sub>t−1</sub>.
- **Initial value.** All zeros (cpp:2196). It is never reset.
- **Layout.** 10 × 29, oldest first. Padding frames are 0.

### 3.5 `his_gravity_dir_10frame_step1`
- **Formula.** `quat_rotate_d(conj(base_quat), (0,0,−1))` (cpp:1626-1627) on the pelvis IMU
  quaternion (wxyz, cpp:2928).
- **How `quat_rotate_d` works.** It computes `(2w²−1)v + 2w(q⃗×v) + 2q⃗(q⃗·v)` without normalising
  (math:132-150). For unit q this is gravity in the pelvis frame, the same as mjlab's
  `projected_gravity` (test `test_gravity_and_heading_match_training_helpers`).
- **Padding.** Padding frames give **(0, 0, +1)**, not (0, 0, −1).

### 3.6 `motion_joint_positions_10frame_step1`
- **Source.** `JointPositions(frame)` (cpp:772) from `joint_pos.csv`: IsaacLab order, **absolute**
  rad (ref.md:45-51). For the planner: MuJoCo qpos permuted to IsaacLab, absolute (planner:493-497).
- **Not offset.** These are not relative to `default_angles`. This is unlike §3.2.
- **Window.** Frames `current + 0..9`, clamped at the clip end. All frames are the current frame
  while paused (§2.2).
- **Layout.** 10 × 29.

### 3.7 `motion_joint_velocities_10frame_step1`
- **Source.** `JointVelocities(frame)`: `joint_vel.csv` in IsaacLab order, rad/s. For the planner it
  is the forward difference `(q[f+1] − q[f])·50`, with the last frame repeating the previous one
  (planner:501-516).
- **Window.** Clamped at the clip end. **All zeros while paused.**
- **Layout.** 10 × 29.

### 3.8 `motion_root_z_position_10frame_step1`
- **Source.** `BodyPositions(frame)[0][2]` (cpp:449): the z of body 0 (pelvis) in the motion's world
  frame, in metres. No heading or offset is applied. For the planner it is qpos z (planner:471-475).
- **Window.** Same window, clamping and pause rule as §3.6.

### 3.9 `motion_anchor_orientation_10frame_step1` and the heading state
- **Anchor.** Body 0 = **pelvis** on both sides:
  - reference side: `BodyQuaternions(frame)[0]`, cpp:660-661, ref.md:79-89;
  - robot side: the pelvis IMU `base_quat` of the current tick, cpp:628-629.
- **Per frame k:**
  ```
  q_rel[k] = conj(left) ⊗ (A ⊗ q_ref[k])          (cpp:664, 678)
  out[k]   = [R00, R01, R10, R11, R20, R21](q_rel[k])   (cpp:679-687)
  ```
  - R comes from the **normalised** q (math:410-436): the first two columns, flattened row-wise.
    This is the same layout as mjlab's `motion_anchor_ori_b` and our `rot6d_from_quat`.
  - `left` = `base_quat` (mode 0).
- **Heading alignment `A`** (`ComputeApplyDeltaHeading`, cpp:593-606):
  `A = Rz(δ) ⊗ H(init_base_quat) ⊗ H(init_ref_root_quat)⁻¹`.
  - The `Rz(δ)` factor is applied only if δ ≠ 0.
  - `H(q)` is the yaw-only quaternion of `atan2` of q's rotated x-axis (math:206-215, 300-318).
    This is the same angle as our `heading_yaw` and mjlab's `yaw_quat`.
- **`UpdateHeadingState`** (cpp:559-585) runs once per tick before the gatherers:
  - When `reinitialize_heading_` is set, it captures `init_base_quat` from the current tick's pelvis
    IMU and sets δ = 0 (cpp:564-570, robot:55-61). It sets `init_ref_root_quat` to the pelvis
    quaternion of the frame current at that tick (cpp:571-572).
  - Whenever the current frame is 0, `init_ref_root_quat` is refreshed to frame 0's quaternion
    (cpp:581-584).
- **What sets `reinitialize_heading_`:**
  - the start value `true`, so the first CONTROL tick re-inits (cpp:255);
  - motion next/previous/restart (kbd:381-431) and the explicit re-init key (kbd:457-460);
  - planner disable (kbd:524-536) and the safety reset (kbd:343-361);
  - the end of a reference clip (cpp:3425) and the first planner motion (cpp:3329-3331);
  - the gamepad / ZMQ / ROS2 handlers in the same situations.
- **Changing δ.** The keyboard changes δ by ±π/12 per press (kbd:440-454). The gamepad and ZMQ change
  it by ±0.1, or ZMQ sets an absolute value.
- **Net effect.** The reference clip is yawed so that its heading at the re-init frame equals the
  robot's heading at the re-init tick. A yaw error that exists at re-init is invisible to the
  policy. Yaw that the robot or reference gains afterwards is visible. Roll and pitch are always
  compared in full.

Variants (documented, implemented):
- `motion_anchor_orientation` (6): the current frame only.
- `…_heading*` (mode 1): `left = H(base_quat)`, so robot roll and pitch are removed (cpp:671-672).
- `…_refheading*` (mode 2): `left = H(A ⊗ q_ref[current frame, clamped])` for every frame
  (cpp:636-645, 673-674).
  - It does not depend on robot state or on `A`, because `A` is pure yaw.
  - It equals the rotation part of our `traj` goal features (test
    `test_refheading_equals_traj_goal_rotation`).
- `heading_diff_robot_ref` (6, not implemented): `rot6d(H(base)⁻¹ ⊗ H(A ⊗ q_ref[current]))`
  (cpp:703-735).

## 4. Decoder input assembly

- **YAML format** (`--obs-config`):
  - Only `observations:` entries matter. Each `- name: "…"` line must be followed **on the next
    line** by `enabled: true` (obscfg:176-190). Only the literal `true` enables an entry
    (obscfg:486-489). Lines starting with `#` are skipped (obscfg:156). A comment between `name` and
    `enabled` would break the pair.
  - Enabled terms are concatenated in file order, each at the running offset (cpp:1828-1852,
    obs.md:32-36).
  - An unknown name throws (cpp:1837-1840). The total dimension must equal the ONNX input size
    (cpp:1855-1860, 1956-1964).
- **No encoder.** Without `token_state` the `encoder:` section is ignored (obscfg:327-333). The mode
  becomes −2, so no token or encoder logic runs (cpp:2403-2412, 3034).
- **Precision.** The buffer is double, zero-filled each tick, and cast to float32 for inference
  (cpp:2004, 3140-3142).

Planned decoder config, generated by the export step (1580 inputs):

```yaml
observations:
  - name: "his_base_angular_velocity_10frame_step1"   # 30
    enabled: true
  - name: "his_body_joint_positions_10frame_step1"    # 290
    enabled: true
  - name: "his_body_joint_velocities_10frame_step1"   # 290
    enabled: true
  - name: "his_last_actions_10frame_step1"            # 290
    enabled: true
  - name: "his_gravity_dir_10frame_step1"             # 30
    enabled: true
  - name: "motion_joint_positions_10frame_step1"      # 290
    enabled: true
  - name: "motion_joint_velocities_10frame_step1"     # 290
    enabled: true
  - name: "motion_root_z_position_10frame_step1"      # 10
    enabled: true
  - name: "motion_anchor_orientation_10frame_step1"   # 60
    enabled: true
```

The term order is our choice. The YAML order, the order of `assemble_decoder_input` and the order
the network was trained with must be identical.

### 4.1 With an encoder (SONIC's release policy)

`policy/release/observation_config.yaml` has a 994-d decoder (`token_state` 64 + the five
`his_*_10frame_step1` terms; the file's "436" comment is stale) and a 1762-d encoder. Implemented
in `realmimic.obs.sonic.assemble_encoder_input` and `realmimic.eval.sonic_release_policy`;
contract tests in `tests/realmimic/test_sonic_release_policy.py`.

- **When.** `token_state` is gathered first; its gatherer fills the encoder buffer, runs the
  encoder and copies the 64 outputs in (cpp:1648-1689).
- **Layout.** Every enabled `encoder_observations` term, in file order, at the running offset
  (cpp:1871-1893).
- **Mode.** Clip and planner motions get encode mode 0 (`g1`) once an encoder is loaded
  (cpp:2402, 2417-2426). The buffer is zeroed and only that mode's `required_observations` are
  gathered (cpp:2030-2031, 2088-2116): `encoder_mode_4` = `[0, 0, 0, 0]` (cpp:1691-1697),
  `motion_joint_positions` / `motion_joint_velocities` / `motion_anchor_orientation` `_10frame_step5`.
  Everything else (root z, teleop, SMPL terms) is 0.
- **Window.** Frames `current + 5k`, k = 0..9, clamped to the last frame; paused: all frames current,
  velocities 0 (§2.2, same gatherers with step 5).
- **ONNX.** `model_encoder.onnx` selects the branch by `int(obs[0])` (one-hot). The g1 branch reads
  `obs[4:584]` as (10, 58) and `obs[601:661]` as (10, 6), concatenates per row, and runs an MLP and
  an FSQ quantizer (tokens are multiples of 1/16). The (10, 58) rows are not per-frame `[q, dq]`;
  SONIC's training term reshapes `cat(q[10×29], dq[10×29])` the same way.

Verified against the runtime: `deploy/sonic/sim2sim.py` writes `release_check.json` for release
clip sessions (token bit-exact; action vs the next tick's logged `last_action`).

## 5. ONNX interface (no encoder)

- **Input.** Exactly one input, named `obs_dict`, float32. Its element count is the product of its
  dims (policy:101-130).
- **Output.** Exactly one output, named `action`, float32, with a product of dims of 29
  (policy:136-172).
- **Engine.** The model is converted to a cached TensorRT engine. It runs in FP32 unless
  `--policy-precision 16` is given (cpp:4174, 4369).
- **Shape.** Export with a static shape `[1, 1580]` → `[1, 29]` (see ambiguity A1).

## 6. Action post-processing (cpp:3155-3165)

For each hardware/MuJoCo motor `i`:

```
q_target[i] = default_angles[i] + action[isaaclab_to_mujoco[i]] * g1_action_scale[i]
kp[i] = kps[i] * kp_scale[i]   kd[i] = kds[i] * kd_scale[i]   dq_target = 0   tau_ff = 0
```

- **Gain scales.** `apply_motor_gain_scales` multiplies kp/kd by the optional per-hardware-index
  `--motor-kp-scale` / `--motor-kd-scale` factors, each in [0, 4] (gains:172-182, cpp:4389-4396).
  The INIT ramp uses the unscaled `kps`/`kds` (cpp:2762-2794).
- **Writer.** A 500 Hz writer sends the latest command (cpp:2694-2716).
- **Reference.** `realmimic.obs.sonic.joint_position_targets` implements this mapping.

## 7. Joint orders and constants

- **`isaaclab_to_mujoco`** (params:100-101): `mujoco[i] = isaaclab[isaaclab_to_mujoco[i]]`, i.e. the
  IsaacLab index of MuJoCo joint *i*.
- **`mujoco_to_isaaclab`** (params:103-104): `isaaclab[i] = mujoco[mujoco_to_isaaclab[i]]`.
  - This is the usage at cpp:2862-2864, 3156 and logger.cpp:296, and the meaning of SONIC's Python
    `G1_*_DOF` arrays (`docs/source/references/conventions.md`, "Joint Ordering").
- **Which side uses which order.** Motors, `default_angles`, `kps`, `kds` and `g1_action_scale` are
  in MuJoCo order. The policy, the motion data and every observation are in IsaacLab order.
- **Formulas.** Constants follow params:40-65:
  - `kp = armature·ω²` and `kd = 2·2·armature·ω`, with ω = 2π·10 Hz.
  - `action_scale = 0.25·effort/kp`.
  - Ankles and waist roll/pitch use **2×** kp/kd, but the single-motor action scale.
  - **Hip pitch is assumed 7520-22** (params:110, 144).
  - `kps`/`kds` are `std::array<float>`. `default_angles` equals mjlab's `KNEES_BENT_KEYFRAME`.

| Joint group (MuJoCo) | Motor | kp | kd | action scale | default |
|---|---|---|---|---|---|
| hip pitch | 7520-22 | 99.098 | 6.309 | 0.35066 | −0.312 |
| hip roll, knee | 7520-22 | 99.098 | 6.309 | 0.35066 | 0 / 0.669 |
| hip yaw, waist yaw | 7520-14 | 40.179 | 2.558 | 0.54755 | 0 |
| ankle pitch/roll, waist roll/pitch | 2×5020 | 28.501 | 1.814 | 0.43858 | −0.363 / 0 / 0 |
| shoulders, elbow, wrist roll | 5020 | 14.251 | 0.907 | 0.43858 | sh. pitch 0.2, sh. roll ±0.2, elbow 0.6 |
| wrist pitch/yaw | 4010 | 16.778 | 1.068 | 0.07450 | 0 |

## 8. Differences vs our training conventions

Training conventions come from these files:
- `supermimic/world_model/heads.py`: `OBS_TERMS`, `analytic_observation`,
  `flatten_history_term_major`, `anchor_relative_obs`;
- `world_model/state_spec.py`;
- `envs/tracking_env_cfg.py`, `envs/mdp/commands.py`, `envs/mdp/motion_library.py`;
- mjlab's `motion_anchor_ori_b`.

| Aspect | Ours (training) | SONIC runtime | Compatible? |
|---|---|---|---|
| Joint order (all 29-d terms, action) | MuJoCo (mjlab G1, `state_spec`) | IsaacLab | **No**: permute (`MUJOCO_TO_ISAACLAB`) |
| Robot joint pos | `q + bias − q0`, q0 = KNEES_BENT | `q − default_angles` | Yes (q0 identical), order differs |
| Joint vel, base ang vel (pelvis frame), gravity (pelvis) | mjlab `joint_vel_rel`, `base_ang_vel`, `projected_gravity` | same quantities from the pelvis IMU / encoders | Yes (joint vel: order differs) |
| History layout | per term, oldest → newest (`flatten_history_term_major`); term order `OBS_TERMS` = jp, jv, ang, grav, act | per term, oldest → newest; term order = YAML | Yes, if the YAML lists terms in `OBS_TERMS` order (or the wrapper reorders) |
| History initialisation | mjlab `CircularBuffer` back-fills the first frame on every reset | zero entries for the first 9 ticks after `start` only (joints at default, gravity **+z**, zero vel/act); afterwards continuous across clips | **No** |
| Last action | raw clean action, MuJoCo order, our action scale (`rm_g1.action_scale(robot_spec)`, hip pitch 0.5475 for the stock 7520-14 spec) | raw ONNX output, IsaacLab order, SONIC scale (hip pitch 0.3507) | **No**, if the wrapper rescales the output, the fed-back action is in SONIC units |
| PD gains | RobotSpec: hip pitch 7520-14 → kp 40.18, kd 2.56 | `kps`/`kds`: hip pitch 7520-22 → 99.10 / 6.31 | **No** for 7520-14 robots: `--motor-k[pd]-scale 0,6=0.405448` |
| Reference joint pos | `joint_pos − q0` (goal v1/traj) | absolute | **No** (constant offset) |
| Goal layout | frame-major, 68 (v1) / 77 (traj) features per frame | term-major: 10× joint pos, 10× joint vel, 10× root z, 10× anchor rot6d | **No** |
| Goal content | ref gravity, ref root ang vel (body), ref root lin vel (heading), height, [traj: root displacement + rot6d vs window heading] | joint pos, joint vel, root z, anchor orientation; no ref velocities, gravity or displacement exist in the registry | **No**: drop gravity / ang vel / lin vel / displacement; traj's rot6d == `…_refheading_10frame_step1` |
| Reference window | frames `frame..frame+9`, clamped at clip end (`goal_window`) | same (step1) | Yes |
| Clip end | `clip_end` time-out terminates | reference clip: stop, back to frame 0 paused, heading re-init; planner: hold last frame | Accept (deploy via planner) |
| Paused playback | never | at start-up and after every clip switch: all window frames = current, velocities 0 | **No**: not covered in training |
| Anchor body | `ANCHOR_BODY = "torso_link"` (`motion_anchor_ori_b`, imagination `compiled.py` FK of torso) | pelvis (body 0 vs pelvis IMU) | **No** |
| Anchor frames | current frame only (6-d) | 10 future frames (60-d) | **No** |
| Anchor heading | raw world frames: the reset yaw noise (±0.2 rad) is visible | yaw alignment `A` captured at heading re-init: the initial yaw error is invisible | **No** |
| rot6d layout, quaternions | `[R00,R01,R10,R11,R20,R21]`, wxyz | same | Yes |
| Root z | ref pelvis z in the clip world frame | `body_pos.csv` body-0 z / planner qpos z | Yes, if the CSV export writes real pelvis z on the same floor |
| Step / dt | 0.02 s (decimation 4 × 5 ms) | `control_dt_` 0.02 s | Yes |

### 8.1 What must change for compatibility

1. **Anchor term** (`tracking_env_cfg.make_env_cfg` → `anchor_terms["anchor_ori"]`,
   `track_mdp.motion_anchor_ori_b`, and `imagination/compiled.py::CompiledStep.goal` via
   `heads.anchor_relative_obs`):
   - use the **pelvis** (the root quat already in the 71-d state, and the reference root quat),
     not `torso_link`;
   - use **10 future frames**;
   - apply the yaw alignment `A` (`sonic.heading_alignment`), captured per env when
     `MultiClipMotionCommand._resample_command` resets the robot: the robot pelvis after reset noise,
     and the reference pelvis at the start frame. Hold `A` fixed for the episode.

   Compute the term with `sonic.motion_anchor_orientation`. The reward's global anchor terms
   (`motion_global_root_pos/ori`, which use the torso anchor in world frames) should use the same
   aligned reference, or they penalise a yaw error the policy cannot observe.
2. **Goal features** (`motion_library.goal_features` / `goal_window`, `goal_spec.GOAL_FEAT_DIMS`):
   - replace the frame-major 68/77-d features with SONIC's term-major
     `motion_joint_positions` (absolute, no `− q0`), `motion_joint_velocities`,
     `motion_root_z_position` and `motion_anchor_orientation`;
   - drop the reference gravity, root ang/lin velocity and traj displacement, which the runtime
     cannot provide.
3. **Joint order** (`heads.analytic_observation`, `OBS_TERMS`, the goal joint terms, and the action
   output): permute to IsaacLab, either in the training observation or inside the ONNX wrapper
   (input slices and the output). Either way, `his_*` term order = YAML order.
4. **History initialisation** (mjlab policy group with `history_length=10`): reproduce SONIC's
   control-start padding (`num_valid` in `sonic.his_*`) for episodes that model a fresh start,
   instead of the back-fill. Otherwise the first 9 ticks on the robot are out of distribution.
   Gravity padding is +z.
5. **Last action units**: if the wrapper converts our action to SONIC units
   (`a_sonic = a_ours · s_ours / ACTION_SCALE`, only hip pitch differs for 7520-14), it must convert
   the `his_last_actions` input slice back (`· ACTION_SCALE / s_ours`). The alternative is to train
   with `sonic.ACTION_SCALE`.
6. **Gains**: for `hip_pitch_motor == "7520_14"` specs, `launch.env` must carry
   `--motor-kp-scale 0,6=0.405448 --motor-kd-scale 0,6=0.405448`. During the 3 s INIT ramp the robot
   still uses 7520-22 gains.
7. **Paused reference**: the runtime shows held frames with zero velocities whenever playback is
   paused. Either train a fraction of episodes with paused windows, or only ever run with the
   planner, starting in idle (patch `0005-start-idle`).

## 9. Ambiguities and how to verify them

| # | Open point | Why | Verification |
|---|---|---|---|
| A1 | ONNX shape rank / dynamic axes | `policy` multiplies the TensorRT dims; how `ConvertONNXToTRT` handles a dynamic batch dim (`options.dynamic_axes_names`) lives in the TRTInference library, not in this tree | Export static `[1, 1580]`; check the "Input dimension: 1580" log line in sim-to-sim |
| A2 | `--policy-input-logfile` precision | values are streamed with the default 6 significant digits (cpp:4064): \|v\| ≥ 10 rad/s loses > 1e-5 absolute | Compare with `rtol = 1e-5` (or patch `std::setprecision(9)`); the design's "≤ 1e-5" must be relative |
| A3 | Heading re-anchor during planner replans | after a replan `current_frame_` is reset to 0 but advanced to 1 in the same call while playing (cpp:3325, 3336-3403), so the frame-0 refresh (cpp:581-584) should never fire while playing; if play is false at a replan it does fire | Sim-to-sim with the planner turning: log `motion_anchor_orientation` across replans and check for yaw jumps |
| A4 | IMU frame on hardware | the term assumes IMU axes = pelvis axes (MJCF `imu_in_pelvis` has identity orientation); the sim publishes the exact pelvis quaternion | On the gantry, compare logged gyro with finite-differenced `base_quat` |
| A5 | Reference CSV conventions | correctness depends on our exporter: IsaacLab order, absolute angles, wxyz, body 0 = pelvis, z on the floor, velocities consistent with positions | `visualize_motion.py` plus a round trip of exported CSV → `MotionDataReader` layout in tests |
| A6 | First-tick timing | history padding and heading capture happen on the first CONTROL tick; motion is usually paused then (velocities 0) | `--policy-input-logfile` on the first 10 rows: frames 0-8 of the joint-position history must be 0, gravity padding +z |

The contract test records `--policy-input-logfile` together with the sim
state in sim-to-sim. It recomputes each term with `realmimic.obs.sonic` and compares the terms
slice by slice using `TERM_DIMS` offsets.
