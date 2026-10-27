// Reference windows for the policy. A row is one 50 Hz reference frame in the training library's layout:
// [root_pos 3 | root_quat 4 (wxyz) | joint_pos 29 | root_lin_vel_w 3 | root_ang_vel_w 3 | joint_vel 29].

export const ROW = 71;

// Heading (yaw-only) quaternion, as mjlab's yaw_quat.
export function yawQuat(q, out, o = 0) {
  const [w, x, y, z] = q;
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  out[o] = Math.cos(yaw / 2);
  out[o + 1] = 0;
  out[o + 2] = 0;
  out[o + 3] = Math.sin(yaw / 2);
}

// A pose held still at a fixed pelvis height, facing wherever the robot faces (no motion data).
export class PoseMotion {
  constructor(jointPos, rootHeight) {
    this.row = new Float64Array(ROW);
    this.row[2] = rootHeight;
    this.row.set(jointPos, 7);
    this.length = Infinity;
  }

  window(tick, rootQuat, frames, out) {
    yawQuat(rootQuat, this.row, 3);
    for (let i = 0; i < frames; i++) out.set(this.row, i * ROW);
  }
}

// Recorded rows (Float32Array, frames x 71). The target frame holds at the last frame, and window
// frames past the end are that frame held still (zero velocities), as the training library does.
// A looping clip (first frame = last frame) wraps around instead.
export class ClipMotion {
  constructor(rows, loop = false) {
    this.rows = rows;
    this.length = rows.length / ROW;
    this.loop = loop;
  }

  // The frame the policy targets at control tick `tick` (row index): the next one, tick + 1, as in training and the
  // benchmark (the env advances the target before the first action; web_closed_loop.clip_reference). The robot
  // starts on frame 0.
  frame(tick) {
    return this.loop ? (tick + 1) % this.length : Math.min(tick + 1, this.length - 1);
  }

  // [qpos 36 | qvel 35] of the first frame: the robot starts there, as in the benchmark. MuJoCo's free joint
  // takes the angular velocity in the body frame; the rows keep it in the world frame.
  startState(out = new Float64Array(71)) {
    const r = this.rows;
    out.set(r.subarray(0, 36), 0);
    out.set(r.subarray(36, 39), 36);
    const [w, x, y, z] = r.subarray(3, 7), a = r.subarray(39, 42);
    const n = w * w + x * x + y * y + z * z;
    // R^T a, with R the rotation of (w, x, y, z)
    const m = [
      [1 - (2 * (y * y + z * z)) / n, (2 * (x * y - w * z)) / n, (2 * (x * z + w * y)) / n],
      [(2 * (x * y + w * z)) / n, 1 - (2 * (x * x + z * z)) / n, (2 * (y * z - w * x)) / n],
      [(2 * (x * z - w * y)) / n, (2 * (y * z + w * x)) / n, 1 - (2 * (x * x + y * y)) / n],
    ];
    for (let i = 0; i < 3; i++) out[39 + i] = m[0][i] * a[0] + m[1][i] * a[1] + m[2][i] * a[2];
    out.set(r.subarray(42, 71), 42);
    return out;
  }

  window(tick, rootQuat, frames, out) {
    if (this.loop) {
      for (let i = 0; i < frames; i++) {
        const f = (tick + 1 + i) % this.length;
        out.set(this.rows.subarray(f * ROW, (f + 1) * ROW), i * ROW);
      }
      return;
    }
    // As training's freeze_clip_end: goal frame i is clip frame tick + 1 + i, clamped to the last; every frame past
    // the end, counting the ticks already played past it, is the last frame with zero velocities
    // (MotionLibrary._window_index with `beyond`).
    const last = this.length - 1;
    for (let i = 0; i < frames; i++) {
      const pos = tick + 1 + i;
      const f = Math.min(pos, last);
      out.set(this.rows.subarray(f * ROW, (f + 1) * ROW), i * ROW);
      if (pos > last) out.fill(0, i * ROW + 36, (i + 1) * ROW);
    }
  }
}

// entry: one item of assets/motions/index.json; meta: assets/policy/meta.json.
export async function loadMotion(entry, meta, fetchBytes) {
  if (entry.kind === 'pose') {
    if (entry.pose !== 'default') throw new Error(`unknown pose ${entry.pose}`);
    return new PoseMotion(meta.default_joint_pos, meta.stand_root_height);
  }
  if (entry.kind === 'clip') {
    const bytes = await fetchBytes(entry.file);
    return new ClipMotion(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4), Boolean(entry.loop));
  }
  throw new Error(`unknown motion kind ${entry.kind}`);
}
