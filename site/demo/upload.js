// "Load your own motion": a G1 motion file chosen by the visitor, converted in the browser into reference
// rows (see motion.js) for the policy to track. Nothing is uploaded anywhere. Formats:
//   BONES-SEED G1 CSV  header "Frame,root_translateX,...": position cm, Euler xyz (extrinsic) degrees,
//                      joints degrees as "<joint>_dof" columns; 120 fps;
//   header-less CSV    per row: position m (3), quaternion xyzw (4), 29 joints rad in G1 order; `fps` (30);
//   realmimic .npz     our converted clips (50 Hz): joint_pos, joint_vel, body_*_w with the pelvis first.

import { ROW, ClipMotion } from './motion.js';

const OUT_FPS = 50;

// ---------------------------------------------------------------- quaternions (wxyz)

const qmul = (a, b) => [
  a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
  a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
  a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
];
const qconj = (q) => [q[0], -q[1], -q[2], -q[3]];
const axis = (i, deg) => {
  const h = (deg * Math.PI) / 360;
  const q = [Math.cos(h), 0, 0, 0];
  q[1 + i] = Math.sin(h);
  return q;
};
// Extrinsic xyz Euler (scipy "xyz"): rotate about world x, then y, then z, i.e. R = Rz Ry Rx.
const eulerXyz = (x, y, z) => qmul(axis(2, z), qmul(axis(1, y), axis(0, x)));

function slerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if (d < 0) {
    b = b.map((v) => -v);
    d = -d;
  }
  if (d > 0.9995) {
    const q = a.map((v, i) => v + t * (b[i] - v));
    const n = Math.hypot(...q);
    return q.map((v) => v / n);
  }
  const th = Math.acos(d), s = Math.sin(th);
  return a.map((v, i) => (Math.sin((1 - t) * th) * v + Math.sin(t * th) * b[i]) / s);
}

function rotvec(q) {
  if (q[0] < 0) q = q.map((v) => -v);
  const s = Math.hypot(q[1], q[2], q[3]);
  if (s < 1e-9) return [2 * q[1], 2 * q[2], 2 * q[3]];
  const a = (2 * Math.atan2(s, q[0])) / s;
  return [a * q[1], a * q[2], a * q[3]];
}

// ---------------------------------------------------------------- raw frames -> 50 Hz rows

// frames: {pos: [[x,y,z]], quat: [[w,x,y,z]], joints: [[29]]} at `fps`.
export function toRows(frames, fps) {
  const n = frames.pos.length;
  if (n < 2) throw new Error('the motion needs at least two frames');
  const duration = (n - 1) / fps;
  const m = Math.floor(duration * OUT_FPS + 1e-6) + 1;
  const pos = [], quat = [], joints = [];
  for (let k = 0; k < m; k++) {
    const u = Math.min((k / OUT_FPS) * fps, n - 1), i = Math.floor(u), j = Math.min(i + 1, n - 1), t = u - i;
    pos.push(frames.pos[i].map((v, c) => v + t * (frames.pos[j][c] - v)));
    quat.push(slerp(frames.quat[i], frames.quat[j], t));
    joints.push(frames.joints[i].map((v, c) => v + t * (frames.joints[j][c] - v)));
  }
  const rows = new Float32Array(m * ROW);
  const dt = 1 / OUT_FPS;
  for (let k = 0; k < m; k++) {
    const a = Math.max(k - 1, 0), b = Math.min(k + 1, m - 1), h = (b - a) * dt || dt;
    const r = rows.subarray(k * ROW, (k + 1) * ROW);
    r.set(pos[k], 0);
    r.set(quat[k], 3);
    r.set(joints[k], 7);
    r.set(pos[b].map((v, c) => (v - pos[a][c]) / h), 36);
    r.set(rotvec(qmul(quat[b], qconj(quat[a]))).map((v) => v / h), 39); // world-frame angular velocity
    r.set(joints[b].map((v, c) => (v - joints[a][c]) / h), 42);
  }
  return rows;
}

// ---------------------------------------------------------------- formats

function parseCsv(text, jointNames, fps) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  if (lines[0].startsWith('Frame,root_translateX')) {
    const head = lines[0].split(',').map((s) => s.trim());
    const col = (name) => {
      const i = head.indexOf(name);
      if (i < 0) throw new Error(`missing column ${name}`);
      return i;
    };
    const p = ['root_translateX', 'root_translateY', 'root_translateZ'].map(col);
    const e = ['root_rotateX', 'root_rotateY', 'root_rotateZ'].map(col);
    const jc = jointNames.map((j) => col(`${j}_dof`));
    const frames = { pos: [], quat: [], joints: [] };
    for (const line of lines.slice(1)) {
      const v = line.split(',').map(Number);
      frames.pos.push(p.map((i) => v[i] / 100));
      frames.quat.push(eulerXyz(v[e[0]], v[e[1]], v[e[2]]));
      frames.joints.push(jc.map((i) => (v[i] * Math.PI) / 180));
    }
    return { rows: toRows(frames, 120), format: 'BONES-SEED G1 CSV, 120 fps' };
  }
  const frames = { pos: [], quat: [], joints: [] };
  for (const line of lines) {
    const v = line.split(',').map(Number);
    if (v.length !== 36 || v.some(Number.isNaN)) throw new Error('expected 36 numbers per row: position m, quaternion xyzw, 29 joints rad');
    frames.pos.push(v.slice(0, 3));
    frames.quat.push([v[6], v[3], v[4], v[5]]);
    frames.joints.push(v.slice(7));
  }
  return { rows: toRows(frames, fps), format: `G1 CSV, ${fps} fps` };
}

// Minimal zip (stored / deflate) + npy reader for numpy's .npz.
async function readNpz(buf) {
  const dv = new DataView(buf);
  let eocd = buf.byteLength - 22;
  while (eocd >= 0 && dv.getUint32(eocd, true) !== 0x06054b50) eocd--;
  if (eocd < 0) throw new Error('not a zip / npz file');
  const count = dv.getUint16(eocd + 10, true);
  let p = dv.getUint32(eocd + 16, true);
  const arrays = {};
  for (let k = 0; k < count; k++) {
    const method = dv.getUint16(p + 10, true), size = dv.getUint32(p + 20, true);
    const nameLen = dv.getUint16(p + 28, true), extraLen = dv.getUint16(p + 30, true), commentLen = dv.getUint16(p + 32, true);
    const local = dv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(new Uint8Array(buf, p + 46, nameLen)).replace(/\.npy$/, '');
    p += 46 + nameLen + extraLen + commentLen;
    const start = local + 30 + dv.getUint16(local + 26, true) + dv.getUint16(local + 28, true);
    let data = new Uint8Array(buf, start, size);
    if (method === 8) data = new Uint8Array(await new Response(new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer());
    else if (method !== 0) throw new Error(`unsupported zip compression ${method}`);
    const hlen = new DataView(data.buffer, data.byteOffset).getUint16(8, true);
    const header = new TextDecoder().decode(data.subarray(10, 10 + hlen));
    const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
    const shape = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1].split(',').filter((s) => s.trim()).map(Number);
    const body = data.slice(10 + hlen).buffer;
    if (descr === '<f4') arrays[name] = { data: new Float32Array(body), shape };
    else if (descr === '<f8') arrays[name] = { data: new Float64Array(body), shape };
    else if (descr === '<i4') arrays[name] = { data: Float64Array.from(new Int32Array(body)), shape };
    else if (descr === '<i8') arrays[name] = { data: Float64Array.from(new BigInt64Array(body), Number), shape }; // e.g. an integer fps
  }
  return arrays;
}

async function parseNpz(buf) {
  const a = await readNpz(buf);
  for (const k of ['joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w']) if (!a[k]) throw new Error(`the npz has no ${k}`);
  const T = a.joint_pos.shape[0], nb = a.body_pos_w.shape[1];
  const fps = a.fps ? a.fps.data[0] : OUT_FPS;
  const frames = { pos: [], quat: [], joints: [] };
  for (let t = 0; t < T; t++) {
    frames.pos.push(Array.from(a.body_pos_w.data.subarray(t * nb * 3, t * nb * 3 + 3)));
    frames.quat.push(Array.from(a.body_quat_w.data.subarray(t * nb * 4, t * nb * 4 + 4)));
    frames.joints.push(Array.from(a.joint_pos.data.subarray(t * 29, t * 29 + 29)));
  }
  return { rows: toRows(frames, fps), format: `realmimic npz, ${fps} fps` };
}

// File -> {motion: ClipMotion (plays once, starts at its first frame), format, seconds}.
export async function motionFromFile(file, jointNames, csvFps = 30) {
  const buf = await file.arrayBuffer();
  const { rows, format } = /\.npz$/i.test(file.name) ? await parseNpz(buf) : parseCsv(new TextDecoder().decode(buf), jointNames, csvFps);
  for (const v of rows) if (!Number.isFinite(v)) throw new Error('the motion contains non-numbers');
  return { motion: new ClipMotion(rows, false), format, seconds: rows.length / ROW / OUT_FPS };
}
