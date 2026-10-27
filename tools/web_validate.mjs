// Headless check of the browser demo with the site's own code (site/demo/sim.js, policy.js, motion.js)
// and vendored MuJoCo / onnxruntime-web:
//   1. policy parity on the export's test vectors (onnxruntime-web vs ONNX Runtime in Python);
//   2. the stand + scripted pushes closed loop vs the Python CPU MuJoCo trace (tools/web_closed_loop.py);
//   3. the world model on its test vectors and the page's world-model rollout next to the closed loop.
//
//   node tools/web_validate.mjs site <trace.json> [vectors.json] [world_model_vectors.json]

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [siteArg, tracePath, vectorsPath, wmVectorsPath] = process.argv.slice(2);
const site = path.resolve(siteArg);
const imp = (p) => import(pathToFileURL(path.join(site, p)).href);
const { Sim } = await imp('demo/sim.js');
const { Policy, Controller } = await imp('demo/policy.js');
const { loadMotion } = await imp('demo/motion.js');
const ort = await imp('vendor/onnxruntime-web/ort.wasm.min.mjs');
const mujoco = await (await imp('vendor/mujoco/mujoco.js')).default();
ort.env.wasm.numThreads = 1;

const bytes = (p) => new Uint8Array(fs.readFileSync(path.join(site, p)));
const meta = JSON.parse(fs.readFileSync(path.join(site, 'assets/policy/meta.json'), 'utf8'));
const index = JSON.parse(fs.readFileSync(path.join(site, 'assets/motions/index.json'), 'utf8'));
const sim = Sim.create(mujoco, bytes('assets/robot/scene.mjb'));
const policy = await Policy.create(ort, bytes('assets/policy/policy.onnx'), meta);
let ok = true;

if (vectorsPath) {
  const vec = JSON.parse(fs.readFileSync(vectorsPath, 'utf8'));
  let err = 0;
  for (const s of vec.samples) {
    policy.states.set(s.state_hist);
    policy.actions.set(s.action_hist);
    const out = await policy.act(s.ref_window);
    for (let j = 0; j < 29; j++) err = Math.max(err, Math.abs(out.action[j] - s.action[j]), Math.abs(out.target[j] - s.joint_target[j]));
  }
  console.log(`policy parity, onnxruntime-web vs Python ONNX Runtime (${vec.samples.length} samples): max abs ${err.toExponential(2)}`);
  ok &&= err < 1e-4;
}

const trace = JSON.parse(fs.readFileSync(tracePath, 'utf8'));
const ctl = new Controller(sim, policy, meta);
ctl.setMotion(await loadMotion(index.motions.find((m) => m.id === 'stand'), meta, async (f) => bytes(path.join('assets/motions', f))));
const pushes = trace.pushes.map((p) => ({ ...p, id: sim.bodyId(p.body) }));
const ticks = trace.t.length;
let dz = 0, dq = 0, fall = null, tPolicy = 0;
const t0 = performance.now();
for (let k = 0; k < ticks; k++) {
  const t = k * meta.control_dt;
  sim.clearForces();
  for (const p of pushes) if (p.start <= t && t < p.start + p.duration - 1e-9) sim.addForce(p.id, p.force);
  const a = performance.now();
  await ctl.step();
  tPolicy += performance.now() - a;
  const q = sim.data.qpos, ref = trace.qpos[k];
  dz = Math.max(dz, Math.abs(q[2] - trace.pelvis_z[k]));
  for (let i = 0; i < q.length; i++) dq = Math.max(dq, Math.abs(q[i] - ref[i]));
  if (fall === null && ctl.fallen) fall = k;
}
const wall = (performance.now() - t0) / 1000;
const z = sim.data.qpos[2];
console.log(`closed loop ${ticks} ticks (${(ticks * meta.control_dt).toFixed(1)} s sim in ${wall.toFixed(2)} s wall, ${(tPolicy / ticks).toFixed(2)} ms per tick incl. 4 physics steps)`);
console.log(`  fall: JS ${fall === null ? 'none' : 't=' + trace.t[fall].toFixed(2) + ' s'}, Python ${trace.fall_tick === null ? 'none' : 't=' + trace.t[trace.fall_tick].toFixed(2) + ' s'}`);
console.log(`  pelvis z: JS final ${z.toFixed(4)} m, Python final ${trace.pelvis_z[ticks - 1].toFixed(4)} m, max |dz| ${dz.toExponential(2)} m; max |dqpos| ${dq.toExponential(2)}`);
ok &&= fall === null && trace.fall_tick === null && dz < 0.03;

// 3. The world model: onnxruntime-web vs Python ONNX Runtime on its test vectors, then the
// page's world-model rollout next to the JS closed loop (standing) for one training horizon.
if (wmVectorsPath) {
  const { WorldModelRollout } = await imp('demo/world_model.js');
  const wm = await ort.InferenceSession.create(bytes('assets/policy/world_model.onnx'), { executionProviders: ['wasm'] });
  const vec = JSON.parse(fs.readFileSync(wmVectorsPath, 'utf8'));
  let err = 0;
  for (const s of vec.samples) {
    const out = await wm.run({ state: new ort.Tensor('float32', Float32Array.from(s.state), [1, 71]), action: new ort.Tensor('float32', Float32Array.from(s.action), [1, 29]), friction: new ort.Tensor('float32', Float32Array.of(s.friction), [1, 1]) });
    for (let k = 0; k < 71; k++) err = Math.max(err, Math.abs(out.next_state.data[k] - s.next_state[k]));
  }
  console.log(`world model parity, onnxruntime-web vs Python ONNX Runtime (${vec.samples.length} samples): max abs ${err.toExponential(2)}`);
  ok &&= err < 1e-4;
  const rollout = new WorldModelRollout(ort, wm, policy, meta);
  const errs = new Float64Array(sim.model.nbody);
  ctl.reset();
  sim.clearForces();
  for (let k = 0; k < 50; k++) await ctl.step();
  rollout.sync(sim, ctl);
  let div = 0;
  for (let k = 0; k < 32; k++) {
    await ctl.step();
    await rollout.step(ctl.motion);
    div = rollout.divergence(sim, errs);
  }
  console.log(`world model vs reality after 32 ticks (standing, JS): mean body position error ${(100 * div).toFixed(2)} cm`);
  ok &&= div < 0.02;
}
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
