// Page wiring: the recorded montage (hero.js) plays while everything downloads and initialises behind it,
// in parallel with per-asset progress (shown on the "Try it live" button); the first policy inference and
// a first render are run ahead so the demo starts at once on the click. Live, the controller runs at 50 Hz
// of simulated time (fixed steps, 4 x 5 ms physics per policy tick) under requestAnimationFrame. Paused
// while the tab is hidden; resets itself after a fall. Only light modules are imported statically, so the
// page is interactive before three.js, MuJoCo and ONNX Runtime have arrived.

import { Sim } from './sim.js';
import { Policy, Controller } from './policy.js';
import { loadMotion } from './motion.js';
import { motionFromFile } from './upload.js';
import { WorldModelRollout, DIVERGENCE_FULL } from './world_model.js';
import { Hero } from './hero.js';

const ASSETS = new URL('../assets/', import.meta.url);
const VENDOR = new URL('../vendor/', import.meta.url);
const MAX_TICKS_PER_FRAME = 5; // drop simulated time rather than spiral when the page is slow
const RESET_AFTER_FALL = 1.5; // s of simulated time
const RECOVER_TRY = 3; // s the policy gets, once switched back on, to bring a collapsed robot back to the reference
const FPS = 50; // control ticks per second
const REPLAY_AFTER_END = 2; // s a clip that plays once holds its last frame (settling into it) before it starts again
const PUSH_CAP = 0.5; // drag force cap, as a fraction of the robot's weight (see interact.js)

// The two WASM binaries come gzipped (tools/fetch_site_vendor.sh) and are inflated here, a quarter of the
// download on any host; browsers without DecompressionStream fetch them as they are.
const GZ = typeof DecompressionStream !== 'undefined';
const wasm = (path, raw, gz) => (GZ ? { url: new URL(`${path}.gz`, VENDOR), bytes: gz } : { url: new URL(path, VENDOR), bytes: raw });

// Downloads shown in the loader. `bytes`: size of the pinned vendor files, for progress when the server
// compresses them (content-length is then the compressed size).
const DOWNLOADS = {
  mujoco: { label: 'Physics engine · MuJoCo 3.16 WASM', ...wasm('mujoco/mujoco.wasm', 10751588, 2635234) },
  ort: { label: 'Inference runtime · ONNX Runtime Web', ...wasm('onnxruntime-web/ort-wasm-simd-threaded.wasm', 14239897, 3659927) },
  policy: { label: 'Policy network', url: new URL('policy/policy.onnx', ASSETS) },
  meshes: { label: 'Robot meshes · Unitree G1', url: new URL('robot/meshes.bin', ASSETS) },
  scene: { label: 'Robot model · MuJoCo scene', url: new URL('robot/scene.mjb', ASSETS) },
};

const $ = (id) => document.getElementById(id);
const MB = (n) => (n / 1e6).toFixed(1);

// Loader panel: one row per asset (download progress, then compile / ready) and a total bar.
class Loader {
  constructor() {
    this.rows = new Map();
    this.list = $('load-list');
  }

  add(key, label, total = 0) {
    const li = document.createElement('li');
    li.className = 'asset';
    li.innerHTML = '<span class="name"></span><span class="state">queued</span><div class="bar"><div class="bar-fill"></div></div>';
    li.querySelector('.name').textContent = label;
    this.list.append(li);
    const row = { li, state: li.querySelector('.state'), fill: li.querySelector('.bar-fill'), got: 0, total, done: false };
    this.rows.set(key, row);
    return row;
  }

  progress(key, got, total) {
    const r = this.rows.get(key);
    r.got = got;
    r.total = Math.max(total || r.total, got);
    r.fill.style.width = r.total ? `${(100 * got) / r.total}%` : '0';
    r.state.textContent = r.total ? `${MB(got)} / ${MB(r.total)} MB` : `${MB(got)} MB`;
    this.sum();
  }

  set(key, text, done = false) {
    const r = this.rows.get(key);
    r.state.textContent = text;
    if (done) {
      r.done = true;
      r.got = r.total = Math.max(r.total, r.got, 1);
      r.fill.style.width = '100%';
      r.li.classList.add('ready');
    }
    this.sum();
  }

  fail(key, msg) {
    const r = this.rows.get(key);
    r.li.classList.add('error');
    r.state.textContent = 'failed';
    $('load-note').textContent = `Could not start the demo: ${msg}`;
    this.onFail?.(msg);
  }

  sum() {
    let got = 0, total = 0;
    for (const r of this.rows.values()) {
      got += r.done ? r.total : r.got;
      total += r.total;
    }
    const f = total ? got / total : 0;
    $('load-total').textContent = `${Math.floor(100 * f)} %`;
    $('load-total-bar').style.width = `${100 * f}%`;
    this.onProgress?.(f);
  }
}

// A dropped connection (flaky network, SSH tunnel) should not end the demo: retry a few times, with a pause.
const TRIES = 3;
async function retry(fn) {
  for (let i = 0; ; i++) {
    try {
      return await fn(i);
    } catch (e) {
      if (i >= TRIES - 1) throw e;
      await new Promise((resolve) => setTimeout(resolve, 600 * (i + 1)));
    }
  }
}

// Dynamic import with retries; a retry imports the module's URL with a query, since a failed import of the
// same URL may stay failed in the page's module map.
const load = (spec, path) => retry((i) => import(i === 0 ? spec : `${new URL(path, import.meta.url).href}?retry=${i}`));

const fetchBytes = (url, onProgress, hint = 0) => retry(() => fetchBytesOnce(url, onProgress, hint));

// The released policy and world model are not committed: tools/fetch_site_weights.sh downloads them from Hugging Face.
function httpError(url, res) {
  if (res.status === 404 && url.pathname.includes('/assets/policy/')) {
    return new Error(`the demo's model files are missing from this copy of the site (${url.pathname.split('/').pop()}; to fix it: run tools/fetch_site_weights.sh)`);
  }
  return new Error(`${url.pathname}: HTTP ${res.status}`);
}

async function fetchBytesOnce(url, onProgress, hint = 0) {
  const res = await fetch(url);
  if (!res.ok) throw httpError(url, res);
  const encoded = res.headers.get('content-encoding');
  const total = (!encoded && Number(res.headers.get('content-length'))) || hint;
  if (!onProgress || !res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    n += value.length;
    onProgress(n, total);
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

// Inflate a gzip download (a host that already decoded it hands over the plain bytes).
async function gunzip(bytes) {
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const fetchJson = (url) => retry(async () => {
  const res = await fetch(url);
  if (!res.ok) throw httpError(url, res);
  return res.json();
});

// Load timings (ms), for measuring load speed: cold load to ready, and click to the first simulated frame.
const timing = { ready: null, clickToFirstFrame: null };

async function main(hero) {
  // WebGL 2 first: without it nothing below can show, so say so before downloading ~35 MB.
  if (!document.createElement('canvas').getContext('webgl2')) {
    throw new Error('this browser has WebGL 2 turned off or does not support it (the 3D view needs it)');
  }
  const loader = new Loader();
  loader.onProgress = (f) => hero.progress(f);
  loader.onFail = (msg) => hero.fail(msg);
  const step = (key, promise, text) => promise.then((v) => (loader.set(key, text, true), v), (e) => (loader.fail(key, e.message), Promise.reject(e)));

  // Everything starts at once; each row reports its own progress.
  loader.add('libs', 'Renderer · three.js + meshoptimizer');
  const libs = step('libs', Promise.all([load('./render.js', './render.js'), load('./interact.js', './interact.js'), load('meshopt', '../vendor/meshopt/meshopt_decoder.mjs')]), 'ready');
  const bytes = {};
  for (const [key, d] of Object.entries(DOWNLOADS)) {
    loader.add(key, d.label, d.bytes);
    bytes[key] = fetchBytes(d.url, (got, total) => loader.progress(key, got, total), d.bytes).catch((e) => (loader.fail(key, e.message), Promise.reject(e)));
    bytes[key].catch(() => {}); // a failure is shown by the loader; main() still sees the rejection when it awaits
  }
  const meta = fetchJson(new URL('policy/meta.json', ASSETS));
  const index = fetchJson(new URL('motions/index.json', ASSETS));
  const visual = fetchJson(new URL('robot/visual.json', ASSETS));

  const mujoco = step('mujoco', bytes.mujoco.then(async (gz) => {
    loader.set('mujoco', 'compiling');
    const wasmBinary = await gunzip(gz);
    return (await load('mujoco', '../vendor/mujoco/mujoco.js')).default({ wasmBinary });
  }), 'ready');
  const ort = step('ort', bytes.ort.then(async (gz) => {
    loader.set('ort', 'compiling');
    const wasmBinary = await gunzip(gz);
    const ort = await load('onnxruntime-web', '../vendor/onnxruntime-web/ort.wasm.min.mjs');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = new URL('onnxruntime-web/', VENDOR).href;
    ort.env.wasm.wasmBinary = wasmBinary;
    return ort;
  }), 'ready');

  // Stage 1: physics + meshes -> the robot stands in its keyframe pose while the policy downloads.
  const [[{ View, decodeMeshes }, { Interact }, { MeshoptDecoder }], mj, vis, mjb, meshBytes, m] = await Promise.all([
    libs, mujoco, visual, step('scene', bytes.scene, 'ready'), bytes.meshes, meta,
  ]);
  loader.set('meshes', 'decoding');
  await MeshoptDecoder.ready;
  let sim = Sim.create(mj, mjb); // replaced by the robot switch
  sim.reset();
  const view = new View($('demo-canvas'), sim, vis, decodeMeshes(vis, meshBytes, MeshoptDecoder));
  view.stepDt = m.control_dt;
  $('cap-ckpt').textContent = `The released policy (${{ fp16: '16-bit', fp32: '32-bit' }[m.weight_encoding ?? 'fp16'] ?? m.weight_encoding} weights)`;
  loader.set('meshes', 'ready', true);
  $('loader').classList.add('compact');
  const weight = sim.totalMass() * 9.81;
  const interact = new Interact(view, sim, PUSH_CAP * weight);
  $('ro-force-sub').textContent = `max ${interact.maxForce.toFixed(0)} N · ${PUSH_CAP} × body weight`;

  // inferMs: wall-clock time of one control tick (policy, physics, world model), shown as "Step time".
  let ctl = null, fallAt = null, running = true, paused = false, busy = false, last = null, acc = 0, inferMs = 0, fps = 0, lastHud = 0;
  let clickAt = null; // performance.now() of "Try it live", until the first simulated frame is drawn
  // The world model: `rollout` once loaded; `wmOn` / `realOn` from the World model / MuJoCo toggles.
  // Every `horizon` ticks (training's imagined-rollout length) a new rollout starts from the real state.
  let rollout = null, wmOn = true, realOn = true, horizon = 32, divergence = null, wmFallAt = null;
  // Policy switch: off = MuJoCo's actuators limp, no actions; the world model (only ever trained with the
  // policy in control) pauses until it is back on.
  let policyOn = true;
  let recoverUntil = -1; // sim time until which a fall is not declared (the policy was just switched back on)
  // Divergence at the end of the last full window (the training horizon), shown next to the live value.
  let divEnd = null;
  // The robot switch: a variant the world model was not trained on (stock G1 only).
  let variantActive = false;
  // MuJoCo off is offered only where it is meaningful: world model loaded and shown, policy on, stock G1.
  function updateRealBtn() {
    $('demo-mujoco').disabled = !rollout || !wmOn || !policyOn || variantActive;
  }
  // The target pose: the reference frame being tracked (looping authored clips only).
  let targetOn = true;
  // Push test: a horizontal velocity added to the pelvis (away from the camera); recovered once the pelvis
  // moves slower than 0.1 m/s for 0.5 s (our criterion; the paper gives none), failed on a fall.
  let pushTest = null;
  // Work that must not run during a control tick (e.g. swapping the simulation); run between ticks.
  const idleQueue = [];
  // The interaction hint fades out once the visitor has pushed the robot or turned the view.
  $('demo-canvas').parentElement.addEventListener('pointerdown', () => $('demo-hint').classList.add('used'), { once: true, capture: true });
  const showTarget = (tick, jump = false) => {
    const motion = ctl?.motion;
    const on = targetOn && motion?.rows !== undefined;
    view.target.setVisible(on);
    if (!on) return;
    const f = motion.frame(tick);
    const fk = sim.kinematics(motion.rows.subarray(f * 71, f * 71 + sim.nq), 1);
    view.targetSnapshot(fk.xpos, fk.xquat, jump);
  };
  const bodyErr = new Float64Array(sim.model.nbody);
  const graph = new DivergenceGraph($('div-graph'));
  window.realmimic = { sim, view, interact, timing }; // for debugging from the console (sim / ctl follow the robot switch)

  const resync = () => {
    rollout.sync(sim, ctl);
    const fk = sim.kinematics(rollout.state.subarray(0, sim.nq));
    view.ghostSnapshot(fk.xpos, fk.xquat, true);
    graph.mark();
  };
  const resetReality = () => {
    ctl.reset();
    view.snapshot(true);
    showTarget(ctl.tick, true);
    fallAt = null;
    recoverUntil = -1;
    if (rollout) resync();
  };

  async function ticks() {
    const dt = m.control_dt;
    for (let n = 0; acc >= dt && n < MAX_TICKS_PER_FRAME; n++) {
      const t = performance.now();
      // A training-horizon window ends: the world model restarts from MuJoCo's current state, before either steps,
      // so both then advance the same tick.
      if (rollout && wmOn && policyOn && realOn && rollout.age >= horizon) {
        divEnd = divergence; // error after a full training-horizon window, just before the re-sync
        resync();
      }
      if (realOn) {
        sim.clearForces();
        interact.apply();
        await ctl.step();
        view.snapshot();
        showTarget(ctl.tick);
        if (pushTest) trackPush(sim.data.qvel[0], sim.data.qvel[1], ctl.fallen, dt);
        // Policy back on: it gets RECOVER_TRY seconds from wherever the robot is; a fall counts only if the robot
        // is still off the reference after that. Limp (policy off): no fall, no reset.
        if (policyOn && sim.time < recoverUntil) ctl.fallen = false;
        else if (ctl.fallen && fallAt === null && policyOn) fallAt = sim.time;
        if (fallAt !== null && sim.time - fallAt > RESET_AFTER_FALL) resetReality();
        // Past its end a clip's reference is its last frame held still (training's freeze_clip_end): the robot
        // settles into the final pose, holds it, then the clip starts again.
        else if (policyOn && sim.time >= recoverUntil && ctl.motion.loop === false && ctl.tick > ctl.motion.length + REPLAY_AFTER_END * FPS) resetReality();
      }
      if (rollout && wmOn && policyOn) {
        // In step with MuJoCo: a reset during this tick (fall, replay) re-synced the rollout to the new state, which
        // MuJoCo has not stepped from yet.
        if (!realOn || rollout.tick !== ctl.tick) await rollout.step(ctl.motion);
        if (realOn) {
          divergence = rollout.divergence(sim, bodyErr); // also sets rollout.fk (its body poses)
          graph.push(divergence);
        } else {
          rollout.fk = sim.kinematics(rollout.state.subarray(0, sim.nq));
          showTarget(rollout.tick);
          sim.time += dt;
          if (pushTest) trackPush(rollout.state[sim.nq], rollout.state[sim.nq + 1], rollout.fallen(m.termination), dt);
          if (rollout.fallen(m.termination) && wmFallAt === null) wmFallAt = sim.time;
          if (wmFallAt !== null && sim.time - wmFallAt > RESET_AFTER_FALL) {
            ctl.reset();
            resync();
            wmFallAt = null;
          }
        }
        if (realOn) view.ghostSnapshot(rollout.fk.xpos, rollout.fk.xquat);
        else view.robot.snapshot(rollout.fk.xpos, rollout.fk.xquat); // alone: the world model is the solid robot
      }
      inferMs = 0.9 * inferMs + 0.1 * (performance.now() - t);
      acc -= dt;
    }
    acc = Math.min(acc, dt); // whatever could not be simulated this frame is dropped
  }

  // Push test outcome from the robot that was pushed: recovered once its pelvis moves slower than 0.1 m/s
  // for 0.5 s, failed on a fall.
  function trackPush(vx, vy, fallen, dt) {
    pushTest.calm = Math.hypot(vx, vy) < 0.1 ? pushTest.calm + dt : 0;
    pushTest.age += dt;
    if (fallen) {
      $('ro-push-result').textContent = `${pushTest.v} m/s: fell`;
      pushTest = null;
    } else if (pushTest.calm >= 0.5) {
      $('ro-push-result').textContent = `${pushTest.v} m/s: recovered in ${(pushTest.age - 0.5).toFixed(1)} s`;
      pushTest = null;
    }
  }

  function hud(now) {
    const f = interact.force, cap = interact.maxForce;
    $('force-hud').classList.toggle('active', f > 0);
    $('force-hud').classList.toggle('capped', f >= cap - 0.5);
    $('ro-force').textContent = f.toFixed(0);
    $('force-fill').style.width = `${(100 * f) / cap}%`;
    if (rollout && wmOn) graph.draw();
    if (now - lastHud < 100) return; // text readouts at 10 Hz
    lastHud = now;
    $('ro-time').textContent = `${sim.time.toFixed(1)} s`;
    $('ro-infer').textContent = ctl ? `${inferMs.toFixed(1)} ms` : '--';
    $('ro-fps').firstChild.textContent = fps ? fps.toFixed(0) : '--';
    if (rollout && wmOn) {
      const d = realOn && divergence !== null ? 100 * divergence : null;
      $('ro-div').textContent = d === null ? '--' : d.toFixed(1);
      $('ro-div-end').textContent = realOn && divEnd !== null ? (100 * divEnd).toFixed(1) : '--';
    }
    const fell = realOn ? fallAt !== null : wmFallAt !== null;
    const trying = realOn && policyOn && sim.time < recoverUntil && ctl.terminated(ctl.window);
    const banner = paused ? 'Paused' : fell ? 'Fall detected // resetting' : trying ? 'Policy back on // recovering' : '';
    $('demo-banner').hidden = !banner;
    $('demo-banner').textContent = banner;
  }

  function frame(now) {
    requestAnimationFrame(frame);
    if (hero.active) { // nothing to draw under the montage
      last = null;
      return;
    }
    if (last !== null) fps = 0.95 * fps + 0.05 * (1000 / Math.max(now - last, 1));
    if (ctl && running && !paused && last !== null) acc += Math.min((now - last) / 1000, 0.1);
    last = now;
    while (!busy && idleQueue.length) idleQueue.shift()();
    if (ctl && !busy && running && !paused) {
      busy = true;
      ticks().catch((e) => {
        running = false;
        $('demo-banner').hidden = false;
        $('demo-banner').textContent = `Error: ${e.message}`;
        console.error(e);
      }).finally(() => (busy = false));
    }
    view.update(now);
    interact.update();
    view.render();
    hud(now);
    if (clickAt !== null && ctl?.tick > 0) {
      timing.clickToFirstFrame = Math.round(now - clickAt);
      clickAt = null;
    }
  }
  requestAnimationFrame(frame);

  // Stage 2: the policy.
  const [o, onnx, idx] = await Promise.all([ort, bytes.policy, index]);
  loader.set('policy', 'compiling');
  const policy = await Policy.create(o, onnx, m);
  loader.set('policy', `ready · iteration ${Number(m.training_iteration).toLocaleString('en')}`, true);
  const controller = new Controller(sim, policy, m);
  const select = $('demo-motion');
  // index.json lists the motions group by group; each group starts with its name as a disabled option (an optgroup
  // would indent the items in the browser's menu).
  let group = null;
  for (const entry of idx.motions) {
    if (entry.group && entry.group !== group) {
      const head = new Option(entry.group, '');
      Object.assign(head, { disabled: true, className: 'group' });
      select.add(head);
    }
    group = entry.group ?? null;
    select.add(new Option(entry.name, entry.id));
  }
  const useMotion = (motion) => {
    const c = ctl ?? controller;
    c.setMotion(motion);
    c.policyOff = !policyOn;
    view.snapshot(true);
    ctl ??= controller;
    showTarget(c.tick, true);
    $('demo-target').disabled = c.motion.rows === undefined;
    fallAt = null;
    if (rollout) resync();
  };
  const setMotion = async (id) => {
    if (id === 'upload') return;
    const entry = idx.motions.find((x) => x.id === id);
    useMotion(await loadMotion(entry, m, (f) => fetchBytes(new URL(f, new URL('motions/', ASSETS)))));
  };
  await setMotion((idx.motions.find((x) => x.default) ?? idx.motions[0]).id);
  select.value = (idx.motions.find((x) => x.default) ?? idx.motions[0]).id;
  select.addEventListener('change', () => setMotion(select.value));
  // Load your own motion: converted in the browser, nothing is sent anywhere.
  const fileInput = $('demo-file');
  fileInput.disabled = false;
  fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    try {
      const { motion, format, seconds } = await motionFromFile(file, m.joint_names);
      let option = select.querySelector('option[value="upload"]');
      if (!option) select.add((option = new Option('', 'upload')));
      option.textContent = `Your file: ${file.name}`;
      select.value = 'upload';
      useMotion(motion);
      $('upload-note').textContent = `${file.name}: ${format}, ${seconds.toFixed(1)} s. Tracked by the policy in your browser; nothing is uploaded.`;
    } catch (e) {
      $('upload-note').textContent = `Could not read ${file.name}: ${e.message}`;
    } finally {
      fileInput.value = '';
    }
  });
  // Policy switch (see policyOn): off makes MuJoCo's robot limp and pauses the world model, which runs only
  // with the policy (MuJoCo is switched on if it was off); on hands control back and re-syncs the world model.
  const policyBtn = $('demo-policy');
  policyBtn.disabled = false;
  policyBtn.addEventListener('click', () => {
    policyOn = !policyOn;
    policyBtn.setAttribute('aria-checked', String(policyOn));
    policyBtn.querySelector('.state').textContent = policyOn ? 'on' : 'off';
    if (!policyOn && !realOn) setReality(true);
    sim.setActuatorsEnabled(policyOn);
    ctl.policyOff = !policyOn;
    fallAt = null;
    ctl.fallen = false;
    recoverUntil = policyOn ? sim.time + RECOVER_TRY : -1;
    updateRealBtn();
    if (rollout) {
      view.ghost.setVisible(policyOn && wmOn);
      if (policyOn && wmOn) resync();
    }
    $('wm-note').hidden = policyOn;
    $('wm-note').textContent = policyOn ? '' : 'World model paused: it was trained with the policy in control.';
  });
  $('demo-target').addEventListener('click', () => {
    targetOn = !targetOn;
    $('demo-target').setAttribute('aria-checked', String(targetOn));
    $('demo-target').querySelector('.state').textContent = targetOn ? 'on' : 'off';
    showTarget(ctl.tick, true);
  });
  for (const btn of document.querySelectorAll('.push-buttons button')) {
    btn.disabled = false;
    btn.addEventListener('click', () => {
      const v = Number(btn.dataset.v);
      const p = view.controls.target, c = view.camera.position;
      const dx = p.x - c.x, dy = p.y - c.y, n = Math.hypot(dx, dy) || 1;
      if (realOn) {
        sim.data.qvel[0] += (v * dx) / n;
        sim.data.qvel[1] += (v * dy) / n;
        // The world model restarts from the pushed state: what follows is its prediction of the recovery.
        if (rollout && wmOn && policyOn) resync();
      } else {
        // World model alone: the push is an edit of its state (it has no force input).
        rollout.state[sim.nq] += (v * dx) / n;
        rollout.state[sim.nq + 1] += (v * dy) / n;
      }
      pushTest = { v, age: 0, calm: 0 };
      $('ro-push-result').textContent = `${v} m/s: recovering…`;
    });
  }
  const pauseBtn = $('demo-pause');
  pauseBtn.addEventListener('click', () => {
    paused = !paused;
    pauseBtn.textContent = paused ? 'Resume' : 'Pause';
    pauseBtn.setAttribute('aria-pressed', String(paused));
  });
  $('demo-reset').addEventListener('click', () => {
    if (!realOn) setReality(true);
    resetReality();
  });
  document.addEventListener('visibilitychange', () => {
    running = !document.hidden;
    last = null;
  });
  ctl = controller;
  window.realmimic.ctl = ctl;
  $('loader').classList.add('done');

  // Warm up under the montage: the first inference (ONNX Runtime allocates on its first run) and a first
  // render (shader compilation), then rewind to the motion's start.
  await ctl.step();
  ctl.reset();
  view.snapshot(true);
  showTarget(ctl.tick, true);
  view.update(performance.now());
  view.render();
  // "Try it live": the motion from its first frame, seen from the demo's default camera (the montage's view).
  hero.onStart = () => {
    clickAt = performance.now();
    resetReality();
    const p = 3 * view.pelvis, x = sim.data.xpos;
    view.controls.target.set(x[p], x[p + 1], x[p + 2]);
    view.camera.position.copy(view.controls.target).add(view.homeOffset);
    view._pelvis.copy(view.controls.target);
    view.lastFrame = null;
    last = null;
  };
  timing.ready = Math.round(performance.now());
  hero.ready();

  // Robot switch: the same policy on a robot variant (its own physics, stand-in shapes for parts without
  // meshes); floor friction for the feet and the floor (the world model is told the same value).
  const robots = { g1: { mjb, variant: null } };
  let friction = 1;
  const robotSel = $('demo-robot'), frictionSel = $('demo-friction');
  robotSel.disabled = frictionSel.disabled = false;
  let robotId = robotSel.value; // the robot running now (the menu goes back to it if a switch fails)
  robotSel.addEventListener('change', async () => {
    const id = robotSel.value;
    robotSel.disabled = true;
    try {
      if (!robots[id]) {
        const base = new URL(`robot/variants/${id}/`, ASSETS);
        const [b, v] = await Promise.all([fetchBytes(new URL('scene.mjb', base)), fetchJson(new URL('variant.json', base))]);
        const geometries = v.geoms?.length ? decodeMeshes(v, await fetchBytes(new URL('meshes.bin', base)), MeshoptDecoder) : [];
        robots[id] = { mjb: b, variant: v, geometries };
      }
      await new Promise((resolve) => idleQueue.push(resolve));
      const next = Sim.create(mj, robots[id].mjb);
      next.setFriction(friction);
      const c = new Controller(next, policy, m);
      c.setMotion(ctl.motion);
      c.policyOff = !policyOn;
      next.setActuatorsEnabled(policyOn);
      sim = next;
      ctl = c;
      view.sim = interact.sim = next;
      interact.setMaxForce(PUSH_CAP * next.totalMass() * 9.81);
      $('ro-force-sub').textContent = `max ${interact.maxForce.toFixed(0)} N · ${PUSH_CAP} × body weight`;
      view.setVariant(robots[id].variant, robots[id].geometries);
      view.snapshot(true);
      showTarget(ctl.tick, true);
      fallAt = wmFallAt = null;
      recoverUntil = -1;
      variantActive = Boolean(robots[id].variant);
      if (!realOn) setReality(true);
      updateRealBtn();
      if (rollout) resync();
      Object.assign(window.realmimic, { sim, ctl });
      $('setup-note').textContent = robots[id].variant
        ? `${robots[id].variant.mass.toFixed(1)} kg with BrainCo hands (Unitree's model, fingers open). Same policy, trained on the stock G1; the world model was trained on the stock G1 too.`
        : 'Same policy on every robot.';
      robotId = id;
    } catch (e) {
      robotSel.value = robotId;
      $('setup-note').textContent = `Could not load that robot (${e.message}).`;
      console.error(e);
    } finally {
      robotSel.disabled = false;
    }
  });
  // Foot friction, live while dragging. Training drew it from 0.3-1.2 per episode; the benchmark uses 1.0.
  frictionSel.addEventListener('input', () => {
    friction = Number(frictionSel.value);
    sim.setFriction(friction);
    if (rollout) rollout.friction[0] = friction;
    $('friction-val').textContent = friction.toFixed(2);
    const outside = friction < 0.3 || friction > 1.2;
    $('friction-note').textContent = outside ? 'outside the training range' : 'training range 0.3–1.2\nbenchmark 1.0';
    $('friction-note').classList.toggle('warn', outside);
  });

  // Stage 3: the world model (~18 MB), downloaded once the visitor has pressed "Try it live" (the demo already
  // runs). If it cannot be loaded, the demo carries on without it and the panel says why.
  const note = (text) => {
    $('wm-note').hidden = !text;
    $('wm-note').textContent = text;
  };
  if (hero.active) {
    await new Promise((resolve) => {
      const start = hero.onStart;
      hero.onStart = () => (start(), resolve());
    });
  }
  let wmSession, wmMeta;
  try {
    note('Loading the world model…');
    const wmBytes = await fetchBytes(new URL('policy/world_model.onnx', ASSETS), (got, total) => {
      note(`Loading the world model… ${total ? Math.floor((100 * got) / total) : MB(got)}${total ? ' %' : ' MB'}`);
    });
    wmMeta = await fetchJson(new URL('policy/world_model.json', ASSETS));
    wmSession = await o.InferenceSession.create(wmBytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  } catch (e) {
    note(`The world model could not be loaded (${e.message}); the demo runs without it.`);
    console.error(e);
    return;
  }
  rollout = new WorldModelRollout(o, wmSession, policy, m);
  rollout.friction[0] = friction;
  horizon = wmMeta.training_horizon || 32;
  resync();
  view.ghost.setVisible(true);
  note('');
  window.realmimic.rollout = rollout;

  const wmBtn = $('demo-wm'), realBtn = $('demo-mujoco');
  const setSwitch = (btn, on) => {
    btn.setAttribute('aria-checked', String(on));
    btn.querySelector('.state').textContent = on ? 'on' : 'off';
  };
  wmBtn.disabled = false;
  updateRealBtn();
  function setReality(on) {
    if (on === realOn) return;
    realOn = on;
    setSwitch(realBtn, on);
    if (on) {
      // Reality takes over from the world model: same state, same policy history, same motion tick.
      sim.setState(Float64Array.from(rollout.state));
      ctl.policy.states.set(rollout.policy.states);
      ctl.policy.actions.set(rollout.policy.actions);
      ctl.tick = rollout.tick;
      ctl.fallen = false;
      view.snapshot(true);
      resync();
      note('');
    } else {
      wmFallAt = null;
      divergence = divEnd = null;
      $('ro-div').style.color = '';
      view.robot.snapshot(rollout.fk.xpos, rollout.fk.xquat, true);
      note('MuJoCo off: the world model runs the robot alone. The push test edits its state; dragging needs MuJoCo.');
    }
    view.ghost.setVisible(on && wmOn && policyOn);
    $('wm-hud').classList.toggle('solo', !on);
    $('force-hud').classList.toggle('nodrag', !on);
    interact.enabled = on;
  }
  realBtn.addEventListener('click', () => setReality(!realOn));
  wmBtn.addEventListener('click', () => {
    wmOn = !wmOn;
    setSwitch(wmBtn, wmOn);
    if (!wmOn) setReality(true);
    else resync();
    updateRealBtn();
    view.ghost.setVisible(wmOn && policyOn && realOn);
    $('wm-body').hidden = !wmOn;
  });
}

// Scrolling plot of the last 10 s of world-model divergence (cm) on a fixed 0-10 cm scale (taller values
// are clipped), area shaded blue (agrees) to red; rollout starts as ticks on the time axis.
class DivergenceGraph {
  constructor(canvas, samples = 500, fullCm = 10) {
    this.canvas = canvas;
    this.v = new Float32Array(samples);
    this.marks = new Uint8Array(samples);
    this.n = 0;
    this.fullCm = fullCm;
  }

  push(x) {
    this.v.copyWithin(0, 1);
    this.marks.copyWithin(0, 1);
    this.v[this.v.length - 1] = x * 100;
    this.marks[this.marks.length - 1] = this.pendingMark ? 1 : 0;
    this.pendingMark = false;
    this.n = Math.min(this.n + 1, this.v.length);
  }

  mark() {
    this.pendingMark = true;
  }

  draw() {
    const c = this.canvas, dpr = Math.min(window.devicePixelRatio, 2);
    const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
    if (!w || !h) return;
    if (c.width !== w || c.height !== h) Object.assign(c, { width: w, height: h });
    const g = c.getContext('2d');
    g.clearRect(0, 0, w, h);
    const N = this.v.length, top = this.fullCm;
    const x = (i) => (i / (N - 1)) * w, y = (v) => h - 2 - (Math.min(v, top) / top) * (h - 6);
    const grad = g.createLinearGradient(0, h, 0, y(100 * DIVERGENCE_FULL));
    grad.addColorStop(0, 'rgba(61, 139, 255, 0.15)');
    grad.addColorStop(1, 'rgba(61, 139, 255, 0.6)');
    g.beginPath();
    g.moveTo(x(N - this.n), h);
    for (let i = N - this.n; i < N; i++) g.lineTo(x(i), y(this.v[i]));
    g.lineTo(x(N - 1), h);
    g.closePath();
    g.fillStyle = grad;
    g.fill();
    g.beginPath();
    for (let i = N - this.n; i < N; i++) (i === N - this.n ? g.moveTo : g.lineTo).call(g, x(i), y(this.v[i]));
    g.strokeStyle = 'rgba(236, 234, 230, 0.9)';
    g.lineWidth = dpr;
    g.stroke();
    g.fillStyle = 'rgba(236, 234, 230, 0.35)';
    g.fillStyle = 'rgba(236, 234, 230, 0.16)';
    for (let i = N - this.n; i < N; i++) if (this.marks[i]) g.fillRect(x(i), 0, dpr, h); // re-syncs
    g.fillStyle = 'rgba(133, 136, 143, 0.9)';
    g.font = `${10 * dpr}px "JetBrains Mono", monospace`;
    g.fillText(`${top} cm`, 2 * dpr, 11 * dpr);
  }
}

// Phones and data-saving connections load the demo when "Try it live" is tapped; everything else loads at
// once, behind the montage.
function needsTap() {
  const c = navigator.connection;
  if (c?.saveData || /2g/.test(c?.effectiveType ?? '')) return true;
  return matchMedia('(pointer: coarse)').matches && matchMedia('(max-width: 720px), (max-height: 520px)').matches; // phones, either way up
}

async function boot() {
  const hero = new Hero($('hero'), { cuesUrl: new URL('hero/cues.json', ASSETS) });
  if (needsTap()) await new Promise((resolve) => (hero.onClick = resolve));
  await main(hero).catch((e) => {
    $('load-note').textContent = `Could not start the demo: ${e.message}`;
    hero.fail(e.message);
    console.error(e);
  });
}

boot();
