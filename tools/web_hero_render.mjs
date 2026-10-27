// Render recorded rollouts (tools/web_hero_record.py) in the site's own three.js scene, headless on the GPU, so
// the hero video looks exactly like the live demo: same robot materials, lights, floor, post-processing and the
// demo's default camera, following the pelvis the same way.
//
//   node tools/web_hero_render.mjs <page url> <rec dir> <out dir> [--width 1280] [--height 720] [--every 1]
//        [--only 00,03] [--from s] [--to s] [--target] [--target-opacity 0.35]
//
// Writes <out dir>/<NN>/<frame>.png, one frame per 50 Hz step (every `--every` steps). The page is served from
// site/ (python3 tools/web_serve.py site 8001); nothing of the recording reaches the site, only the pixels.
// --target also draws the reference as the demo's white outline (--target-opacity dims it). A recording with a third
// track (web_hero_record.py --world-model) also draws the world model as the demo's blue overlay. Needs Chrome and playwright-core, resolved from
// the working directory (run it from a directory where `npm i playwright-core` was done).
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

const { chromium } = createRequire(path.join(process.cwd(), 'noop.js'))('playwright-core');

const argv = process.argv.slice(2);
const [url, recDir, outDir] = argv;
const opt = (name, dflt) => { const i = argv.indexOf(`--${name}`); return i < 0 ? dflt : argv[i + 1]; };
const W = Number(opt('width', 1280)), H = Number(opt('height', 720)), EVERY = Number(opt('every', 1));
const ONLY = opt('only', null)?.split(','), FROM = Number(opt('from', 0)), TO = Number(opt('to', 1e9));
const TARGET = argv.includes('--target'), TARGET_OPACITY = Number(opt('target-opacity', 1));
const FPS = 50;

const index = JSON.parse(fs.readFileSync(path.join(recDir, 'montage.json'), 'utf8'));
const browser = await chromium.launch({
  executablePath: '/usr/bin/google-chrome',
  args: ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist', '--enable-gpu'],
});
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.error('page error:', e.message));
await page.goto(url);
await page.waitForFunction(() => !!window.realmimic?.ctl, null, { timeout: 120000 });
// Stop the page's own loop (the demo keeps simulating otherwise), make the viewport the whole window, hide the
// HUD, and keep the demo's default camera offset.
await page.evaluate(async ({ target, targetOpacity }) => {
  window.requestAnimationFrame = () => 0;
  await new Promise((r) => setTimeout(r, 200));
  for (const el of document.querySelectorAll('.viewport > :not(canvas)')) el.style.display = 'none';
  const v = document.querySelector('.viewport');
  v.style.cssText = 'position:fixed;inset:0;z-index:999;border:0;border-radius:0;aspect-ratio:auto;max-height:none;margin:0';
  document.body.style.overflow = 'hidden';
  const { view, sim } = window.realmimic;
  for (const slot of [5, 6, 7]) sim.kinematics(new Float64Array(sim.nq), slot); // allocate the FK buffers up front
  view.ghost.setVisible(false);
  view.target.setVisible(target);
  view.outline.uniforms.opacity.value = targetOpacity;
  window.__offset = view.camera.position.clone().sub(view.controls.target);
  // Warm up (shader compilation, environment map): the first frames otherwise come out black.
  for (let i = 0; i < 20; i++) {
    view.render();
    await new Promise((r) => setTimeout(r, 50));
  }
}, { target: TARGET, targetOpacity: TARGET_OPACITY });

for (const entry of index) {
  const id = entry.file.replace('.f32', '');
  if (ONLY && !ONLY.includes(id)) continue;
  const buf = fs.readFileSync(path.join(recDir, entry.file));
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const nq = entry.nq, a = Math.max(0, Math.round(FROM * FPS)), b = Math.min(entry.frames, Math.round(TO * FPS));
  const dir = path.join(outDir, id);
  fs.mkdirSync(dir, { recursive: true });
  for (let f = a; f < b; f += EVERY) {
    const n = entry.tracks ?? 2;
    const q = Array.from(all.subarray(f * n * nq, (f + 1) * n * nq));
    const png = await page.evaluate(({ q, nq, n, first, dt }) => {
      const { sim, view } = window.realmimic;
      // Read the robot's poses before the next kinematics call: allocating its buffer can grow the WASM memory,
      // which detaches the views returned here.
      const robot = sim.kinematics(Float64Array.from(q.slice(0, nq)), 5);
      view.robot.snapshot(robot.xpos, robot.xquat, true);
      const p = 3 * sim.bodyId('pelvis'), pelvis = [robot.xpos[p], robot.xpos[p + 1], robot.xpos[p + 2]];
      const ref = sim.kinematics(Float64Array.from(q.slice(nq, 2 * nq)), 6);
      view.targetSnapshot(ref.xpos, ref.xquat, true);
      view.ghost.setVisible(n > 2);
      if (n > 2) {
        const wm = sim.kinematics(Float64Array.from(q.slice(2 * nq, 3 * nq)), 7);
        view.ghostSnapshot(wm.xpos, wm.xquat, true);
      }
      if (first) { // the demo's default view of the pelvis, as when a motion starts
        view.controls.target.set(...pelvis);
        view.camera.position.copy(view.controls.target).add(window.__offset);
        view._pelvis.copy(view.controls.target);
        view.lastFrame = null;
        window.__t = 0;
      }
      window.__t += dt;
      view.update(window.__t * 1000);
      view.render();
      return view.canvas.toDataURL('image/png');
    }, { q, nq, n, first: f === a, dt: EVERY / FPS });
    fs.writeFileSync(path.join(dir, `${String(f).padStart(4, '0')}.png`), Buffer.from(png.split(',')[1], 'base64'));
  }
  console.log(`${id} ${entry.caption}: frames ${a}-${b}`);
}
await browser.close();
