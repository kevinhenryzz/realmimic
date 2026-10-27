# Project website

A static page with a live demo: the policy runs in the browser (MuJoCo WASM physics, ONNX Runtime Web
inference, three.js rendering; no build step). On every push to main, `.github/workflows/pages.yml` fetches the
libraries and the model files (checksummed) and publishes `site/`. Locally, from the repo root:

```bash
tools/fetch_site_vendor.sh            # pinned libraries into site/vendor/, SHA-256 checked
tools/fetch_site_weights.sh           # the released policy and world model (web/ of huggingface.co/kevinhenryz/realmimic)
                                      # into site/assets/policy/, SHA-256 checked (HF_TOKEN while the repository is
                                      # private); or `tools/fetch_site_weights.sh <dir>` from a local copy of web/
python3 tools/web_serve.py site 8000  # http://localhost:8000
```

Both scripts need only bash, curl, gzip and sha256sum (the weights script uses the hf CLI instead of curl if it is
installed).

- Not committed: `site/vendor/` (except `vendor/README.md`, the checksum list) and `site/assets/policy/` (the released
  exports, CC BY-NC 4.0 like the weights). Without the exports the page says so ("run tools/fetch_site_weights.sh").
  They are reproduced from the released checkpoint by `tools/export_web_policy.py <ckpt> site/assets/policy --weights
  fp16`, then `tools/export_web_world_model.py <ckpt> site` (the same bytes as the published files).
- Committed, generated from code: `site/assets/robot/` (scene, meshes, the BrainCo variant; no weights), by
  `uv run --with meshoptimizer==0.2.30a0 python tools/export_web_scene.py site/assets/robot` and, for the BrainCo
  variant, `... tools/export_web_scene.py site/assets/robot/variants/g1_brainco --variant g1_brainco --hand-urdf
  <unitree_ros @15ebe53>/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf` (mjlab's G1 model, Unitree's
  BrainCo hand model).
- Committed: the page, `demo/`, the demo motions (`assets/motions/`), the figures and results data, the videos and the
  montage, the RTX 4090 model. Figures and results are redrawn with
  `python3 tools/web_figures.py site/assets/figures --runs runs/benchmark --web runs/web --data site/assets/results`
  (`--end-of-clip <run>="no fix, run A" <run>="no fix, run B" <run>="with the fix"`), the motions checked with
  `uv run python tools/web_author_motions.py site --check`.
- The demo's reference indexing (tick k targets clip frame k + 1, as training and the benchmark) is tested by
  `tests/realmimic/test_web_reference_indexing.py`.

Retargeted motion capture (CMU, 100STYLE) for the motion menu, checked with the site's policy in CPU MuJoCo
(no fall, joint error <= 0.16 rad, pelvis drift <= 0.35 m mean / 0.7 m max; `--robust 5` adds starting-pose
noise and a mid-clip push), then installed into `assets/motions/` with per-clip source and licence:

```bash
uv run python tools/web_retarget.py site <scratch> --cmu <dir of NN.asf / NN_TT.amc> --style <dir of 100STYLE .bvh> \
    --clips clips.json --check --robust 5 [--install <ids> --default <id>]
```

Opening montage (`assets/hero/`): the released policy on benchmark clips it completes, rendered in this
page's own scene with the demo's default camera, so the video and the live canvas are the same view.
Record (GPU; BONES-derived trajectories stay local and are deleted after rendering), render (headless
Chrome on the GPU, from a directory with `playwright-core`), encode:

```bash
uv run python tools/web_hero_record.py --ckpt logs/release/ckpt_final_86000.pt --npz-dir data/npz/bones_seed --montage montage.json --out <scratch>/rec
node tools/web_hero_render.mjs http://127.0.0.1:8000/ <scratch>/rec/ours <scratch>/frames --only NN --from s --to s   # per montage segment
ffmpeg -framerate 50 -i <scratch>/seq/%05d.png -vf fps=30 -c:v libx264 -preset veryslow -crf 26 -pix_fmt yuv420p -g 60 -movflags +faststart -an site/assets/hero/hero.mp4
```

Videos are H.264 only (hardware-decoded almost everywhere; AV1 decodes in software on many machines and
stuttered), at 30 fps with a keyframe every 2 s and faststart.

World model in the videos: the montage and our pane of every comparison come from the browser demo's own loop
(`tools/web_hero_record.py --world-model site`: CPU MuJoCo, the site's policy and world model, re-synced every 32
ticks), and the world model is drawn as the demo's blue overlay (`web_hero_render.mjs` draws a recording's third
track). SONIC's pane is the benchmark rollout as before. Every clip shown completes in that loop.

Result videos (`assets/videos/NN.mp4`, in Results): the released checkpoint and SONIC's release policy (its
own gains) on the same clip (an excerpt around the action), rendered in this page's scene like the montage
(one pane each, 640x720, the demo's camera, the reference as a dimmed outline) and put side by side. Six clips in a
3 x 2 grid (a sideways-scrolling row on phones); no real-robot video in v1. The pane
labels and their "off track at" / "fell at" markers are HTML over the video, from `assets/videos/cues.json`
(generic captions and event times only; off track = the benchmark's failure, fell = render_rollouts' fall test).
The six are BONES-SEED clips of different kinds (flip with a hero landing, 360 flip, vogue dance drop, back-kick
dance move, high jump, broad jump), each one SONIC's release fails in the benchmark and on camera while ours completes
it, picked among such clips for staying close to the reference outline (the policy has no root-position goal, so
travelling motions drift).

```bash
uv run python tools/web_hero_record.py --ckpt logs/release/ckpt_final_86000.pt --npz-dir <dir> --montage compare.json \
    --out <scratch>/rec --sonic-release <SONIC release policy dir>      # rec/ours, rec/sonic, rec/events.json
node tools/web_hero_render.mjs http://127.0.0.1:8000/ <scratch>/rec/ours <scratch>/frames/ours --width 640 --height 720 --target --target-opacity 0.35
node tools/web_hero_render.mjs http://127.0.0.1:8000/ <scratch>/rec/sonic <scratch>/frames/sonic --width 640 --height 720 --target --target-opacity 0.35
ffmpeg -framerate 50 -i <scratch>/frames/ours/NN/%04d.png -framerate 50 -i <scratch>/frames/sonic/NN/%04d.png \
    -filter_complex "[0]crop=638:720:1:0,pad=640:720:0:0:color=0x0b0c0e[a];[1]crop=638:720:1:0,pad=640:720:2:0:color=0x0b0c0e[b];[a][b]hstack,fps=30,scale=960:540:flags=lanczos" \
    -c:v libx264 -preset veryslow -crf 28 -pix_fmt yuv420p -g 60 -movflags +faststart -an site/assets/videos/NN.mp4
```

Delete the recordings after rendering (they are derived from BONES-SEED); only the pixels and the cue files ship.

`assets/hero/cues.json` holds the segment times and their generic captions ("360 flip"): never clip names.
The page credits Bones Studio on the video. Behind the montage the demo downloads and initialises (first
inference and first render included); "Try it live" crossfades to the canvas, or shows the loading progress
and switches when done. Phones and Save-Data connections start loading on that tap.

Validation (the site's own JS against Python CPU MuJoCo):

```bash
uv run python tools/export_web_policy.py logs/release/ckpt_final_86000.pt site/assets/policy --weights fp16 --vectors exports/web/vectors.json
uv run python tools/web_closed_loop.py site exports/web/trace.json
uv run python tools/export_web_world_model.py logs/<run>/ckpt_XXXXXXX.pt site --vectors exports/web/wm_vectors.json
node tools/web_validate.mjs site exports/web/trace.json exports/web/vectors.json exports/web/wm_vectors.json
WEB_POLICY_CHECKPOINT=logs/<run>/ckpt_XXXXXXX.pt uv run pytest tests/realmimic/test_web_policy.py
```

Weights: the released policy (iteration 86,000) and its world model ship as fp16 with float32 compute. int8
(one scale per output channel) tracked like fp32 for an earlier run but fell once in 8 directions at a 2 m/s
push with the released one,
and doubled the world model's median drift after a push.

Results (section 02 Results with What we found, and the compute table in 05) are filled by `demo/results.js` from
`assets/results/results.json` and `groups.json`, which `tools/web_figures.py --web runs/web` writes from the project's
aggregate results, pruned to the keys the page shows (`PUBLIC` in that script).

Demo behaviour:

- MuJoCo runs the main (solid) robot. Control tick k targets clip frame k + 1, as training and the benchmark
  (`tests/realmimic/test_web_reference_indexing.py`). The world model (`world_model.onnx`, fp16, ~17.7 MB) is
  downloaded once the visitor presses "Try it live"; if it fails, the demo runs without it and the panel says why. It
  runs the same policy as a translucent blue overlay, re-synced to MuJoCo every training horizon (32 ticks, 0.64 s)
  before either steps, so both advance the same ticks. Divergence = mean position error of the tracked bodies, shown
  live and at the end of each 0.64 s window; re-syncs are lines on the graph. The overlay is one blue throughout. It
  gets the foot-friction slider's value (0.1-1.5; training drew 0.3-1.2) as its friction input; the slider changes the
  foot geoms only (they have contact priority over the floor, which stays at 1.0).
- Start of a clip: the robot starts on frame 0, its joints clipped into the soft limits (0.9 of each range) like the
  benchmark's reset (matters for uploads). The page checks for WebGL 2 before downloading anything; any start-up
  failure is shown on the video ("Could not start the demo: …").
- Pushes and drags: a drag (spring force, capped at 0.5 x body weight) is not visible to the world model (no force
  input); the push test (0.5-4 m/s added to the pelvis velocity; recovered = pelvis slower than 0.1 m/s for 0.5 s)
  re-syncs the world model to the pushed state. Both are outside its training data (no pushes in training), which
  the caption under the demo says once; a note in the panel that switched with every drag was removed (it flickered).
  On the BrainCo robot (the world model was trained on the stock G1) world-model-alone is not offered.
- MuJoCo off: the world model runs the robot alone, drawn solid; no divergence is shown;
  dragging is disabled; the push test edits the world model's state.
- Policy switch: off sets MuJoCo's actuator stiffness and damping to 0 (the robot collapses; no auto-reset) and
  pauses the world model, which was only trained with the policy in control (switching MuJoCo on if it was off).
  A limp world model would need a no-control input and a fine-tune on zero-gain transitions (optional v1.1).
  Back on, the policy gets 3 s from wherever the robot is before a fall is declared (banner "Policy back on //
  recovering"): from a crouch (pelvis down to ~0.16 m, about 0.5 s limp) it stands back up in ~0.45 s; lying
  flat it does not get up even in 10 s (training ends an episode at a fall, so it never learned to), and the
  robot is reset.
- Clips that play once: past the end the goal window is the last frame held still with zero velocities
  (training's freeze_clip_end, `ClipMotion.window`); after a 2 s hold the clip restarts.
  Clips retargeted from walks get a short stop appended (`web_retarget.py --settle`), so the held pose is one
  the robot can stand in. Default motion: Elated walk (100STYLE).
- Comparison videos (in Results): each starts from its beginning when half of it comes on screen, loops while there and
  pauses off screen; it is downloaded about a screen before it is reached;
  reduced-motion and Save-Data visitors get the browser controls and start them themselves.

RTX 4090 beside the title (wide screens, `demo/gpu.js`): `assets/gpu/rtx4090.glb` (~1 MB) from "GeForce RTX 4090 Founders
Edition" by exéla (https://skfb.ly/oyBLN), credit kept in the card and in Credits:

> site/assets/gpu/rtx4090.glb is CC BY-NC 4.0 by exéla (https://skfb.ly/oyBLN), modified (lettering painted out, label
> turned upright, compressed); it is not covered by the repository's Apache-2.0 licence.
>
> NVIDIA, GeForce and RTX are trademarks of NVIDIA Corporation. This project is not affiliated with or endorsed by NVIDIA.

 `tools/web_gpu_model.mjs` paints out the brand lettering ("GEFORCE RTX" on the side, the marks
on the bracket), turns "RTX 4090" the right way up, drops unused UV sets and tangents, simplifies, quantizes,
meshopt-compresses and writes WebP textures (14 MB to 1.07 MB). It faces the viewer; a drag tilts it within limits that
keep "RTX 4090" readable (sideways only on touch screens, so vertical swipes still scroll) and it eases back when let
go. It loads after the page has loaded (the demo's downloads first), on wide screens only, draws only while visible,
is skipped on Save-Data, and its fans stay still under reduced motion.

Motions: only clips whose licence allows redistribution go in `assets/motions/` (each entry in
`index.json` records its source and licence): motions authored from the robot's default pose, and CMU /
100STYLE clips retargeted by `tools/web_retarget.py`. BONES-SEED and LAFAN1 motions appear as videos only.

## Launch checklist

- The Weights button and the credits link point at huggingface.co/kevinhenryz/realmimic: public on release day.
- `tools/fetch_site_weights.sh` downloads from that repository; its SHA-256 list matches the released web/ files.
- The demo runs the released checkpoint (iteration 86,000, fp16); the caption reads iteration and weight
  encoding from `assets/policy/meta.json`.
- Refresh results and figures: `python3 tools/web_figures.py site/assets/figures --runs runs/benchmark --web runs/web --data site/assets/results`.
- Stills (`assets/hero/poster.jpg`, the montage's first frame; `og.jpg`, a screenshot of the page with the
  HUD hidden): re-shoot if the look changes. Link previews point at https://kevinhenryzz.github.io/realmimic.
- Fonts and libraries are fetched by `tools/fetch_site_vendor.sh` (not committed), which also writes the
  gzipped WASM copies the page downloads.
