// The RTX 4090 beside the title (assets/gpu/rtx4090.glb, ~1 MB, built by tools/web_gpu_model.mjs), lit like the
// demo's scene: facing the viewer, "RTX 4090" readable; drag to tilt it within limits that keep that face in view, it
// eases back to the front when let go. Until the model is loaded, a still of the same view stands in (style.css,
// assets/gpu/rtx4090.webp); the model takes its place without a visible change and its fans spin up from rest.
// Wide screens only; drawn only while on screen; the still alone on Save-Data connections; fans still under reduced
// motion.

const box = document.getElementById('gpu-card');
const canvas = document.getElementById('gpu-canvas');
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const FRONT = { yaw: 0.12, pitch: 0.1 }; // rad: nearly square to the camera, "RTX 4090" facing it
const LIMIT = { yaw: 0.5, pitch: 0.3 }; // rad from FRONT the drag may tilt it (the face stays readable)
const DRAG = 0.006; // rad per pixel
const SETTLE = 0.6; // s, time constant of the ease back to FRONT after a drag
const SPIN_UP = 3; // s for the fans to reach full speed from rest

async function start() {
  const [THREE, { GLTFLoader }, { RoomEnvironment }, { MeshoptDecoder }] = await Promise.all([
    import('three'), import('three/addons/loaders/GLTFLoader.js'), import('three/addons/environments/RoomEnvironment.js'), import('meshopt'),
  ]);
  const r = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'low-power' });
  r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  r.outputColorSpace = THREE.SRGBColorSpace;
  r.toneMapping = THREE.ACESFilmicToneMapping;
  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(r);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.45;
  pmrem.dispose();
  // The demo's lights: warm key, cool fill, orange rim back left, white rim back right (y-up here).
  for (const [color, intensity, pos] of [['#fff1e2', 2.0, [2, 4, 3]], ['#c9d6ff', 0.3, [-4, 1, 3]], ['#ff6a1a', 3.2, [-3, 2, -3.5]], ['#ffffff', 1.2, [3.5, 2.5, -3]]]) {
    const light = new THREE.DirectionalLight(color, intensity);
    light.position.set(...pos);
    scene.add(light);
  }
  const cam = new THREE.PerspectiveCamera(30, 1, 0.01, 50);

  const gltf = await new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).loadAsync(new URL('../assets/gpu/rtx4090.glb', import.meta.url).href);
  const model = gltf.scene;
  // The download stands the card on its end (length along y, height along z, thickness along x): lay it down,
  // length across the screen, height up, fans towards the camera.
  model.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1));
  model.updateMatrixWorld(true);
  // Centre it and scale its longest side to 1.
  const bounds = new THREE.Box3().setFromObject(model);
  const size = bounds.getSize(new THREE.Vector3()), centre = bounds.getCenter(new THREE.Vector3());
  const s = 1 / Math.max(size.x, size.y, size.z);
  model.position.copy(centre).multiplyScalar(-s);
  model.scale.setScalar(s);
  const turntable = new THREE.Group();
  turntable.add(model);
  scene.add(turntable);
  // Camera on a fixed line of sight; far enough back that the card (1 long), tilted to a drag limit, fits the width.
  const view = new THREE.Vector3(0, 0.25, 1.5).normalize();
  const place = () => {
    const dist = Math.max(1.5, 0.66 / (Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)) * cam.aspect));
    cam.position.copy(view).multiplyScalar(dist);
    cam.lookAt(0, -0.02, 0);
  };
  const mixer = new THREE.AnimationMixer(model);
  if (!reduced) for (const clip of gltf.animations) mixer.clipAction(clip).play(); // the fans

  // Drag to tilt: sideways turns it, up and down tips it (mouse only: on touch screens vertical drags scroll the page).
  const pose = { ...FRONT };
  let drag = null;
  const clamp = (v, c, lim) => Math.min(c + lim, Math.max(c - lim, v));
  canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, id: e.pointerId }; canvas.setPointerCapture(e.pointerId); canvas.classList.add('dragging'); });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    pose.yaw = clamp(pose.yaw + DRAG * (e.clientX - drag.x), FRONT.yaw, LIMIT.yaw);
    if (e.pointerType === 'mouse') pose.pitch = clamp(pose.pitch + DRAG * (e.clientY - drag.y), FRONT.pitch, LIMIT.pitch);
    drag.x = e.clientX;
    drag.y = e.clientY;
  });
  const release = () => { drag = null; canvas.classList.remove('dragging'); };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);

  const resize = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * r.getPixelRatio()) || canvas.height !== Math.round(h * r.getPixelRatio())) {
      r.setSize(w, h, false);
      cam.aspect = w / h;
      cam.updateProjectionMatrix();
      place();
    }
  };
  let visible = true, last = null, spun = 0, shown = false;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; last = null; }).observe(canvas);
  const frame = (now) => {
    requestAnimationFrame(frame);
    if (!visible || document.hidden) return;
    const dt = last === null ? 0 : Math.min((now - last) / 1000, 0.1);
    last = now;
    if (!drag) { // ease back to the front
      const k = 1 - Math.exp(-dt / SETTLE);
      pose.yaw += k * (FRONT.yaw - pose.yaw);
      pose.pitch += k * (FRONT.pitch - pose.pitch);
    }
    turntable.rotation.set(pose.pitch, pose.yaw, 0, 'XYZ');
    spun += dt;
    mixer.timeScale = Math.min(spun / SPIN_UP, 1) ** 2; // the fans ease up from rest
    mixer.update(dt);
    resize();
    r.render(scene, cam);
    if (!shown) {
      shown = true;
      box.classList.add('ready'); // the first frame is drawn: the still steps aside
    }
  };
  requestAnimationFrame(frame);
}

const WIDE = matchMedia('(min-width: 1000px)'); // style.css shows the card from here
let started = false;
function maybeStart() {
  if (started || !WIDE.matches) return;
  started = true;
  start().catch((err) => { console.warn('RTX 4090 model:', err); box.classList.add('failed'); });
}
if (box && canvas && !navigator.connection?.saveData) {
  maybeStart();
  WIDE.addEventListener('change', maybeStart);
}
