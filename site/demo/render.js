// three.js view of the MuJoCo state (z-up, like MuJoCo). Visual meshes come from assets/robot/visual.json
// + meshes.bin (full detail, uint16-quantized positions, meshopt-compressed, in the compiled model's mesh
// frames); each body is a group posed from data.xpos / data.xquat, interpolated between control ticks.
// Look: the robot in its real colours, lit like a product shot in a dark graphite studio (softbox
// reflections, warm key, one orange rim), subtle bloom and vignette.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

export const PALETTE = { bg: '#0d0e11', accent: '#ff6a1a' };
const FOLLOW_TAU = 0.15; // s, camera follow time constant
// Light offsets from the pelvis (z-up: x right, y away from the default camera, z up).
const LIGHTS = [
  { color: '#fff1e2', intensity: 2.2, offset: [2, -3, 5], shadow: true }, // key, warm white, front right
  { color: '#c9d6ff', intensity: 0.35, offset: [-4, -2, 1.5] }, // fill, cool, front left
  { color: PALETTE.accent, intensity: 3.0, offset: [-3, 3.5, 2.5] }, // rim, orange, back left
  { color: '#ffffff', intensity: 1.2, offset: [3.5, 3, 3] }, // rim, white, back right
];

// Weak devices (touch + few cores): quarter-resolution bloom, 512 shadow map.
export const LOW_END = matchMedia('(pointer: coarse)').matches && (navigator.hardwareConcurrency || 8) <= 4;

// Decode meshes.bin (see visual.json "layout") into indexed geometries with smooth vertex normals.
export function decodeMeshes(visual, bytes, decoder) {
  return visual.geoms.map((g) => {
    const q = new Uint16Array(4 * g.vert_count);
    decoder.decodeVertexBuffer(new Uint8Array(q.buffer), g.vert_count, 8, bytes.subarray(g.vertices[0], g.vertices[0] + g.vertices[1]));
    const index = g.vert_count < 65536 ? new Uint16Array(3 * g.face_count) : new Uint32Array(3 * g.face_count);
    decoder.decodeIndexBuffer(new Uint8Array(index.buffer), index.length, index.BYTES_PER_ELEMENT, bytes.subarray(g.indices[0], g.indices[0] + g.indices[1]));
    const pos = new Float32Array(3 * g.vert_count);
    for (let i = 0; i < g.vert_count; i++)
      for (let k = 0; k < 3; k++) pos[3 * i + k] = g.pos_min[k] + (q[4 * i + k] / 65535) * g.pos_extent[k];
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geom.setIndex(new THREE.BufferAttribute(index, 1));
    geom.computeVertexNormals();
    return geom;
  });
}

// Dark photo studio: a big overhead softbox, two tall side strips and a faint orange strip behind, so
// the shells get long clean highlights. Built y-up and turned into the z-up world.
function studioEnv() {
  const s = new THREE.Scene();
  s.background = new THREE.Color('#050506');
  const room = new THREE.Group();
  room.rotation.x = Math.PI / 2;
  const panel = (color, intensity, w, h, pos, rot) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(intensity), side: THREE.DoubleSide }));
    m.position.set(...pos);
    m.rotation.set(...rot);
    room.add(m);
  };
  panel('#ffffff', 2.2, 5, 3, [0, 6, 0], [Math.PI / 2, 0, 0]); // overhead softbox
  panel('#ffffff', 1.4, 0.8, 6, [-5, 2.5, -1], [0, Math.PI / 2, 0]); // strip, left
  panel('#ffffff', 1.0, 0.8, 6, [5, 2.5, -1], [0, -Math.PI / 2, 0]); // strip, right
  panel(PALETTE.accent, 1.5, 6, 0.3, [0, 1.2, 6], [0, Math.PI, 0]); // orange strip, behind the robot
  s.add(room);
  return s;
}

// Radial falloff, black in the centre and transparent at the edge (contact shadow under the robot).
function radialTexture() {
  const n = 128, c = document.createElement('canvas');
  c.width = c.height = n;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(n / 2, n / 2, 0, n / 2, n / 2, n / 2);
  grad.addColorStop(0, 'rgba(0,0,0,1)');
  grad.addColorStop(0.5, 'rgba(0,0,0,0.5)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, n, n);
  return new THREE.CanvasTexture(c);
}

// Faint 0.5 m / 1 m floor grid (anti-aliased with fwidth), fading out with distance from the robot.
function gridMaterial() {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    uniforms: { center: { value: new THREE.Vector2() } },
    vertexShader: /* glsl */ `
      varying vec2 vXY;
      void main() {
        vec4 w = modelMatrix * vec4(position, 1.0);
        vXY = w.xy;
        gl_Position = projectionMatrix * viewMatrix * w;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec2 center;
      varying vec2 vXY;
      float lines(vec2 p, float scale) {
        vec2 g = abs(fract(p * scale - 0.5) - 0.5) / fwidth(p * scale);
        return 1.0 - clamp(min(g.x, g.y) - 0.3, 0.0, 1.0);
      }
      void main() {
        float fade = 1.0 - smoothstep(1.0, 7.0, distance(vXY, center));
        float a = max(0.35 * lines(vXY, 2.0), 0.8 * lines(vXY, 1.0));
        gl_FragColor = vec4(vec3(0.62, 0.64, 0.68), 0.16 * a * fade);
      }`,
  });
}

// Vignette only.
function vignettePass() {
  return new ShaderPass({
    uniforms: { tDiffuse: { value: null } },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D tDiffuse;
      varying vec2 vUv;
      void main() {
        vec2 c = vUv - 0.5;
        vec4 col = texture2D(tDiffuse, vUv);
        gl_FragColor = vec4(col.rgb * (1.0 - dot(c, c) * 0.8), col.a);
      }`,
  });
}

// World-model ghost: x-ray look (faint faces, a soft blue silhouette that stays below the bloom threshold), drawn
// just in front of coincident surfaces so it reads on top of the real robot when the two agree. One colour
// throughout: how far it is from MuJoCo is the panel's number and graph, not a colour on the robot.
function ghostMaterial() {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    fog: false,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
    uniforms: { color: { value: new THREE.Color(GHOST_NEAR) }, opacity: { value: 1 } },
    vertexShader: /* glsl */ `
      varying vec3 vN;
      varying vec3 vV;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vN = normalize(normalMatrix * normal);
        vV = -mv.xyz;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 color;
      uniform float opacity;
      varying vec3 vN;
      varying vec3 vV;
      void main() {
        float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.0);
        gl_FragColor = vec4(color * (0.55 + 0.6 * f), (0.05 + 0.42 * f) * opacity);
      }`,
  });
}

// Target pose (the reference frame being tracked): rim-only white outline, no face fill.
function outlineMaterial() {
  const m = ghostMaterial();
  m.uniforms.color.value.set('#f4f1ec');
  const fill = '(0.05 + 0.42 * f) * opacity'; // ghostMaterial's alpha: replaced by a rim-only one
  if (!m.fragmentShader.includes(fill)) throw new Error('outlineMaterial: ghost shader changed');
  m.fragmentShader = m.fragmentShader.replace(fill, 'smoothstep(0.3, 0.95, f) * 0.6 * opacity');
  return m;
}

export const GHOST_NEAR = '#3d8bff'; // the world model's colour

// One posed copy of the robot: a group per MuJoCo body, posed from xpos / xquat interpolated between
// the previous and the current control tick.
class Rig {
  constructor(scene, sim, visual, geometries, material) {
    this.groups = new Map(); // MuJoCo body id -> group
    this.meshes = [];
    visual.geoms.forEach((g, i) => {
      const id = sim.bodyId(g.body);
      if (!this.groups.has(id)) {
        const group = new THREE.Group();
        group.matrixAutoUpdate = false;
        this.groups.set(id, group);
        scene.add(group);
      }
      const mesh = new THREE.Mesh(geometries[i], material(g, id));
      mesh.position.fromArray(g.pos);
      mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
      mesh.userData.body = id;
      mesh.userData.mesh = g.mesh;
      this.groups.get(id).add(mesh);
      this.meshes.push(mesh);
    });
    const nb = sim.model.nbody;
    this.prev = { xpos: new Float64Array(3 * nb), xquat: new Float64Array(4 * nb) };
    this.cur = { xpos: new Float64Array(3 * nb), xquat: new Float64Array(4 * nb) };
    this.lastStep = 0;
    this.visible = true;
  }

  snapshot(xpos, xquat, jump = false) {
    const { prev, cur } = this;
    if (jump) {
      cur.xpos.set(xpos);
      cur.xquat.set(xquat);
    }
    prev.xpos.set(cur.xpos);
    prev.xquat.set(cur.xquat);
    cur.xpos.set(xpos);
    cur.xquat.set(xquat);
    this.lastStep = performance.now();
  }

  setVisible(on) {
    this.visible = on;
    for (const g of this.groups.values()) g.visible = on;
  }

  // Pose every body at `alpha` between the two ticks; the pose of body `track` goes to `out`.
  pose(alpha, track, out, tmp) {
    const { prev, cur } = this;
    for (const [id, group] of this.groups) {
      const p = 3 * id, q = 4 * id;
      tmp.p.set(
        prev.xpos[p] + (cur.xpos[p] - prev.xpos[p]) * alpha,
        prev.xpos[p + 1] + (cur.xpos[p + 1] - prev.xpos[p + 1]) * alpha,
        prev.xpos[p + 2] + (cur.xpos[p + 2] - prev.xpos[p + 2]) * alpha,
      );
      tmp.q.set(prev.xquat[q + 1], prev.xquat[q + 2], prev.xquat[q + 3], prev.xquat[q]);
      tmp.q2.set(cur.xquat[q + 1], cur.xquat[q + 2], cur.xquat[q + 3], cur.xquat[q]);
      const s = tmp.q.dot(tmp.q2) < 0 ? -1 : 1; // shortest arc
      tmp.q.set(
        tmp.q.x + (s * tmp.q2.x - tmp.q.x) * alpha,
        tmp.q.y + (s * tmp.q2.y - tmp.q.y) * alpha,
        tmp.q.z + (s * tmp.q2.z - tmp.q.z) * alpha,
        tmp.q.w + (s * tmp.q2.w - tmp.q.w) * alpha,
      ).normalize();
      group.matrix.compose(tmp.p, tmp.q, tmp.s);
      group.matrixWorldNeedsUpdate = true;
      if (id === track) out.copy(tmp.p);
    }
  }
}

export class View {
  constructor(canvas, sim, visual, geometries) {
    this.sim = sim;
    this.canvas = canvas;
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' }));
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.0;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFShadowMap; // three r186 removed PCFSoftShadowMap (it maps to this)

    const scene = (this.scene = new THREE.Scene());
    scene.background = new THREE.Color(PALETTE.bg);
    scene.fog = new THREE.FogExp2(PALETTE.bg, 0.07);
    const pmrem = new THREE.PMREMGenerator(r);
    scene.environment = pmrem.fromScene(studioEnv(), 0.04).texture;
    scene.environmentIntensity = 0.9;
    pmrem.dispose();

    const cam = (this.camera = new THREE.PerspectiveCamera(40, 1, 0.05, 80));
    cam.up.set(0, 0, 1);
    cam.position.set(2.0, -2.3, 1.3);
    const ctl = (this.controls = new OrbitControls(cam, canvas));
    ctl.target.set(0, 0, 0.75);
    ctl.enableDamping = true;
    ctl.maxPolarAngle = Math.PI / 2 - 0.03;
    ctl.minDistance = 0.8;
    ctl.maxDistance = 8;
    this.homeOffset = cam.position.clone().sub(ctl.target); // the default view, relative to what it looks at

    this.lights = [];
    for (const spec of LIGHTS) {
      const light = new THREE.DirectionalLight(spec.color, spec.intensity);
      if (spec.shadow) {
        light.castShadow = true;
        const n = LOW_END ? 512 : 1024;
        light.shadow.mapSize.set(n, n);
        Object.assign(light.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.5, far: 20 });
        light.shadow.camera.updateProjectionMatrix();
        light.shadow.bias = -5e-4;
      }
      this.lights.push({ light, offset: new THREE.Vector3(...spec.offset) });
      scene.add(light, light.target);
    }

    // Matte floor (a glossy one shows the rim lights as hot spots), faint grid, contact shadow.
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshStandardMaterial({ color: '#0c0d10', metalness: 0, roughness: 0.9 }));
    floor.receiveShadow = true;
    this.grid = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), gridMaterial());
    this.grid.position.z = 0.001;
    this.contact = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: radialTexture(), transparent: true, depthWrite: false }));
    this.contact.position.z = 0.002;
    scene.add(floor, this.grid, this.contact);

    // Reality: the model's own two colours (light grey shells, dark grey joints and hands), read as sRGB.
    const materials = new Map();
    this.robot = new Rig(scene, sim, visual, geometries, (g) => {
      const key = g.rgba.join(',');
      if (!materials.has(key)) {
        const color = new THREE.Color().setRGB(g.rgba[0], g.rgba[1], g.rgba[2], THREE.SRGBColorSpace);
        const light = g.rgba[0] > 0.45;
        materials.set(key, new THREE.MeshStandardMaterial({ color, metalness: light ? 0.15 : 0.1, roughness: light ? 0.38 : 0.5 }));
      }
      return materials.get(key);
    });
    for (const m of this.robot.meshes) m.castShadow = m.receiveShadow = true;
    this.darkMaterial = [...materials.entries()].find(([k]) => Number(k.split(',')[0]) <= 0.45)?.[1];
    this.bodies = this.robot.groups;
    this.pickables = this.robot.meshes;
    this.extraPickables = [];
    // The world model: the same meshes, one tintable material per body.
    this.ghostMaterials = new Map(); // body id -> material
    this.ghost = new Rig(scene, sim, visual, geometries, (g, id) => {
      if (!this.ghostMaterials.has(id)) this.ghostMaterials.set(id, ghostMaterial());
      return this.ghostMaterials.get(id);
    });
    for (const m of this.ghost.meshes) m.renderOrder = 2;
    this.ghost.setVisible(false);
    // The target pose: the reference frame the policy is tracking, as a white outline.
    const outline = (this.outline = outlineMaterial());
    this.target = new Rig(scene, sim, visual, geometries, () => outline);
    for (const m of this.target.meshes) m.renderOrder = 3;
    this.target.setVisible(false);
    this.follow = this.robot; // the rig the camera and the lights follow
    this.pelvis = sim.bodyId('pelvis');

    // Post-processing into a multisampled half-float target (the canvas itself is not antialiased).
    const rt = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
    const composer = (this.composer = new EffectComposer(r, rt));
    composer.addPass(new RenderPass(scene, cam));
    const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.25, 0.4, 0.9);
    if (LOW_END) {
      const setSize = bloom.setSize.bind(bloom);
      bloom.setSize = (w, h) => setSize(Math.round(w / 2), Math.round(h / 2)); // quarter resolution
    }
    composer.addPass(bloom);
    composer.addPass(vignettePass());
    composer.addPass(new OutputPass()); // tone mapping + sRGB

    this.stepDt = 0.02;
    this.snapshot(true);
    this.tmp = { p: new THREE.Vector3(), q: new THREE.Quaternion(), q2: new THREE.Quaternion(), s: new THREE.Vector3(1, 1, 1) };
    this._pelvis = new THREE.Vector3(0, 0, 0.75);
    this._unused = new THREE.Vector3();
    this._d = new THREE.Vector3();
    this._near = new THREE.Color(GHOST_NEAR);
    this.lastFrame = null;
  }

  // Record the simulation pose after a control tick; `jump` (reset) skips the interpolation.
  snapshot(jump = false) {
    this.robot.snapshot(this.sim.data.xpos, this.sim.data.xquat, jump);
  }

  // Record the world model's body poses (forward kinematics of its state) after a tick.
  ghostSnapshot(xpos, xquat, jump = false) {
    this.ghost.snapshot(xpos, xquat, jump);
  }

  // Robot variant (variant.json from tools/export_web_scene.py --variant), or null for the base robot:
  // hide the base meshes the variant does not have, add its own meshes (`geometries`, decoded from its
  // meshes.bin in variant.geoms order) and stand-in shapes to every rig.
  setVariant(variant, geometries = []) {
    const hide = new Set(variant?.hide_meshes ?? []);
    for (const extra of this.extras ?? []) extra.removeFromParent();
    this.extras = [];
    for (const [rig, material] of [[this.robot, () => this.darkMaterial], [this.ghost, (id) => this.ghostMaterials.get(id)], [this.target, () => this.outline]]) {
      for (const mesh of rig.meshes) mesh.visible = !hide.has(mesh.userData.mesh);
      (variant?.geoms ?? []).forEach((g, i) => {
        const id = this.sim.bodyId(g.body);
        const mesh = new THREE.Mesh(geometries[i], material(id) ?? this.darkMaterial);
        mesh.position.fromArray(g.pos);
        mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
        mesh.userData.body = id;
        mesh.castShadow = mesh.receiveShadow = rig === this.robot;
        mesh.renderOrder = rig === this.robot ? 0 : rig === this.ghost ? 2 : 3;
        rig.groups.get(id).add(mesh);
        this.extras.push(mesh);
        if (rig === this.robot) this.extraPickables.push(mesh);
      });
      for (const e of variant?.extras ?? []) {
        const id = this.sim.bodyId(e.body);
        // Capsules and cylinders run along z as in MuJoCo (size: radius, half-length); boxes take half-sizes.
        const geom = e.type === 'capsule'
          ? new THREE.CapsuleGeometry(e.size[0], 2 * e.size[1], 6, 16).rotateX(Math.PI / 2)
          : e.type === 'cylinder'
            ? new THREE.CylinderGeometry(e.size[0], e.size[0], 2 * e.size[1], 32).rotateX(Math.PI / 2)
            : new THREE.BoxGeometry(2 * e.size[0], 2 * e.size[1], 2 * e.size[2]);
        const mesh = new THREE.Mesh(geom, material(id) ?? this.darkMaterial);
        mesh.position.fromArray(e.pos);
        mesh.quaternion.set(e.quat[1], e.quat[2], e.quat[3], e.quat[0]);
        mesh.userData.body = id;
        mesh.castShadow = rig === this.robot;
        mesh.renderOrder = rig === this.robot ? 0 : rig === this.ghost ? 2 : 3;
        rig.groups.get(id).add(mesh);
        this.extras.push(mesh);
        if (rig === this.robot) this.extraPickables.push(mesh);
      }
    }
    this.pickables = [...this.robot.meshes.filter((m) => m.visible), ...this.extraPickables.splice(0)];
  }

  // Record the target pose's body poses (forward kinematics of the reference frame) after a tick.
  targetSnapshot(xpos, xquat, jump = false) {
    this.target.snapshot(xpos, xquat, jump);
  }


  neutralGhost() {
    for (const m of this.ghostMaterials.values()) m.uniforms.color.value.copy(this._near);
  }

  // Pose both rigs (interpolated); the camera and the lights follow the pelvis of `this.follow`.
  update(now = performance.now()) {
    const dt = this.lastFrame === null ? 0 : Math.min((now - this.lastFrame) / 1000, 0.1);
    this.lastFrame = now;
    for (const rig of [this.robot, this.ghost, this.target]) {
      if (!rig.visible) continue;
      const alpha = Math.min(Math.max((now - rig.lastStep) / 1000 / this.stepDt, 0), 1);
      rig.pose(alpha, this.pelvis, rig === this.follow ? this._pelvis : this._unused, this.tmp);
    }

    const pel = this._pelvis;
    const k = 1 - Math.exp(-dt / FOLLOW_TAU);
    this._d.subVectors(pel, this.controls.target).multiplyScalar(k);
    this.controls.target.add(this._d);
    this.camera.position.add(this._d);
    for (const { light, offset } of this.lights) {
      light.position.copy(pel).add(offset);
      light.target.position.copy(pel);
    }
    const h = Math.min(Math.max((pel.z - 0.1) / 0.9, 0), 1);
    this.contact.position.set(pel.x, pel.y, 0.002);
    this.contact.scale.setScalar(0.9 + 0.6 * h);
    this.contact.material.opacity = 0.5 * (1 - 0.5 * h);
    this.grid.material.uniforms.center.value.set(pel.x, pel.y);
    this.controls.update();
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    const pr = this.renderer.getPixelRatio();
    if (this.canvas.width !== Math.floor(w * pr) || this.canvas.height !== Math.floor(h * pr)) {
      this.renderer.setSize(w, h, false);
      this.composer.setPixelRatio(pr);
      this.composer.setSize(w, h);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
  }

  render() {
    this.resize();
    this.composer.render();
  }
}
