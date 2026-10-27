// Drag-to-push: grab a point on the robot and pull it with a spring; the force goes into
// xfrc_applied (at the grabbed point) on every control tick while the pointer is held. The force is
// capped at `maxForce` and reaches the cap at REACH of pointer travel.

import * as THREE from 'three';
import { PALETTE } from './render.js';

const REACH = 0.5; // m of drag at which the spring reaches the force cap

export class Interact {
  constructor(view, sim, maxForce) {
    this.view = view;
    this.sim = sim;
    this.maxForce = maxForce; // N
    this.enabled = true; // false while reality (MuJoCo) is off: the world model has no force input
    this.stiffness = maxForce / REACH; // N/m
    this.grab = null; // {body, local: Vector3 in the body frame, target: Vector3 world}
    this.ray = new THREE.Raycaster();
    this.plane = new THREE.Plane();
    this.ndc = new THREE.Vector2();
    this.force = 0;
    // Accent-orange spring line (not tone mapped, so it blooms) with glowing ends at the grab point and the cursor.
    const amber = new THREE.Color(PALETTE.accent).multiplyScalar(2);
    const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    this.line = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: amber, toneMapped: false, depthTest: false }));
    const dot = new THREE.MeshBasicMaterial({ color: amber, toneMapped: false, depthTest: false });
    this.handle = new THREE.Mesh(new THREE.SphereGeometry(0.018, 16, 12), dot);
    this.cursor = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 12), dot);
    for (const o of [this.line, this.handle, this.cursor]) {
      o.renderOrder = 1;
      o.visible = false;
      view.scene.add(o);
    }
    const c = view.canvas;
    c.addEventListener('pointerdown', (e) => this.down(e), { capture: true }); // before OrbitControls
    c.addEventListener('pointermove', (e) => this.move(e));
    c.addEventListener('pointerup', (e) => this.up(e));
    c.addEventListener('pointercancel', (e) => this.up(e));
  }

  // Force cap (N), e.g. after a switch to a heavier robot; the spring still reaches it at REACH.
  setMaxForce(maxForce) {
    this.maxForce = maxForce;
    this.stiffness = maxForce / REACH;
  }

  cast(e) {
    const r = this.view.canvas.getBoundingClientRect();
    this.ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(this.ndc, this.view.camera);
  }

  down(e) {
    if (e.button !== 0 || !this.enabled) return;
    this.cast(e);
    const hit = this.ray.intersectObjects(this.view.pickables, false)[0];
    if (!hit) return;
    const group = this.view.bodies.get(hit.object.userData.body);
    group.updateMatrixWorld(true);
    this.grab = { body: hit.object.userData.body, group, local: group.worldToLocal(hit.point.clone()), target: hit.point.clone() };
    this.plane.setFromNormalAndCoplanarPoint(this.view.camera.getWorldDirection(new THREE.Vector3()), hit.point);
    this.view.controls.enabled = false;
    this.view.canvas.setPointerCapture(e.pointerId);
    e.stopImmediatePropagation();
    e.preventDefault();
  }

  move(e) {
    if (!this.grab) return;
    this.cast(e);
    this.ray.ray.intersectPlane(this.plane, this.grab.target);
  }

  up(e) {
    if (!this.grab) return;
    this.grab = null;
    this.force = 0;
    this.view.controls.enabled = true;
    if (this.view.canvas.hasPointerCapture(e.pointerId)) this.view.canvas.releasePointerCapture(e.pointerId);
  }

  // Called once per control tick, after the forces are cleared and before the physics steps.
  apply() {
    if (!this.grab) return;
    const g = this.grab;
    const b = g.body, xp = this.sim.data.xpos, xq = this.sim.data.xquat; // current physics pose
    const q = new THREE.Quaternion(xq[4 * b + 1], xq[4 * b + 2], xq[4 * b + 3], xq[4 * b]);
    const p = g.local.clone().applyQuaternion(q).add(new THREE.Vector3(xp[3 * b], xp[3 * b + 1], xp[3 * b + 2]));
    const f = g.target.clone().sub(p).multiplyScalar(this.stiffness);
    if (f.length() > this.maxForce) f.setLength(this.maxForce);
    this.force = f.length();
    this.sim.addForce(g.body, [f.x, f.y, f.z], [p.x, p.y, p.z]);
  }

  // Draw the spring (every frame).
  update() {
    const on = this.grab !== null;
    this.line.visible = this.handle.visible = this.cursor.visible = on;
    if (!on) return;
    const g = this.grab;
    g.group.updateMatrixWorld(true);
    const p = g.group.localToWorld(g.local.clone());
    this.line.geometry.setFromPoints([p, g.target]);
    this.handle.position.copy(p);
    this.cursor.position.copy(g.target);
  }
}
