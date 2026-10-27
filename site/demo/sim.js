// MuJoCo side of the demo, shared by the page and the Node check (no DOM).
// The model is the training scene compiled to scene.mjb (solver options and friction included).

const PREFIX = 'robot/';

export class Sim {
  static create(mujoco, mjbBytes) {
    const vfs = new mujoco.MjVFS();
    try {
      vfs.addBuffer('scene.mjb', mjbBytes);
      return new Sim(mujoco, mujoco.MjModel.from_binary_path('scene.mjb', vfs));
    } finally {
      vfs.delete();
    }
  }

  constructor(mujoco, model) {
    this.mj = mujoco;
    this.model = model;
    this.data = new mujoco.MjData(model);
    this.nq = model.nq;
    this.nv = model.nv;
    this.nu = model.nu;
    // Actuator i drives joint actJoint[i] (joint order without the free joint); not the identity.
    const trn = model.actuator_trnid;
    this.actJoint = Int32Array.from({ length: this.nu }, (_, i) => trn[2 * i] - 1);
    this.key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY.value, 'init_state');
    this.dt = model.opt.timestep;
    this.time = 0;
  }

  bodyId(name) {
    return this.mj.mj_name2id(this.model, this.mj.mjtObj.mjOBJ_BODY.value, PREFIX + name);
  }

  totalMass() {
    return this.model.body_mass.reduce((a, b) => a + b, 0);
  }

  bodyName(id) {
    return this.mj.mj_id2name(this.model, this.mj.mjtObj.mjOBJ_BODY.value, id).replace(PREFIX, '');
  }

  reset() {
    this.mj.mj_resetDataKeyframe(this.model, this.data, this.key);
    this.mj.mj_forward(this.model, this.data);
    this.time = 0;
  }

  // [qpos | qvel] (float64 copy).
  state(out = new Float64Array(this.nq + this.nv)) {
    out.set(this.data.qpos, 0);
    out.set(this.data.qvel, this.nq);
    return out;
  }

  // Sliding friction of the foot collision geoms, as training randomised it (the floor stays at 1.0). The feet have
  // geom_priority 1 over the floor's 0, so a foot-floor contact takes the foot's value. The exported scene has 1.0
  // (evaluation protocol).
  setFriction(mu) {
    const m = this.model, obj = this.mj.mjtObj.mjOBJ_GEOM.value;
    this.frictionGeoms ??= Array.from({ length: m.ngeom }, (_, g) => g).filter((g) => /^robot\/(left|right)_foot[1-7]_collision$/.test(this.mj.mj_id2name(m, obj, g) ?? ''));
    for (const g of this.frictionGeoms) m.geom_friction[3 * g] = mu;
    this.friction = mu;
  }

  // Joint positions into the soft limits (0.9 of each joint's range, about its middle), as the benchmark's reset does
  // (mjlab's G1 articulation; web_closed_loop.SOFT_JOINT_LIMIT_FACTOR).
  clipToSoftLimits() {
    const m = this.model, r = m.jnt_range, q = this.data.qpos;
    for (let j = 1; j < m.njnt; j++) { // joint 0: the free joint
      const mid = (r[2 * j] + r[2 * j + 1]) / 2, half = ((r[2 * j + 1] - r[2 * j]) / 2) * 0.9, a = m.jnt_qposadr[j];
      q[a] = Math.min(mid + half, Math.max(mid - half, q[a]));
    }
    this.mj.mj_forward(m, this.data);
  }

  // Policy off: every actuator's stiffness and damping to zero (the robot goes limp); on: the model's values.
  setActuatorsEnabled(on) {
    const m = this.model, g = m.actuator_gainprm, b = m.actuator_biasprm, n = 10; // mjNGAIN = mjNBIAS = 10
    this.gains ??= { g: Float64Array.from(g), b: Float64Array.from(b) };
    for (let i = 0; i < this.nu; i++) {
      g[i * n] = on ? this.gains.g[i * n] : 0;
      b[i * n + 1] = on ? this.gains.b[i * n + 1] : 0;
      b[i * n + 2] = on ? this.gains.b[i * n + 2] : 0;
    }
    this.actuatorsOn = on;
  }

  // Overwrite the physics state with [qpos | qvel] (e.g. reality taking over from the world model).
  setState(state) {
    this.data.qpos.set(state.subarray(0, this.nq));
    this.data.qvel.set(state.subarray(this.nq, this.nq + this.nv));
    this.mj.mj_forward(this.model, this.data);
  }

  // Body poses of an arbitrary qpos (forward kinematics on a separate MjData per `slot`, so the world
  // model and the target pose keep their own buffers): {xpos, xquat}.
  kinematics(qpos, slot = 0) {
    this.fk ??= [];
    let d = (this.fk[slot] ??= new this.mj.MjData(this.model));
    // Allocating another slot can grow the WASM memory, which detaches this one's array views: make it again.
    if (d.qpos.length !== this.nq) d = this.fk[slot] = new this.mj.MjData(this.model);
    d.qpos.set(qpos);
    this.mj.mj_kinematics(this.model, d);
    return { xpos: d.xpos, xquat: d.xquat };
  }

  setJointTargets(target) {
    const ctrl = this.data.ctrl;
    for (let i = 0; i < this.nu; i++) ctrl[i] = target[this.actJoint[i]];
  }

  clearForces() {
    this.data.xfrc_applied.fill(0);
  }

  // World-frame force on a body, at `point` (world) or at its centre of mass.
  addForce(body, force, point = null) {
    const f = this.data.xfrc_applied;
    const o = 6 * body;
    f[o] += force[0];
    f[o + 1] += force[1];
    f[o + 2] += force[2];
    if (point) {
      const c = this.data.xipos;
      const r = [point[0] - c[3 * body], point[1] - c[3 * body + 1], point[2] - c[3 * body + 2]];
      f[o + 3] += r[1] * force[2] - r[2] * force[1];
      f[o + 4] += r[2] * force[0] - r[0] * force[2];
      f[o + 5] += r[0] * force[1] - r[1] * force[0];
    }
  }

  step(n) {
    for (let i = 0; i < n; i++) this.mj.mj_step(this.model, this.data);
    this.time += n * this.dt;
  }
}
