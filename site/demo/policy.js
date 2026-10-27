// Policy inference (onnxruntime-web, single-threaded WASM) and the 50 Hz control loop, shared by the
// page and the Node check (no DOM). The ONNX graph builds the observations itself from raw MuJoCo
// state (see assets/policy/meta.json); this file only keeps the history buffers.

import { ROW } from './motion.js';

const NA = 29;

export class Policy {
  static async create(ort, onnxBytes, meta) {
    const session = await ort.InferenceSession.create(onnxBytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    return new Policy(ort, session, meta);
  }

  constructor(ort, session, meta) {
    this.ort = ort;
    this.session = session;
    this.H = meta.inputs.state_hist.shape[1];
    this.K = meta.inputs.ref_window.shape[1];
    this.S = meta.inputs.state_hist.shape[2];
    this.states = new Float32Array(this.H * this.S);
    this.actions = new Float32Array(this.H * NA);
    this.ref = new Float32Array(this.K * ROW);
  }

  // Every history slot holds the reset state and a zero action (mjlab's observation history).
  reset(state) {
    for (let i = 0; i < this.H; i++) this.states.set(state, i * this.S);
    this.actions.fill(0);
  }

  // Append the state reached after applying `action` (oldest entry dropped).
  push(state, action) {
    this.states.copyWithin(0, this.S);
    this.states.set(state, (this.H - 1) * this.S);
    this.actions.copyWithin(0, NA);
    this.actions.set(action, (this.H - 1) * NA);
  }

  // refWindow: K x 71 rows (float64 or float32). -> {action, target} (Float32Array(29), joint order).
  async act(refWindow) {
    this.ref.set(refWindow);
    const T = this.ort.Tensor;
    const out = await this.session.run({
      state_hist: new T('float32', this.states, [1, this.H, this.S]),
      action_hist: new T('float32', this.actions, [1, this.H, NA]),
      ref_window: new T('float32', this.ref, [1, this.K, ROW]),
    });
    return { action: out.action.data, target: out.joint_target.data };
  }
}

// Gravity's z in the body frame of a wxyz quaternion.
export function gravityZ(q, o = 0) {
  const w = q[o], x = q[o + 1], y = q[o + 2], z = q[o + 3];
  const n2 = w * w + x * x + y * y + z * z;
  return -(1 - (2 * (x * x + y * y)) / n2);
}

export class Controller {
  constructor(sim, policy, meta) {
    this.sim = sim;
    this.policy = policy;
    this.decimation = meta.decimation;
    this.term = meta.termination;
    if (Math.abs(sim.dt * this.decimation - meta.control_dt) > 1e-12) throw new Error('physics dt x decimation != control dt');
    this.state = new Float64Array(sim.nq + sim.nv);
    this.window = new Float64Array(policy.K * ROW);
    this.motion = null;
  }

  setMotion(motion) {
    this.motion = motion;
    this.reset();
  }

  // A clip that plays once starts at its first frame (pose and velocities); poses and looping clips (which
  // start in the default pose) start at the model's keyframe.
  reset() {
    this.sim.reset();
    if (this.motion?.startState && !this.motion.loop) {
      this.sim.setState(this.motion.startState());
      this.sim.clipToSoftLimits(); // a clip (an upload in particular) may start past them
    }
    this.policy.reset(this.sim.state(this.state));
    this.tick = 0;
    this.fallen = false;
  }

  // One control step: reference window, policy, `decimation` physics steps with the external forces
  // already in xfrc_applied, history update. Returns the policy outputs.
  async step() {
    const qpos = this.sim.data.qpos;
    this.motion.window(this.tick, qpos.subarray(3, 7), this.policy.K, this.window);
    if (this.policyOff) { // actuators limp (Sim.setActuatorsEnabled): physics only, no action
      this.sim.step(this.decimation);
      this.policy.push(this.sim.state(this.state), (this.zeroAction ??= new Float32Array(this.sim.nu)));
      this.tick += 1;
      this.fallen ||= this.terminated(this.window);
      return null;
    }
    const out = await this.policy.act(this.window);
    this.sim.setJointTargets(out.target);
    this.sim.step(this.decimation);
    this.policy.push(this.sim.state(this.state), out.action);
    this.tick += 1;
    this.fallen ||= this.terminated(this.window);
    return out;
  }

  // Training termination against the current reference frame: pelvis height or tilt error.
  terminated(ref) {
    const q = this.sim.data.qpos;
    return Math.abs(ref[2] - q[2]) > this.term.height || Math.abs(gravityZ(ref, 3) - gravityZ(q, 3)) > this.term.tilt;
  }
}
