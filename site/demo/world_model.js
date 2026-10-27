// World-model rollout: the policy driving the trained world model instead of MuJoCo, as in training
// (imagined rollouts: the policy acts, the world model predicts the next state 20 ms later). Shared by the page and the Node
// check (no DOM). Next to reality it is re-synced every `horizon` ticks (training used 32) from the real
// state and the real policy history; the divergence is the mean distance between the world model's
// and reality's bodies (forward kinematics of both poses).

import { Policy, gravityZ } from './policy.js';
import { ROW } from './motion.js';

export const DIVERGENCE_FULL = 0.08; // m of body error drawn fully red (ghost tint, graph, readout)

export class WorldModelRollout {
  // policy: the page's Policy (its session is shared; the rollout keeps its own history buffers).
  constructor(ort, wmSession, policy, meta) {
    this.ort = ort;
    this.session = wmSession;
    this.policy = new Policy(ort, policy.session, meta);
    this.state = new Float32Array(71);
    this.window = new Float64Array(this.policy.K * ROW);
    this.tick = 0; // motion tick of the rollout
    this.friction = new Float32Array([1]); // foot friction of the scene (world-model context)
    this.age = 0; // ticks since the last sync
  }

  // Start from reality: the real state, the real policy history, the controller's motion tick.
  sync(sim, ctl) {
    this.state.set(sim.state());
    this.policy.states.set(ctl.policy.states);
    this.policy.actions.set(ctl.policy.actions);
    this.tick = ctl.tick;
    this.age = 0;
  }

  // One imagined tick: policy on the rollout's state history, then the world model.
  async step(motion) {
    motion.window(this.tick, this.state.subarray(3, 7), this.policy.K, this.window);
    const { action } = await this.policy.act(this.window);
    const T = this.ort.Tensor;
    const out = await this.session.run({ state: new T('float32', this.state, [1, 71]), action: new T('float32', Float32Array.from(action), [1, 29]), friction: new T('float32', this.friction, [1, 1]) });
    this.state.set(out.next_state.data);
    this.policy.push(this.state, action);
    this.tick += 1;
    this.age += 1;
  }

  // Training termination of the rollout against its current reference frame (pelvis height or tilt error).
  fallen(term) {
    const w = this.window, s = this.state;
    return Math.abs(w[2] - s[2]) > term.height || Math.abs(gravityZ(w, 3) - gravityZ(s, 3)) > term.tilt;
  }

  // Per-body distance (m) between the world model and reality, into `out` (indexed by body id); returns the
  // mean over the robot's bodies (ids 1.., body 0 is the world). The world model's body poses stay in
  // this.fk ({xpos, xquat}).
  divergence(sim, out) {
    const real = sim.data.xpos;
    const { xpos } = (this.fk = sim.kinematics(this.state.subarray(0, sim.nq)));
    let sum = 0;
    const nb = sim.model.nbody;
    for (let b = 1; b < nb; b++) {
      const dx = xpos[3 * b] - real[3 * b], dy = xpos[3 * b + 1] - real[3 * b + 1], dz = xpos[3 * b + 2] - real[3 * b + 2];
      out[b] = Math.hypot(dx, dy, dz);
      sum += out[b];
    }
    return sum / (nb - 1);
  }
}
