"""Goal layouts, critic error features and the learning-rate schedule."""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
needs_npz = pytest.mark.skipif(not NPZ.exists(), reason="LAFAN1 npz not built")


def test_anchor_mode_and_goal_dim():
    from supermimic.goal_spec import anchor_mode, goal_dim

    assert anchor_mode(True) == "pos_ori" and anchor_mode(False) == "none" and anchor_mode("ori") == "ori"
    with pytest.raises(ValueError):
        anchor_mode("pos")
    assert goal_dim(10, "v1", "none") == 680
    assert goal_dim(10, "v1", "ori") == 686
    assert goal_dim(10, "v1", True) == 689
    assert goal_dim(10, "traj", "ori") == 776


def test_cosine_lr_by_iteration():
    from supermimic.training.schedule import cosine_lr

    assert cosine_lr(0, 2e-4) == pytest.approx(2e-4)
    assert cosine_lr(125_000, 2e-4) == pytest.approx(2e-4 * (0.03 + 0.97 * 0.5))
    assert cosine_lr(250_000, 2e-4) == pytest.approx(6e-6)
    assert cosine_lr(320_000, 2e-4) == pytest.approx(6e-6)


def test_goal_scale_vector_layouts():
    from supermimic.policy.nets import PolicyCfg, _goal_scale_vector

    v1 = _goal_scale_vector(PolicyCfg(), 680)
    assert v1.shape == (680,) and v1[29] == pytest.approx(0.2)  # joint_vel / 5
    old_e3 = _goal_scale_vector(PolicyCfg(), 689)  # v1 config without future_frames/goal_frame_dim info
    assert old_e3.shape == (689,) and torch.all(old_e3[680:] == 1)
    traj = _goal_scale_vector(PolicyCfg(goal_frame_dim=77), 776)
    assert traj.shape == (776,) and traj[77 + 29] == pytest.approx(0.2) and torch.all(traj[68:77] == 1)


@pytest.fixture(scope="module")
def cpu_lib():
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads

    files = discover_npz(NPZ, names=["walk1_subject1", "dance1_subject1"])
    q0 = torch.zeros(29)
    lib = MotionLibrary(files, TRACKED_BODIES, q0, device="cpu")
    model = load_g1_model()
    lim = torch.as_tensor(model.jnt_range[1:], dtype=torch.float32)
    heads = AnalyticHeads(G1Kinematics(model, device="cpu"), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, lim, q0)
    return lib, heads


@needs_npz
def test_traj_goal_features(cpu_lib):
    lib, _ = cpu_lib
    f = torch.tensor([0, 100, 2000, lib.clip_last[0].item() - 3, lib.clip_start[1].item() + 5])
    v1 = lib.goal_window(f, 10, "v1").reshape(5, 10, 68)
    tr = lib.goal_window(f, 10, "traj").reshape(5, 10, 77)
    torch.testing.assert_close(tr[..., :68], v1)
    # The first frame of the window is the reference of its own heading frame: no displacement,
    # and the rotation is tilt-only (its heading is yaw-free).
    torch.testing.assert_close(tr[:, 0, 68:71], torch.zeros(5, 3), atol=1e-6, rtol=0)
    from supermimic.kinematics.rotations import matrix_from_quat, quat_mul, quat_inv, yaw_quat

    q = lib.ref(f).root_quat_w
    local = matrix_from_quat(quat_mul(quat_inv(yaw_quat(q)), q))[..., :2].reshape(5, 6)
    torch.testing.assert_close(tr[:, 0, 71:77], local, atol=1e-5, rtol=0)
    # Frames past the clip end repeat the last frame (window clamped inside the clip).
    torch.testing.assert_close(tr[3, 3:], tr[3, 3:4].expand(7, 77))


@needs_npz
@pytest.mark.parametrize("frame", ["body", "heading"])
def test_critic_error_features_vanish_on_reference(cpu_lib, frame):
    """At the reference state itself every privileged error is ~0 (FK of the library state vs the
    npz body states written by the converter); the anchor rotation is the identity."""
    from supermimic.world_model.heads import ROOT_ERR_DIMS, critic_err_dim
    from supermimic.world_model.wm_trainer import reference_at

    lib, heads = cpu_lib
    heads.state_frame = frame
    f = torch.randint(0, lib.num_frames, (64,), generator=torch.Generator().manual_seed(0))
    qpos, qvel = lib.ref_qpos_qvel(f)
    s = torch.cat([qpos, qvel], -1)
    ref = reference_at(lib, f)
    e = heads.critic_error_features(s, ref)
    assert e.shape == (64, critic_err_dim(frame))
    r = ROOT_ERR_DIMS[frame] + 3  # root error, then the anchor position error
    ident = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expand(64, 6)
    np.testing.assert_allclose(e[:, r : r + 6].numpy(), ident.numpy(), atol=1e-4)
    rest = torch.cat([e[:, :r], e[:, r + 6 :]], -1)
    assert rest.abs().max() < 1e-3, rest.abs().max()
    # Moving the robot shows up in the root / anchor / joint errors, and the features are differentiable in s.
    s2 = s.clone().requires_grad_(True)
    e2 = heads.critic_error_features(s2 + torch.cat([torch.tensor([0.1, 0.0, 0.0]), torch.zeros(68)]), ref)
    torch.testing.assert_close(e2[:, :3].norm(dim=-1).detach(), torch.full((64,), 0.1), atol=1e-4, rtol=0)
    e2.sum().backward()
    assert torch.isfinite(s2.grad).all() and s2.grad.abs().sum() > 0
    heads.state_frame = "body"


def test_running_norm_matches_batch_statistics():
    from supermimic.world_model.dynamics import RunningNorm

    torch.manual_seed(0)
    xs = [torch.randn(1000, 5) * torch.arange(1, 6) + 3.0 for _ in range(4)]
    rn = RunningNorm(5)
    for x in xs:
        rn.update(x)
    allx = torch.cat(xs)
    torch.testing.assert_close(rn.mean, allx.mean(0), atol=1e-4, rtol=0)
    torch.testing.assert_close(rn.var, allx.var(0, unbiased=False), atol=1e-3, rtol=1e-4)
    frozen = RunningNorm(5, max_count=1000)
    frozen.update(xs[0])
    m = frozen.mean.clone()
    frozen.update(xs[1] + 100.0)
    torch.testing.assert_close(frozen.mean, m)  # no updates past max_count


@needs_npz
def test_target_delta_inverts_integration(cpu_lib):
    """integrate_physical(s, target_delta(s, s')) == s' (running output normalisation relies on it)."""
    from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

    lib, _ = cpu_lib
    spec = StateSpec(StateSpecCfg(), "cpu")
    f = torch.randint(0, lib.num_frames - 5, (256,), generator=torch.Generator().manual_seed(1))
    s0 = torch.cat(lib.ref_qpos_qvel(f), -1)
    s1 = torch.cat(lib.ref_qpos_qvel(f + 1), -1)
    s1r = spec.integrate_physical(s0, spec.target_delta(s0, s1))
    torch.testing.assert_close(s1r[:, :3], s1[:, :3], atol=1e-5, rtol=0)
    torch.testing.assert_close(s1r[:, 7:], s1[:, 7:], atol=1e-5, rtol=0)
    dq = (s1r[:, 3:7] * s1[:, 3:7]).sum(-1).abs()  # same rotation up to sign
    torch.testing.assert_close(dq, torch.ones(256), atol=1e-5, rtol=0)


@needs_npz
def test_world_model_options(cpu_lib):
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    lib, heads = cpu_lib
    f = torch.randint(0, lib.num_frames - 5, (64,), generator=torch.Generator().manual_seed(2))
    s0 = torch.cat(lib.ref_qpos_qvel(f), -1)
    s1 = torch.cat(lib.ref_qpos_qvel(f + 1), -1)
    a = torch.zeros(64, 29)
    xi = torch.zeros(64, 33)
    xi[:, 0] = 0.75
    wm = WorldModel(WorldModelCfg(hidden=(64, 64), normalization="running", layer_norm=True, termination="analytic"), heads, "cpu")
    wm.update_normalizers(s0, a, xi, s1)
    # Zero-initialised last layer + running output stats: the first prediction is the mean increment.
    pred = wm.step(s0, a, xi)
    mean_next = wm.spec.integrate_physical(s0, wm.out_norm.mean.expand(64, 70))
    torch.testing.assert_close(pred, mean_next)
    # Analytic termination: ~0 on the reference, ~1 once the pelvis is 0.4 m too low.
    from supermimic.world_model.wm_trainer import reference_at

    ref = reference_at(lib, f)
    p_ok = torch.sigmoid(wm.termination_logits(s0, ref))
    low = s0.clone()
    low[:, 2] -= 0.4
    p_bad = torch.sigmoid(wm.termination_logits(low, ref))
    assert p_ok.max() < 1e-3 and p_bad.min() > 0.99


@pytest.fixture(scope="module")
def frozen_lib(cpu_lib):
    from supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES

    files = discover_npz(NPZ, names=["walk1_subject1", "dance1_subject1"])
    return MotionLibrary(files, TRACKED_BODIES, torch.zeros(29), device="cpu", freeze_end=True)


@needs_npz
@pytest.mark.parametrize("mode", ["sonic", "v1", "traj"])
def test_goal_window_freezes_past_the_clip_end(cpu_lib, frozen_lib, mode):
    """Past the clip end the goal is the last frame held still: poses as before, velocities zero.
    Inside the clip nothing changes."""
    lib, _ = cpu_lib
    last = int(lib.clip_last[0])
    f = torch.tensor([100, last - 3, last])
    old = lib.goal_window(f, 10, mode).reshape(3, 10, -1)
    new = frozen_lib.goal_window(f, 10, mode).reshape(3, 10, -1)
    torch.testing.assert_close(new[0], old[0])  # far from the end
    torch.testing.assert_close(new[1, :4], old[1, :4])  # frames last-3 .. last
    torch.testing.assert_close(new[2, 0], old[2, 0])
    # Per-frame velocity columns of each layout: sonic [q 29 | qd 29 | z]; v1/traj [q 29 | qd 29 | grav 3 | w_b 3 | v_h 3 | z | ...].
    vel = torch.zeros(new.shape[-1], dtype=torch.bool)
    vel[29:58] = True
    if mode != "sonic":
        vel[61:67] = True
    for row, start in ((1, 4), (2, 1)):
        torch.testing.assert_close(new[row, start:][..., vel], torch.zeros_like(new[row, start:][..., vel]))
        torch.testing.assert_close(new[row, start:][..., ~vel], old[row, start:][..., ~vel])
    # A target already held past the end (beyond > 0): every frame of the window is frozen.
    held = frozen_lib.goal_window(torch.tensor([last]), 10, mode, beyond=torch.tensor([2])).reshape(10, -1)
    torch.testing.assert_close(held[..., vel], torch.zeros_like(held[..., vel]))
    torch.testing.assert_close(held[..., ~vel], old[2, :][..., ~vel])


@needs_npz
def test_rollout_targets_hold_the_last_frame_still(frozen_lib):
    from supermimic.imagination.rollout import rollout_targets
    from supermimic.world_model.wm_trainer import reference_at

    lib = frozen_lib
    last = int(lib.clip_last[0])
    start = torch.tensor([last - 5, 50])
    frames, ref, goal = rollout_targets(lib, start, 8, 10, "sonic")
    assert frames[:, 0].tolist() == [last - 5 + t for t in range(6)] + [last] * 3
    plain = reference_at(lib, frames.reshape(-1))
    lin = ref.body_lin_vel_w.reshape(9, 2, -1, 3)
    held = torch.zeros(9, 2, dtype=torch.bool)
    held[6:, 0] = True  # targets past the end of clip 0
    assert torch.all(lin[held] == 0) and torch.all(ref.body_ang_vel_w.reshape(9, 2, -1, 3)[held] == 0)
    torch.testing.assert_close(lin[~held], plain.body_lin_vel_w.reshape(9, 2, -1, 3)[~held])
    torch.testing.assert_close(ref.body_pos_w, plain.body_pos_w)
    # The goal at the first held target equals the env-side window of the last frame one step past it.
    torch.testing.assert_close(goal[6, 0], lib.goal_window(torch.tensor([last]), 10, "sonic", beyond=torch.tensor([1]))[0])
    # Inside the clip, the goal is the env's goal window at that frame.
    torch.testing.assert_close(goal[2, 1], lib.goal_window(torch.tensor([52]), 10, "sonic")[0])
