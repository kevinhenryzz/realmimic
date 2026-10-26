"""SONIC's released policy (encoder + decoder) on our tracking env, with the inputs the runtime builds.

Per CONTROL tick the deploy runtime (``gear_sonic_deploy`` @ ``sonic.SONIC_COMMIT``; ``cpp`` =
``g1_deploy_onnx_ref.cpp``, as in ``docs/sonic-observation-contract.md``) gathers the encoder
input from the reference window and the pelvis IMU (cpp:2022-2146), encodes it into ``token_state``
(cpp:1648-1689), gathers the decoder input ``[token_state, history terms]`` (cpp:2002-2017), and turns
the decoder output into ``q* = default_angles + action * g1_action_scale`` driven with SONIC's
``kps``/``kds`` (cpp:3137-3167). :class:`SonicReleaseActor` does the same for every env of a
benchmark env, batched on the GPU.

The networks are the release ONNX graphs transcribed to torch, with the weights read from the ONNX
files at load time (never copied). The encoder graph only runs at batch 1 (its reshapes are constant
``[1, ...]`` shapes) and onnxruntime is CPU-only here, while both graphs are plain MLPs plus an FSQ
quantizer; ``tests/realmimic/test_sonic_release_policy.py`` checks the transcription against
onnxruntime on the original graphs.

Benchmark episodes start at a clip's first frame with the robot placed in the reference start
state, so this follows the runtime as if each clip were started fresh in normal mode:

* playback is on and the window starts at the env's target frame (``MultiClipMotionCommand.frame``,
  the frame the action is scored against, as in SONIC's training command);
* the heading alignment is captured on each episode's first tick against the clip's frame 0, as a
  heading re-init followed by frame 0 (cpp:559-585, 3413-3427) does;
* histories are back-filled with the first tick's state and zero last actions, as SONIC's training
  env (IsaacLab history buffers) starts every episode. The runtime's zero padding (contract doc
  §2.1) happens only once per program start, with the robot standing in its default pose.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from realmimic.obs import sonic

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

G1_MODE = 0  # encoder_modes "g1" in policy/release/observation_config.yaml
HISTORY_FRAMES = 10  # his_*_10frame_step1
WINDOW_FRAMES, WINDOW_STEP = 10, 5  # motion_*_10frame_step5 (cpp:1737, 1749-1750)
# Required observations of the g1 mode that encoder_terms computes.
G1_TERMS = (
    "encoder_mode_4",
    "motion_joint_positions_10frame_step5",
    "motion_joint_velocities_10frame_step5",
    "motion_anchor_orientation_10frame_step5",
)

# model_encoder.onnx reads the "g1" encoder's input from obs_dict[:, 1:] (/Slice_output_0):
# [3:583] as (10, 58) (/Slice_1, /Reshape_1) and [600:660] as (10, 6) (/Slice_3, /Reshape_3), then
# concatenates them per row into (10, 64) -> 640 (/Concat_2, /g1/Reshape). In the release encoder
# layout those are motion_joint_positions + motion_joint_velocities_10frame_step5 (offsets 4, 294)
# and motion_anchor_orientation_10frame_step5 (offset 601). The (10, 58) rows therefore are not
# per-frame [q, dq]: SONIC's training term does the same reshape of cat(q[10x29], dq[10x29])
# (gear_sonic/envs/manager_env/mdp/observations.py:584-585, commands.py:917).
G1_COMMAND = slice(4, 584)
G1_ANCHOR = slice(601, 661)


@dataclass(frozen=True)
class ReleaseConfig:
    """The parts of ``observation_config.yaml`` the runtime uses with an encoder."""

    decoder_order: list[str]
    token_dim: int
    encoder_order: list[str]
    g1_required: list[str]

    @classmethod
    def load(cls, path: str | Path) -> ReleaseConfig:
        cfg = yaml.safe_load(Path(path).read_text())
        enc = cfg["encoder"]
        modes = {m["mode_id"]: m["required_observations"] for m in enc["encoder_modes"]}
        return cls(
            decoder_order=[o["name"] for o in cfg["observations"] if o.get("enabled") is True],
            token_dim=int(enc["dimension"]),
            encoder_order=[
                o["name"] for o in enc["encoder_observations"] if o.get("enabled") is True
            ],
            g1_required=list(modes[G1_MODE]),
        )


# ---------------------------------------------------------------------------- networks


def _onnx_tensors(path: Path) -> tuple[list, dict[str, Tensor]]:
    """Graph nodes and every initializer / Constant output of an ONNX file, as float32 tensors."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(path)).graph
    values = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            values[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
    return list(graph.node), {k: torch.from_numpy(v.copy()).float() for k, v in values.items()}


def _encoder_layers(nodes: list, values: dict[str, Tensor]) -> list[tuple[Tensor, Tensor]]:
    """(weight (out, in), bias) of the /g1/ Gemm chain (transB=1: torch Linear layout)."""
    gemms = [n for n in nodes if n.op_type == "Gemm" and n.name.startswith("/g1/")]
    return [(values[n.input[1]], values[n.input[2]]) for n in gemms]


def _decoder_layers(nodes: list, values: dict[str, Tensor]) -> list[tuple[Tensor, Tensor]]:
    """(weight (out, in), bias) of the decoder's MatMul + Add chain."""
    adds = {i: n for n in nodes if n.op_type == "Add" for i in n.input}
    layers = []
    for n in nodes:
        if n.op_type == "MatMul":
            add = adds[n.output[0]]
            bias = next(i for i in add.input if i != n.output[0])
            layers.append((values[n.input[1]].T.contiguous(), values[bias]))
    return layers


@contextmanager
def _ieee_fp32() -> Iterator[None]:
    """Full-precision fp32 matmuls, also when the process runs with TF32 enabled (mjlab's
    ``configure_torch_backends``): the runtime builds both engines without FP16 by default
    (``use_fp16: false``, cpp:2170, 2332, 2381)."""
    prev = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = prev


class SonicReleaseNets(torch.nn.Module):
    """``model_encoder.onnx`` (its "g1" branch) and ``model_decoder.onnx``."""

    def __init__(
        self,
        encoder: list[tuple[Tensor, Tensor]],
        quantizer: dict[str, Tensor],
        decoder: list[tuple[Tensor, Tensor]],
    ):
        super().__init__()
        self.encoder = torch.nn.ParameterList(p for layer in encoder for p in layer)
        self.decoder = torch.nn.ParameterList(p for layer in decoder for p in layer)
        for name, value in quantizer.items():
            self.register_buffer(name, value)
        self.requires_grad_(False)

    @classmethod
    def from_onnx(cls, policy_dir: str | Path) -> SonicReleaseNets:
        policy_dir = Path(policy_dir)
        enc_nodes, enc = _onnx_tensors(policy_dir / "model_encoder.onnx")
        dec_nodes, dec = _onnx_tensors(policy_dir / "model_decoder.onnx")
        # /quantizer: FSQ with tanh bound, 32 levels per channel, groups of 32 channels.
        quantizer = {
            "fsq_shift": enc["/quantizer/Constant_1_output_0"],
            "fsq_scale": enc["/quantizer/Constant_2_output_0"],
            "fsq_offset": enc["/quantizer/Constant_3_output_0"],
            "fsq_levels": enc["/quantizer/Constant_4_output_0"],
        }
        return cls(_encoder_layers(enc_nodes, enc), quantizer, _decoder_layers(dec_nodes, dec))

    @staticmethod
    def _mlp(params: torch.nn.ParameterList, x: Tensor) -> Tensor:
        """Linear layers with SiLU (Sigmoid * x in the graphs) between them, none after the last."""
        n = len(params) // 2
        for i in range(n):
            x = F.linear(x, params[2 * i], params[2 * i + 1])
            if i < n - 1:
                x = F.silu(x)
        return x

    def encode(self, obs: Tensor) -> Tensor:
        """Encoder ``obs_dict`` (N, 1762) -> ``encoded_tokens`` (N, 64)."""
        # The graph one-hot-selects the encoder by int(obs[:, 0]) (/Cast, /ScatterND) and sums the
        # three branches weighted by it (/Mul_3, /ReduceSum); only the g1 branch is transcribed.
        if bool((obs[:, 0].long() != G1_MODE).any()):
            raise ValueError("only the g1 encoder mode is implemented")
        n = obs.shape[0]
        command = obs[:, G1_COMMAND].reshape(n, WINDOW_FRAMES, -1)
        anchor = obs[:, G1_ANCHOR].reshape(n, WINDOW_FRAMES, -1)
        with _ieee_fp32():
            z = self._mlp(self.encoder, torch.cat((command, anchor), dim=-1).reshape(n, -1))
        z = z.reshape(n, -1, self.fsq_shift.shape[0])
        z = torch.tanh(z + self.fsq_shift) * self.fsq_scale - self.fsq_offset
        z = z + (torch.round(z) - z)  # straight-through rounding, op for op as exported
        return (z / self.fsq_levels).reshape(n, -1)

    def decode(self, obs: Tensor) -> Tensor:
        """Decoder ``obs_dict`` (N, 994) -> ``action`` (N, 29), IsaacLab order, raw."""
        with _ieee_fp32():
            return self._mlp(self.decoder, obs)


# ---------------------------------------------------------------------------- observation terms


def encoder_terms(
    base_quat: Tensor,
    ref_joint_pos: Tensor,
    ref_joint_vel: Tensor,
    ref_root_quat: Tensor,
    alignment: Tensor,
    playing: bool | Tensor = True,
) -> dict[str, Tensor]:
    """The terms the "g1" encoder mode requires. ``ref_*`` are (N, 10, D) windows of the frames
    ``sonic.future_frame_indices(frame, last, 10, step=5)``, joints in MuJoCo order."""
    mode = base_quat.new_full((base_quat.shape[0],), float(G1_MODE))
    return {
        "encoder_mode_4": sonic.encoder_mode(mode),
        "motion_joint_positions_10frame_step5": sonic.motion_joint_positions(ref_joint_pos),
        "motion_joint_velocities_10frame_step5": sonic.motion_joint_velocities(
            ref_joint_vel, playing
        ),
        "motion_anchor_orientation_10frame_step5": sonic.motion_anchor_orientation(
            base_quat, ref_root_quat, alignment
        ),
    }


def decoder_terms(
    token: Tensor,
    base_ang_vel: Tensor,
    joint_pos: Tensor,
    joint_vel: Tensor,
    last_action: Tensor,
    base_quat: Tensor,
    num_valid: Tensor | None = None,
) -> dict[str, Tensor]:
    """``token_state`` and the five ``his_*_10frame_step1`` terms from (N, 10, D) histories
    (oldest -> newest, MuJoCo order; ``last_action`` = raw actions in MuJoCo order)."""
    return {
        "token_state": token,
        "his_base_angular_velocity_10frame_step1": sonic.his_base_angular_velocity(
            base_ang_vel, num_valid
        ),
        "his_body_joint_positions_10frame_step1": sonic.his_body_joint_positions(
            joint_pos, num_valid
        ),
        "his_body_joint_velocities_10frame_step1": sonic.his_body_joint_velocities(
            joint_vel, num_valid
        ),
        "his_last_actions_10frame_step1": sonic.his_last_actions(last_action, num_valid),
        "his_gravity_dir_10frame_step1": sonic.his_gravity_dir(base_quat, num_valid),
    }


class SonicReleasePolicy:
    """The release policy directory (``gear_sonic_deploy/policy/release``): networks + config."""

    def __init__(self, policy_dir: str | Path, device: str | torch.device = "cuda:0"):
        self.policy_dir = Path(policy_dir).expanduser().resolve()
        self.config = ReleaseConfig.load(self.policy_dir / "observation_config.yaml")
        unsupported = set(self.config.g1_required) - set(G1_TERMS)
        if unsupported:
            raise ValueError(f"g1 encoder mode requires unsupported terms {sorted(unsupported)}")
        self.nets = SonicReleaseNets.from_onnx(self.policy_dir).to(device)

    def encoder_input(self, terms: dict[str, Tensor]) -> Tensor:
        return sonic.assemble_encoder_input(
            terms, self.config.encoder_order, self.config.g1_required
        )

    def decoder_input(self, terms: dict[str, Tensor]) -> Tensor:
        return sonic.assemble_decoder_input(
            terms, self.config.decoder_order, {"token_state": self.config.token_dim}
        )

    def actor(self, env: ManagerBasedRlEnv) -> SonicReleaseActor:
        return SonicReleaseActor(self, env)


class SonicReleaseActor:
    """Runs the release policy on every env of a tracking env (``supermimic.envs``); call once per
    env step, after ``reset()`` / ``step()``, for the env action. The env must run SONIC's action
    scale and default offset (``actuator_gains="sonic"``), so its action is the raw network output
    in MuJoCo order and ``last_action`` / the action-rate reward see SONIC's own units."""

    def __init__(self, policy: SonicReleasePolicy, env: ManagerBasedRlEnv):
        self.policy = policy
        self.env = env
        self.robot = env.scene["robot"]
        self.command = env.command_manager.get_term("motion")
        if tuple(self.robot.joint_names) != sonic.JOINT_NAMES_MUJOCO:
            raise ValueError("robot joints are not in SONIC's MuJoCo (hardware) order")
        term = env.action_manager.get_term("joint_pos")
        scale = term.scale.new_tensor(sonic.ACTION_SCALE).expand_as(term.scale)
        offset = term.offset.new_tensor(sonic.DEFAULT_ANGLES).expand_as(term.offset)
        if not (torch.allclose(term.scale, scale) and torch.allclose(term.offset, offset)):
            raise ValueError("env action term does not use SONIC's action scale / default angles")
        n, dev = env.num_envs, env.device
        self.his = {
            "base_ang_vel": torch.zeros(n, HISTORY_FRAMES, 3, device=dev),
            "joint_pos": torch.zeros(n, HISTORY_FRAMES, 29, device=dev),
            "joint_vel": torch.zeros(n, HISTORY_FRAMES, 29, device=dev),
            "last_action": torch.zeros(n, HISTORY_FRAMES, 29, device=dev),
            "base_quat": torch.zeros(n, HISTORY_FRAMES, 4, device=dev),
        }
        self.action = torch.zeros(n, 29, device=dev)  # previous raw action, MuJoCo order
        self.alignment = torch.zeros(n, 4, device=dev)
        self.alignment[:, 0] = 1.0

    @torch.no_grad()
    def __call__(self) -> Tensor:
        data, cmd, lib = self.robot.data, self.command, self.command.library
        new = self.env.episode_length_buf == 0  # reset since the last call
        self.action[new] = 0.0
        state = {
            "base_ang_vel": data.root_link_ang_vel_b,
            "joint_pos": data.joint_pos,
            "joint_vel": data.joint_vel,
            "last_action": self.action,
            "base_quat": data.root_link_quat_w,
        }
        for key, value in state.items():
            h = self.his[key]
            h[:] = torch.cat((h[:, 1:], value[:, None]), dim=1)
            h[new] = value[new, None].expand(-1, HISTORY_FRAMES, -1)
        if bool(new.any()):
            first = lib.clip_start[lib.frame_clip[cmd.frame[new]]]
            self.alignment[new] = sonic.heading_alignment(
                data.root_link_quat_w[new], lib.ref(first).root_quat_w
            )

        idx = sonic.future_frame_indices(
            cmd.frame, lib.last_frame_of(cmd.frame), WINDOW_FRAMES, WINDOW_STEP
        )
        ref = lib.ref(idx.reshape(-1))
        window = lambda x: x.reshape(idx.shape + x.shape[-1:])
        enc = self.policy.encoder_input(
            encoder_terms(
                data.root_link_quat_w,
                window(ref.joint_pos),
                window(ref.joint_vel),
                window(ref.root_quat_w),
                self.alignment,
            )
        )
        token = self.policy.nets.encode(enc)
        dec = self.policy.decoder_input(
            decoder_terms(
                token,
                self.his["base_ang_vel"],
                self.his["joint_pos"],
                self.his["joint_vel"],
                self.his["last_action"],
                self.his["base_quat"],
            )
        )
        self.action = sonic.isaaclab_to_mujoco(self.policy.nets.decode(dec))
        return self.action
