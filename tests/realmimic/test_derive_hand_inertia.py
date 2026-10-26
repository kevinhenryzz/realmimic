"""tools/derive_hand_inertia.py: SDK pose -> URDF joint angles (mimics included) and posed lumping."""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("derive_hand_inertia", Path(__file__).parents[2] / "tools/derive_hand_inertia.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

# Wrist with a one-link "finger" per actuated joint, plus a mimic distal on the index finger.
_LINK = '<link name="{n}"><inertial><origin xyz="{c}"/><mass value="{m}"/><inertia ixx="0" iyy="0" izz="0" ixy="0" ixz="0" iyz="0"/></inertial></link>'
_JOINT = '<joint name="{n}" type="revolute"><origin xyz="{o}"/><parent link="{p}"/><child link="{c}"/><axis xyz="0 0 1"/><limit lower="0" upper="{u}"/>{mimic}</joint>'


def _urdf(tmp_path: Path) -> Path:
    parts = [_LINK.format(n="left_wrist_yaw_link", c="0 0 0", m=1.0)]
    for name in tool.ACTUATED:
        parts.append(_LINK.format(n=f"left_{name}_link", c="0.1 0 0", m=0.5))
        parts.append(_JOINT.format(n=f"left_{name}_joint", o="0 0 0", p="left_wrist_yaw_link", c=f"left_{name}_link", u=2.0, mimic=""))
    parts.append(_LINK.format(n="left_index_distal_link", c="0 0 0", m=0.0))
    parts.append(_JOINT.format(n="left_index_distal_joint", o="0.1 0 0", p="left_index_proximal_link", c="left_index_distal_link", u=3.0,
                               mimic='<mimic joint="left_index_proximal_joint" multiplier="1.5" offset="0"/>'))
    path = tmp_path / "hand.urdf"
    path.write_text(f'<robot name="t">{"".join(parts)}</robot>')
    return path


def test_pose_maps_onto_limits_and_mimics(tmp_path):
    robot = ET.parse(_urdf(tmp_path)).getroot()
    q = tool.joint_angles(robot, "left", (0.0, 0.5, 1.0, 0.25, 0.0, 0.0))
    assert q["left_thumb_proximal_joint"] == pytest.approx(1.0)
    assert q["left_index_proximal_joint"] == pytest.approx(2.0)
    assert q["left_middle_proximal_joint"] == pytest.approx(0.5)
    assert q["left_index_distal_joint"] == pytest.approx(3.0)  # 1.5 x proximal


def test_posed_lumping_rotates_children(tmp_path):
    path = _urdf(tmp_path)
    robot = ET.parse(path).getroot()
    open_ = tool.lump_subtree(str(path), "left_wrist_yaw_link")
    # Index proximal at 90 degrees about z: its COM moves from +x to +y; the mimic child follows.
    q = tool.joint_angles(robot, "left", (0.0, 0.0, np.pi / 4, 0.0, 0.0, 0.0))
    posed = tool.lump_subtree(str(path), "left_wrist_yaw_link", q)
    assert posed["mass"] == pytest.approx(open_["mass"])
    np.testing.assert_allclose(posed["link_origins"]["left_index_distal_link"], [0.0, 0.1, 0.0], atol=1e-12)
    shift = (np.array(posed["com"]) - np.array(open_["com"])) * posed["mass"]
    np.testing.assert_allclose(shift, [-0.05, 0.05, 0.0], atol=1e-12)


def test_mount_override_moves_the_whole_subtree_along_x(tmp_path):
    """Moving a joint's origin x (the hand mount) shifts its subtree rigidly; the rest stays put."""
    path = _urdf(tmp_path)
    base = tool.lump_subtree(str(path), "left_wrist_yaw_link")
    moved = tool.lump_subtree(str(path), "left_wrist_yaw_link", origins={"left_index_proximal_joint": 0.02})
    assert moved["mass"] == pytest.approx(base["mass"])
    np.testing.assert_allclose(moved["link_origins"]["left_index_distal_link"], [0.12, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(moved["link_origins"]["left_middle_proximal_link"], [0.0, 0.0, 0.0], atol=1e-12)
    # Index proximal (0.5 kg) moves 0.02 m: total mass 4 kg -> COM moves 0.0025 m.
    np.testing.assert_allclose(np.array(moved["com"]) - np.array(base["com"]), [0.0025, 0.0, 0.0], atol=1e-12)
