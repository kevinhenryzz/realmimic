"""Export the training MuJoCo scene for the browser demo (scene.mjb + compressed visual meshes).

    uv run --with meshoptimizer==0.2.30a0 python tools/export_web_scene.py site/assets/robot
    uv run --with meshoptimizer==0.2.30a0 python tools/export_web_scene.py site/assets/robot/variants/g1_brainco --variant g1_brainco \
        --hand-urdf <unitree_ros>/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from realmimic.export.web_scene import export_scene, export_variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir")
    parser.add_argument("--robot", default="g1")
    parser.add_argument("--variant", help="export a robot variant instead (scene.mjb + variant.json), drawn with --robot's meshes")
    parser.add_argument("--hand-urdf", help="URDF with the variant's hand meshes (unitreerobotics/unitree_ros robots/g1_with_brainco_hand)")
    parser.add_argument("--mesh-fraction", type=float, default=1.0, help="simplify meshes to this fraction of faces (1 = full detail)")
    args = parser.parse_args()
    out = Path(args.out_dir)  # files are overwritten; other contents (e.g. variants/) are kept
    if args.variant:
        r = export_variant(out, args.variant, base=args.robot, hand_urdf=args.hand_urdf)
        print(f"variant {args.variant}: {out / 'scene.mjb'} ({r['mjb_bytes'] / 1e6:.2f} MB), hides {r['hide']}, {r['hand_faces']} hand faces ({r['mesh_bytes'] / 1e6:.2f} MB), {r['extras']} stand-in shapes")
        print(f"rollout vs training scene: max |qpos| diff {r['rollout_max_abs_diff']:.1e}")
        return
    report = export_scene(out, robot=args.robot, mesh_fraction=args.mesh_fraction)
    print(f"scene: {out / 'scene.mjb'} ({report['mjb_bytes'] / 1e6:.2f} MB)")
    print(f"visual meshes: {report['faces']} faces, {report['mesh_bytes'] / 1e6:.2f} MB (meshopt)")
    print(f"rollout vs training scene: max |qpos| diff {report['rollout_max_abs_diff']:.1e}")


if __name__ == "__main__":
    main()
