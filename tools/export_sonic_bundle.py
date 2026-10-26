"""Export a training checkpoint as a SONIC decoder bundle (policy.onnx + observation_config.yaml +
launch.env + manifest.json).

    uv run python tools/export_sonic_bundle.py logs/<run>/ckpt_XXXXXXX.pt exports/<name>
"""

from __future__ import annotations

import argparse

from realmimic.export.sonic_bundle import export_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    report = export_bundle(args.checkpoint, args.out_dir)
    print(f"bundle: {report.out_dir}")
    print(f"ONNX Runtime vs torch max abs error: {report.max_abs_err:.2e}")
    print(f"runtime flags: {' '.join(report.flags) or '(none)'}")


if __name__ == "__main__":
    main()
