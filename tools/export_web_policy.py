"""Export a training checkpoint for the browser demo (policy.onnx + meta.json, observations built in the graph).

    uv run python tools/export_web_policy.py logs/<run>/ckpt_XXXXXXX.pt site/assets/policy
"""

from __future__ import annotations

import argparse

from realmimic.export.web_policy import export_web_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument("--weights", default="fp16", help="weight encoding, float32 compute: fp32 | fp16 (default) | int<b> (b-bit codes, scale per output channel) | int<b>g<G> (scale per group of G inputs)")
    parser.add_argument("--vectors", help="also write parity inputs/outputs as JSON (JavaScript check)")
    args = parser.parse_args()
    report = export_web_policy(args.checkpoint, args.out_dir, encoding=args.weights, vectors=args.vectors)
    print(f"policy: {report['onnx']} ({report['bytes'] / 1e6:.1f} MB)")
    print("ONNX Runtime vs torch (decoded weights) max abs error: " + ", ".join(f"{k} {v:.2e}" for k, v in report["max_abs_err"].items()))
    print("ONNX Runtime vs torch (trained weights) max abs error: " + ", ".join(f"{k} {v:.2e}" for k, v in report["encoding_max_abs_err"].items()))


if __name__ == "__main__":
    main()
