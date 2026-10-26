"""Compare a SONIC checkout's compile-time constants with ``realmimic.obs.sonic``.

Parses ``policy_parameters.hpp`` (joint maps, default angles, kp/kd, action scale) and the
observation registry of ``g1_deploy_onnx_ref.cpp`` (term dimensions) from any ``gear_sonic_deploy``
directory, e.g. the one built on the robot, and exits non-zero on any difference.

    uv run python tools/check_sonic_constants.py <path/to/gear_sonic_deploy>
"""

from __future__ import annotations

import argparse
import ast
import operator
import re
import sys
from pathlib import Path

SRC = Path("src/g1/g1_deploy_onnx_ref")
HEADER = SRC / "include/policy_parameters.hpp"
REGISTRY = SRC / "src/g1_deploy_onnx_ref.cpp"

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval(expr: str, names: dict[str, float]) -> float:
    """Evaluate a C++ arithmetic constant expression (numbers, known names, + - * /, parens)."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError(f"unsupported expression {expr!r}")

    return ev(ast.parse(expr.strip(), mode="eval").body)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def parse_policy_parameters(text: str) -> dict[str, list[float]]:
    """Scalar ``const double`` values and every ``std::array<T, 29>`` of the header, evaluated."""
    text = _strip_comments(text)
    names: dict[str, float] = {}
    for name, expr in re.findall(r"const\s+double\s+(\w+)\s*=\s*([^;]+);", text):
        names[name] = _eval(expr, names)
    arrays = {}
    pattern = r"const\s+std::array<\s*(?:int|float|double)\s*,\s*29\s*>\s+(\w+)\s*=\s*\{(.*?)\};"
    for name, body in re.findall(pattern, text, flags=re.DOTALL):
        items = [s for s in (p.strip() for p in body.split(",")) if s]
        arrays[name] = [_eval(s, names) for s in items]
    return arrays


def parse_registry_dims(text: str) -> dict[str, int]:
    """Name -> literal dimension of the GetObservationRegistry entries."""
    return {name: int(dim) for name, dim in re.findall(r'\{"(\w+)",\s*(\d+),\s*\[this\]', text)}


def compare(arrays: dict[str, list[float]], dims: dict[str, int] | None) -> list[str]:
    """Human-readable differences between the parsed SONIC values and realmimic.obs.sonic."""
    from realmimic.obs import sonic

    expected = {
        "isaaclab_to_mujoco": (sonic.ISAACLAB_TO_MUJOCO, 0.0),
        "mujoco_to_isaaclab": (sonic.MUJOCO_TO_ISAACLAB, 0.0),
        "default_angles": (sonic.DEFAULT_ANGLES, 1e-12),
        "g1_action_scale": (sonic.ACTION_SCALE, 1e-12),
        "kps": (sonic.KPS, 1e-12),
        "kds": (sonic.KDS, 1e-12),
    }
    errors = []
    for name, (ours, rtol) in expected.items():
        if name not in arrays:
            errors.append(f"{name}: not found in policy_parameters.hpp")
            continue
        theirs = arrays[name]
        for i, (a, b) in enumerate(zip(theirs, ours, strict=True)):
            if abs(a - b) > rtol * max(abs(a), abs(b)):
                errors.append(
                    f"{name}[{i}] ({sonic.JOINT_NAMES_MUJOCO[i]}): SONIC {a!r} != ours {b!r}"
                )
    if dims is not None:
        for name, dim in sonic.TERM_DIMS.items():
            if name not in dims:
                errors.append(f"term {name}: not in the observation registry")
            elif dims[name] != dim:
                errors.append(f"term {name}: SONIC dim {dims[name]} != ours {dim}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("deploy_dir", type=Path, help="path to gear_sonic_deploy")
    args = parser.parse_args()
    arrays = parse_policy_parameters((args.deploy_dir / HEADER).read_text())
    registry = args.deploy_dir / REGISTRY
    dims = parse_registry_dims(registry.read_text()) if registry.exists() else None
    if dims is None:
        print(f"note: {registry} not found, term dimensions not checked")
    errors = compare(arrays, dims)
    for e in errors:
        print(f"MISMATCH {e}")
    if errors:
        return 1
    parsed = f"{len(arrays)} arrays" + ("" if dims is None else f", {len(dims)} registry terms")
    print(f"OK: realmimic.obs.sonic matches {args.deploy_dir} ({parsed} parsed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
