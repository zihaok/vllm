# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the pinned nightly's Quack converter compatible with CuTe DSL 4.8."""

from importlib.util import find_spec
from pathlib import Path


def main() -> None:
    spec = find_spec("quack")
    if spec is None or spec.origin is None:
        raise RuntimeError("The pinned nightly must provide quack-kernels")
    path = Path(spec.origin).parent / "cute_dsl_utils.py"
    source = path.read_text()
    signature = "def _patched_convert_single_arg(arg, arg_name, arg_type, ctx):"
    replacement = (
        "def _patched_convert_single_arg(arg, arg_name, arg_type, ctx, **kwargs):\n"
        "    # CuTe DSL 4.8 dataclass constants retain their literal ABI values.\n"
        '    if kwargs.get("is_constexpr", False):\n'
        "        return _original_convert_single_arg(\n"
        "            arg, arg_name, arg_type, ctx, **kwargs\n"
        "        )"
    )
    if replacement in source:
        print(f"Quack CuTe DSL compatibility already applied: {path}")
        return
    replacements = {
        signature: replacement,
        "return _original_convert_single_arg(arg, arg_name, type(arg), ctx)": (
            "return _original_convert_single_arg("
            "arg, arg_name, type(arg), ctx, **kwargs)"
        ),
        "return _original_convert_single_arg(arg, arg_name, arg_type, ctx)": (
            "return _original_convert_single_arg("
            "arg, arg_name, arg_type, ctx, **kwargs)"
        ),
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Unexpected Quack converter in {path}: {before}")
        source = source.replace(before, after)
    compile(source, str(path), "exec")
    path.write_text(source)
    print(f"Applied Quack CuTe DSL argument compatibility: {path}")


if __name__ == "__main__":
    main()
