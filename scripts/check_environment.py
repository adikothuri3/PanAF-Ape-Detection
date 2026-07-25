#!/usr/bin/env python3
"""Report the runtime environment, without requiring the package to be installed.

``panaf-phase1 doctor`` is the richer version of this and should be preferred.
This script exists for the two cases where the CLI is not available:

* diagnosing a broken or partial install, where ``import panaf_ape_detection``
  itself is what fails;
* a fresh Colab runtime, before ``pip install`` has run.

It therefore depends only on the standard library and degrades gracefully when
anything optional is missing. It downloads nothing and needs no GPU.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CORE_PACKAGES: tuple[str, ...] = ("pydantic", "typer", "rich", "yaml")
INFERENCE_PACKAGES: tuple[str, ...] = (
    "PytorchWildlife",
    "torch",
    "torchvision",
    "cv2",
    "supervision",
    "numpy",
    "pandas",
    "imageio",
)

MINIMUM_PYTHON = (3, 11)


def _module_version(module_name: str) -> str:
    """Return an installed module's version, or a status marker."""
    try:
        if importlib.util.find_spec(module_name) is None:
            return "not installed"
    except (ImportError, ValueError):
        return "not installed"

    from importlib.metadata import PackageNotFoundError, version

    distribution = {
        "cv2": "opencv-python-headless",
        "yaml": "PyYAML",
        "PytorchWildlife": "pytorchwildlife",
    }.get(module_name, module_name)
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "installed (version unknown)"


def _report(title: str, rows: list[tuple[str, str]]) -> None:
    """Print a simple two-column section."""
    print(f"\n{title}")
    print("-" * len(title))
    width = max((len(name) for name, _ in rows), default=0)
    for name, value in rows:
        print(f"  {name.ljust(width)}  {value}")


def _accelerators() -> list[tuple[str, str]]:
    """Report CUDA and MPS availability, importing torch only when present."""
    if importlib.util.find_spec("torch") is None:
        return [("CUDA", "unknown (torch not installed)"), ("Apple MPS", "unknown")]
    try:
        import torch
    except Exception as exc:
        return [("CUDA", f"torch import failed: {exc}"), ("Apple MPS", "unknown")]

    rows = [("CUDA", "available" if torch.cuda.is_available() else "not available")]
    try:
        mps = "available" if torch.backends.mps.is_available() else "not available"
    except AttributeError:
        mps = "not available"
    rows.append(("Apple MPS", mps))
    return rows


def main() -> int:
    """Print the environment report.

    Returns:
        ``0`` always when the interpreter is new enough, ``1`` if it is too old.
    """
    print("panaf-ape-detection environment check")
    print("(for the full report use: uv run panaf-phase1 doctor)")

    _report(
        "Interpreter",
        [
            ("Python", platform.python_version()),
            ("Executable", sys.executable),
            ("Implementation", platform.python_implementation()),
        ],
    )
    _report(
        "System",
        [
            ("OS", f"{platform.system()} {platform.release()}"),
            ("Machine", platform.machine()),
            ("Repository root", str(REPO_ROOT)),
            ("FFmpeg", shutil.which("ffmpeg") or "not found on PATH"),
        ],
    )
    _report("Accelerators", _accelerators())
    _report("Core packages", [(name, _module_version(name)) for name in CORE_PACKAGES])
    _report(
        "Inference extra (optional)",
        [(name, _module_version(name)) for name in INFERENCE_PACKAGES],
    )

    print()
    if sys.version_info < MINIMUM_PYTHON:
        required = ".".join(str(part) for part in MINIMUM_PYTHON)
        print(f"ERROR: Python {required}+ is required; this interpreter is too old.")
        return 1

    missing_core = [name for name in CORE_PACKAGES if _module_version(name) == "not installed"]
    if missing_core:
        print(f"Core packages missing: {', '.join(missing_core)}. Run `uv sync`.")
    missing_inference = [
        name for name in INFERENCE_PACKAGES if _module_version(name) == "not installed"
    ]
    if missing_inference:
        print(
            "Inference extra not fully installed "
            f"({len(missing_inference)} of {len(INFERENCE_PACKAGES)} missing). "
            "Run `uv sync --extra inference` when you are ready to run detection."
        )
    if not missing_core and not missing_inference:
        print("Environment looks complete.")

    print("\nNote: the detection pipeline is not implemented yet; this only inspects setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
