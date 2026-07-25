#!/usr/bin/env python3
"""End-to-end smoke test **with real MegaDetector weights**. Local only.

This is the gate to clear before touching real PanAf clips. It proves the whole
chain works — config → device → weight download → model load → inference on a
frame → run metadata — so that when Phase 1c fails, the failure is in the new
pipeline code rather than somewhere in the stack underneath it.

**Downloads model weights** (roughly a gigabyte on first run, cached afterwards)
from the Zenodo record PyTorch-Wildlife points at. It therefore never runs in
CI, per the repository's rule that CI downloads no weights.

Runs on a **synthetic frame**, not dataset footage. It asserts the output is
well-formed; it makes **no claim about detection quality**, and a detection on
random noise would be a false positive rather than a success.

Usage::

    make smoke-detect
    uv run --extra inference python scripts/smoke_detect.py --variant MDV6-yolov10-c
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    """Load the configured detector and run it on one synthetic frame.

    Returns:
        ``0`` on success, ``1`` on failure.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "base.yaml",
        help="Config to read the model variant, threshold and device from.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Override model.variant for this run (e.g. MDV6-apa-rtdetr-c).",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Write the run metadata into artifacts/metadata/ instead of only printing it.",
    )
    arguments = parser.parse_args(argv)

    from panaf_ape_detection.config import load_config
    from panaf_ape_detection.provenance import (
        build_run_metadata,
        default_metadata_path,
        write_run_metadata,
    )
    from panaf_ape_detection.runtime import module_device, resolve_device, set_seeds
    from panaf_ape_detection.types import DeviceKind

    print("MegaDetector smoke test (downloads real weights)")

    config = load_config(arguments.config)
    variant = arguments.variant or config.model.variant
    print(f"  config    {arguments.config}")
    print(f"  model     {config.model.model_name} / {variant}")
    print(f"  threshold {config.model.confidence_threshold}")

    set_seeds(config.project.seed)
    device = resolve_device(config.model.device)
    print(f"  device    {config.model.device.value} -> {device.value}")

    if device is DeviceKind.CPU:
        print("  note      running on CPU; this will be slow but is a valid check")

    import numpy as np
    from PytorchWildlife.models import detection as pw

    print(f"\nLoading {variant} (first run downloads weights, please wait)...")
    started = time.perf_counter()
    try:
        detector = pw.MegaDetectorV6(device=device.value, pretrained=True, version=variant)
    except ValueError as exc:
        # The documented trap: both classes ship defaults their own validation rejects.
        print(f"\nFAILED: {exc}")
        print("Valid variant strings are listed in 05 Technical/model.md")
        return 1
    except Exception as exc:
        print(f"\nFAILED to load the model: {type(exc).__name__}: {exc}")
        return 1
    load_seconds = time.perf_counter() - started
    print(f"  loaded in {load_seconds:.1f}s")

    # A synthetic frame at MegaDetector's native input size. Structured noise, not
    # an animal -- this checks the plumbing, never the accuracy.
    rng = np.random.default_rng(config.project.seed)
    frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    print("\nRunning inference on one synthetic 1280x720 frame...")
    started = time.perf_counter()
    try:
        result = detector.single_image_detection(frame)
    except Exception as exc:
        print(f"FAILED during inference: {type(exc).__name__}: {exc}")
        return 1
    infer_seconds = time.perf_counter() - started

    # ------------------------------------------------------------------ #
    # Verify the model is where we asked for it.
    #
    # PyTorch-Wildlife 1.3.0 accepts `device=`, stores it, and never applies it:
    # `yolov8_base._load_model` has `# self.predictor.args.device = device
    # # Will uncomment later`. So a model built with device="cuda" silently runs
    # on CPU. This is the reference pattern for the Phase 1c adapter.
    # ------------------------------------------------------------------ #
    actual = module_device(detector)
    print(f"\nDevice check: requested {device.value!r}, weights on {actual!r}")

    if actual is None:
        # Never treat "could not determine" as success -- that is how the
        # mismatch stays invisible.
        print("  FAILED — could not locate the model's parameters to verify the device.")
        print("  runtime.module_device() needs updating for this library version.")
        return 1

    if not actual.startswith(device.value):
        print(f"  MISMATCH — the library ignored device={device.value!r} (known upstream bug).")
        print("  Applying the workaround (all three lines are required):")
        import torch

        torch_device = torch.device(device.value)
        # 1. move the weights ...
        detector.predictor.model.to(torch_device)
        # 2. ... tell ultralytics' preprocessing where to put the *input*, or
        #    the next forward pass dies with "input(device='cpu') and
        #    weight(device='mps:0') must be on the same device" ...
        detector.predictor.device = torch_device
        # 3. ... and keep args consistent for anything that reads them later.
        detector.predictor.args.device = device.value
        print("    predictor.model.to(device); predictor.device = device; args.device = device")

        moved = module_device(detector)
        if moved is not None and moved.startswith(device.value):
            print(f"  fixed — weights now on {moved!r}")
            started = time.perf_counter()
            result = detector.single_image_detection(frame)
            infer_seconds = time.perf_counter() - started
            print(f"  re-ran on {device.value}: {infer_seconds:.2f}s")
        else:
            print(f"  FAILED — still on {moved!r} after .to({device.value!r})")
            return 1
    else:
        print("  ok — the library honoured the requested device")

    print(f"  inference completed in {infer_seconds:.2f}s")
    print(f"  result type: {type(result).__name__}")
    if isinstance(result, dict):
        for key, value in result.items():
            shape = getattr(value, "shape", None)
            print(f"    {key}: {type(value).__name__}{f' shape={shape}' if shape else ''}")
        detections = result.get("detections")
        count = len(detections) if detections is not None else 0
        print(f"  detections returned: {count}")
        print("  (on random noise, any detection is a false positive -- this checks plumbing only)")

    elapsed = load_seconds + infer_seconds
    metadata = build_run_metadata(config, device=device, elapsed_seconds=elapsed)
    print("\nRun metadata:")
    for field in ("experiment_name", "git_commit", "git_dirty", "device", "model_variant"):
        print(f"  {field}: {getattr(metadata, field)}")
    print(f"  dependency_versions: {metadata.dependency_versions}")

    if arguments.keep_metadata:
        written = write_run_metadata(metadata, default_metadata_path(config))
        print(f"\n  written to {written}")

    print("\nPASSED: weights load and inference runs end to end.")
    print("The stack is ready for real clips. Quality is still entirely unmeasured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
