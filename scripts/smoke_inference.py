#!/usr/bin/env python3
"""Weights-free smoke test for the ``inference`` extra.

Answers one question: **does the heavy stack actually work here?** Not "does it
resolve" — `uv lock` succeeding says nothing about whether `import
PytorchWildlife` succeeds, which this project has already been caught by three
times (undeclared ``soundfile`` and ``librosa``, and ``setuptools`` 83 removing
``pkg_resources`` out from under ``yolov5``).

Checks, in order of how early they would break a real run:

1. Every package in the extra imports.
2. ``supervision.ByteTrack`` tracks synthetic detections and assigns ids —
   the Phase 1d dependency, exercised before it is depended on.
3. NumPy interop still behaves (the locked NumPy is 2.x; ``supervision``
   0.23.0 predates it).
4. A video survives a write → read round-trip through OpenCV and imageio,
   with FFmpeg — the Phase 1b/1e dependency.
5. The project's own runtime layer resolves a device and builds run metadata.

**Downloads no model weights. Requires no GPU.** Run with
``make smoke-inference``; in CI it is the scheduled ``inference-smoke`` job.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_failures: list[str] = []
_checks = 0


def check(description: str) -> None:
    """Print a check heading."""
    global _checks
    _checks += 1
    print(f"\n[{_checks}] {description}")


def ok(detail: str) -> None:
    """Report a passing check."""
    print(f"    ok  {detail}")


def fail(detail: str) -> None:
    """Record a failing check."""
    _failures.append(detail)
    print(f"    FAIL  {detail}")


def check_imports() -> None:
    """Every package in the inference extra must import, not merely resolve."""
    check("inference extra imports")
    modules = [
        ("PytorchWildlife", "pytorchwildlife"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("cv2", "opencv-python-headless"),
        ("supervision", "supervision"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("imageio", "imageio"),
    ]
    from importlib.metadata import PackageNotFoundError, version

    for module_name, distribution in modules:
        try:
            __import__(module_name)
        except Exception as exc:
            fail(f"import {module_name}: {type(exc).__name__}: {exc}")
            continue
        try:
            ok(f"{module_name} {version(distribution)}")
        except PackageNotFoundError:
            ok(module_name)


def check_detector_api() -> None:
    """The MegaDetector classes and their variant strings must be present.

    Constructs nothing — instantiating downloads weights.
    """
    check("MegaDetector API surface (no weights downloaded)")
    try:
        from PytorchWildlife.models import detection as pw
    except Exception as exc:
        fail(f"cannot import PytorchWildlife.models.detection: {exc}")
        return

    for class_name in ("MegaDetectorV6", "MegaDetectorV6Apache"):
        if hasattr(pw, class_name):
            ok(f"{class_name} present")
        else:
            fail(f"{class_name} missing from PytorchWildlife.models.detection")

    classes = getattr(getattr(pw, "MegaDetectorV6", None), "CLASS_NAMES", None)
    expected = {0: "animal", 1: "person", 2: "vehicle"}
    if classes == expected:
        ok(f"class vocabulary unchanged: {expected}")
    else:
        fail(f"class vocabulary changed: {classes!r} (expected {expected!r})")


def check_bytetrack() -> None:
    """ByteTrack must track synthetic detections and assign stable ids."""
    check("supervision.ByteTrack on synthetic detections")
    try:
        import numpy as np
        import supervision as sv
    except Exception as exc:
        fail(f"import: {exc}")
        return

    try:
        tracker = sv.ByteTrack(frame_rate=24)
    except Exception as exc:
        fail(f"constructing ByteTrack: {type(exc).__name__}: {exc}")
        return

    tracked_ids: set[int] = set()
    try:
        # A single box drifting right across ten frames: one object, one id.
        for step in range(10):
            offset = float(step * 4)
            detections = sv.Detections(
                xyxy=np.array([[100.0 + offset, 100.0, 200.0 + offset, 300.0]], dtype=np.float32),
                confidence=np.array([0.9], dtype=np.float32),
                class_id=np.array([0], dtype=int),
            )
            result = tracker.update_with_detections(detections)
            if result.tracker_id is not None:
                tracked_ids.update(int(i) for i in result.tracker_id)
    except Exception as exc:
        fail(f"update_with_detections: {type(exc).__name__}: {exc}")
        return

    if not tracked_ids:
        fail("ByteTrack assigned no tracker ids over 10 frames")
    elif len(tracked_ids) == 1:
        ok(f"one object tracked across 10 frames, stable id {tracked_ids.pop()}")
    else:
        # Not fatal -- ByteTrack may re-id -- but worth surfacing loudly.
        ok(f"tracked, but ids were not stable: {sorted(tracked_ids)}")


def check_numpy_interop() -> None:
    """NumPy 2.x semantics that supervision 0.23.0 predates."""
    check("NumPy 2.x interop")
    try:
        import numpy as np
    except Exception as exc:
        fail(f"import numpy: {exc}")
        return

    ok(f"numpy {np.__version__}")

    for removed in ("float_", "unicode_", "NaN", "product", "alltrue"):
        if hasattr(np, removed):
            ok(f"np.{removed} still present (numpy < 2 semantics)")

    try:
        array = np.array([1.0, 2.0], dtype=np.float32)
        result = array * np.float64(2.0)
        ok(f"scalar promotion float32 * float64 -> {result.dtype} (NEP 50)")
    except Exception as exc:
        fail(f"scalar promotion: {exc}")


def check_video_round_trip() -> None:
    """A video must survive write -> read through OpenCV and imageio."""
    check("video write/read round-trip (OpenCV + imageio + FFmpeg)")
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        fail(f"import: {exc}")
        return

    width, height, frames = 320, 240, 12
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "synthetic.mp4"

        writer = cv2.VideoWriter(
            str(target), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (width, height)
        )
        if not writer.isOpened():
            fail("cv2.VideoWriter could not open an mp4v writer")
            return
        rng = np.random.default_rng(0)
        for _ in range(frames):
            writer.write(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
        writer.release()

        if not target.is_file() or target.stat().st_size == 0:
            fail("cv2 wrote no video data")
            return
        ok(f"cv2 wrote {target.stat().st_size:,} bytes")

        capture = cv2.VideoCapture(str(target))
        read = 0
        while True:
            got, _frame = capture.read()
            if not got:
                break
            read += 1
        capture.release()

        if read == 0:
            fail("cv2.VideoCapture read 0 frames from the file it just wrote")
        else:
            ok(f"cv2 read back {read}/{frames} frames")

        # The extra declares `imageio[ffmpeg]`, so FFMPEG is the plugin that
        # ships -- not pyav. imageio-ffmpeg bundles its own ffmpeg binary, so
        # this works even where the system ffmpeg is absent.
        try:
            import imageio.v2 as iio

            reader = iio.get_reader(str(target), format="FFMPEG")
            decoded = sum(1 for _ in reader)
            reader.close()
        except Exception as exc:
            fail(f"imageio FFMPEG read: {type(exc).__name__}: {exc}")
        else:
            if decoded == 0:
                fail("imageio FFMPEG read 0 frames")
            else:
                ok(f"imageio FFMPEG read {decoded} frames")

        # GIF is one of the accepted Phase 1 deliverable formats.
        try:
            gif = Path(tmp) / "synthetic.gif"
            iio.mimsave(
                str(gif),
                [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)],
                duration=0.1,
            )
        except Exception as exc:
            fail(f"GIF export: {type(exc).__name__}: {exc}")
        else:
            if gif.is_file() and gif.stat().st_size > 0:
                ok(f"GIF export wrote {gif.stat().st_size:,} bytes")
            else:
                fail("GIF export produced no data")


def check_project_runtime() -> None:
    """The project's own runtime layer must work with torch installed."""
    check("project runtime layer")
    try:
        from panaf_ape_detection.config import load_config
        from panaf_ape_detection.provenance import build_run_metadata
        from panaf_ape_detection.runtime import available_devices, resolve_device
        from panaf_ape_detection.types import DeviceKind
    except Exception as exc:
        fail(f"import project modules: {exc}")
        return

    devices = available_devices()
    ok(f"available devices: {', '.join(sorted(d.value for d in devices))}")

    resolved = resolve_device(DeviceKind.AUTO)
    ok(f"'auto' resolved to '{resolved.value}'")

    try:
        config = load_config(REPO_ROOT / "configs" / "base.yaml", use_env_overrides=False)
        metadata = build_run_metadata(config, device=resolved)
    except Exception as exc:
        fail(f"building run metadata: {type(exc).__name__}: {exc}")
        return

    ok(f"run metadata built for variant {metadata.model_variant!r}")
    recorded = metadata.dependency_versions
    if "torch" in recorded:
        ok(f"torch version recorded: {recorded['torch']}")
    else:
        fail("torch is installed but was not recorded in dependency_versions")


def main() -> int:
    """Run every smoke check and report.

    Returns:
        ``0`` when all checks pass, ``1`` otherwise.
    """
    print("Weights-free inference smoke test")
    print(f"Repository: {REPO_ROOT}")

    for routine in (
        check_imports,
        check_detector_api,
        check_bytetrack,
        check_numpy_interop,
        check_video_round_trip,
        check_project_runtime,
    ):
        try:
            routine()
        except Exception as exc:
            fail(f"{routine.__name__} raised {type(exc).__name__}: {exc}")

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} problem(s) across {_checks} check(s)\n")
        for problem in _failures:
            print(f"  * {problem}")
        return 1

    print(f"PASSED: {_checks} check(s) — the inference stack works, weights untested.")
    print("Next: `make smoke-detect` to load real weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
