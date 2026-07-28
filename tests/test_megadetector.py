"""Tests for the MegaDetector adapter that need no weights and no GPU.

The adapter's job is to hand PyTorch-Wildlife the values this project configured,
and the library has twice accepted a value and then not applied it: ``device=``
is stored and ignored, and ``det_conf_thres`` defaults to 0.2 when omitted. Both
failed *silently*, so both are asserted here against observed calls rather than
against the absence of an exception.

The runner is built with ``object.__new__`` and a stub model, because its real
constructor downloads about a gigabyte of weights.
"""

from __future__ import annotations

import logging
from typing import Any

from panaf_ape_detection.inference.megadetector import (
    DEFAULT_DETECTION_THRESHOLD,
    MegaDetectorV6Runner,
)


class _RecordingModel:
    """Stands in for ``PytorchWildlife`` and records how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def single_image_detection(self, _img: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {}


class _Frame:
    """The only thing `detect` asks of a frame is its shape."""

    shape = (64, 64, 3)


def _runner(threshold: float) -> tuple[MegaDetectorV6Runner, _RecordingModel]:
    runner = object.__new__(MegaDetectorV6Runner)
    model = _RecordingModel()
    runner._model = model  # type: ignore[attr-defined]
    runner._confidence_threshold = threshold  # type: ignore[attr-defined]
    runner._frames_seen = 0  # type: ignore[attr-defined]
    runner._seconds_spent = 0.0  # type: ignore[attr-defined]
    return runner, model


def test_detect_passes_the_configured_threshold_to_the_model():
    """Omitting it pins every run to the library's 0.2, whatever the config says.

    This was a real bug: detections were filtered at the configured threshold
    *after* inference but the model was never told, so a sweep down to 0.05
    returned byte-identical results to one at 0.20.
    """
    runner, model = _runner(0.05)

    runner.detect(_Frame())  # type: ignore[arg-type]

    assert model.calls == [{"det_conf_thres": 0.05}]


def test_a_different_threshold_reaches_the_model_unchanged():
    runner, model = _runner(0.45)

    runner.detect(_Frame())  # type: ignore[arg-type]
    runner.detect(_Frame())  # type: ignore[arg-type]

    assert [call["det_conf_thres"] for call in model.calls] == [0.45, 0.45]


def test_the_recorded_library_default_matches_what_the_library_uses():
    """If upstream changes its default, this constant is what goes stale.

    Skipped without the ``inference`` extra, so the default suite stays offline.
    """
    import inspect

    import pytest

    pytest.importorskip("PytorchWildlife", reason="requires the inference extra")
    from PytorchWildlife.models.detection.ultralytics_based import yolov8_base

    signature = inspect.signature(yolov8_base.YOLOV8Base.single_image_detection)
    assert signature.parameters["det_conf_thres"].default == DEFAULT_DETECTION_THRESHOLD


# --------------------------------------------------------------------------- #
# Per-frame logging -- silenced, because 500 clips is 180,000 lines
# --------------------------------------------------------------------------- #


class _Args:
    """Stands in for the Ultralytics predictor's parsed arguments."""

    def __init__(self) -> None:
        self.verbose = True


class _Predictor:
    def __init__(self) -> None:
        self.args = _Args()


def test_ultralytics_per_frame_logging_is_turned_off():
    """`0: 1280x1280 1 animal, 45.2ms`, once per frame, is 180,000 lines at 500 clips.

    It buries the pipeline's own progress reporting, and a Colab cell holding
    that much output slows the browser badly.
    """
    runner = object.__new__(MegaDetectorV6Runner)
    model = _RecordingModel()
    model.predictor = _Predictor()  # type: ignore[attr-defined]
    runner._model = model  # type: ignore[attr-defined]

    runner._silence_per_frame_logging()  # type: ignore[attr-defined]

    assert model.predictor.args.verbose is False  # type: ignore[attr-defined]
    assert logging.getLogger("ultralytics").level >= logging.WARNING


def test_silencing_survives_a_model_with_no_predictor_yet():
    """The predictor is built lazily, so it may legitimately not exist."""
    runner = object.__new__(MegaDetectorV6Runner)
    runner._model = _RecordingModel()  # type: ignore[attr-defined]

    runner._silence_per_frame_logging()  # type: ignore[attr-defined]

    assert logging.getLogger("ultralytics").level >= logging.WARNING
