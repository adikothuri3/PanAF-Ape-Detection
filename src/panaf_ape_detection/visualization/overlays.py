"""Draw detections and ground truth onto a frame.

**Predictions and ground truth must never look alike.** A viewer seeing a box
with `climbing_up` beside it will assume the model produced both, and it did
not: MegaDetector emits `animal` and nothing else. So predictions are drawn in
one colour with their confidence, ground truth in another with its dataset
label, and a legend states which is which on every frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from panaf_ape_detection.data.annotations import GroundTruthDetection, canonical_behaviour
from panaf_ape_detection.types import Detection

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = ["GROUND_TRUTH_COLOUR", "PREDICTION_COLOUR", "draw_frame"]

# BGR, because OpenCV.
PREDICTION_COLOUR = (60, 220, 60)  # green -- what the model said
GROUND_TRUTH_COLOUR = (60, 160, 255)  # amber -- what a human recorded
_TEXT_COLOUR = (255, 255, 255)
_SHADOW_COLOUR = (0, 0, 0)


def _label(
    frame: np.ndarray[Any, Any],
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float = 0.4,
) -> None:
    """Draw *text* with a filled backing box so it stays legible on any frame."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(y, height + 4)
    cv2.rectangle(
        frame, (x, y - height - baseline - 2), (x + width + 4, y + 1), colour, thickness=-1
    )
    cv2.putText(frame, text, (x + 2, y - 2), font, scale, _SHADOW_COLOUR, thickness, cv2.LINE_AA)


def draw_frame(
    frame: np.ndarray[Any, Any],
    *,
    detections: Sequence[Detection] = (),
    ground_truth: Sequence[GroundTruthDetection] = (),
    frame_index: int | None = None,
    clip_id: str | None = None,
    draw_confidence: bool = True,
    draw_behaviour: bool = True,
    draw_track_id: bool = True,
) -> np.ndarray[Any, Any]:
    """Return a copy of *frame* with boxes and labels drawn on it.

    Args:
        frame: BGR frame; **not** modified in place.
        detections: Model predictions, drawn in green with confidence.
        ground_truth: Dataset annotations, drawn in amber with the behaviour
            label and individual id.
        frame_index: Zero-based index, shown in the corner.
        clip_id: Clip identifier, shown in the corner.
        draw_confidence: Show each prediction's score.
        draw_behaviour: Show each annotation's behaviour label.
        draw_track_id: Show the tracker's identity on predictions that carry
            one. Tracker ids are only meaningful within a single clip.

    Returns:
        A new annotated frame.
    """
    import cv2

    canvas = frame.copy()

    # Ground truth first, so predictions sit on top where they overlap.
    for truth in ground_truth:
        box = truth.box
        top_left = (int(box.x_min), int(box.y_min))
        bottom_right = (int(box.x_max), int(box.y_max))
        cv2.rectangle(canvas, top_left, bottom_right, GROUND_TRUTH_COLOUR, 1)
        if draw_behaviour:
            text = f"GT ape {truth.ape_id}: {canonical_behaviour(truth.behaviour)}"
            _label(canvas, text, (top_left[0], bottom_right[1] + 14), GROUND_TRUTH_COLOUR)

    for detection in detections:
        box = detection.box
        top_left = (int(box.x_min), int(box.y_min))
        bottom_right = (int(box.x_max), int(box.y_max))
        cv2.rectangle(canvas, top_left, bottom_right, PREDICTION_COLOUR, 2)
        text = detection.category_name
        # `track_id` only exists on TrackedDetection; a plain Detection has none.
        track_id = getattr(detection, "track_id", None)
        if draw_track_id and track_id is not None:
            text = f"#{track_id} {text}"
        if draw_confidence:
            text = f"{text} {detection.confidence:.2f}"
        _label(canvas, text, top_left, PREDICTION_COLOUR)

    # A legend on every frame, because a still pulled out of the video must
    # still be unambiguous about which box came from where.
    height = canvas.shape[0]
    _label(canvas, "green = MegaDetector prediction", (6, height - 20), PREDICTION_COLOUR, 0.38)
    _label(canvas, "amber = dataset ground truth", (6, height - 6), GROUND_TRUTH_COLOUR, 0.38)

    corner = " ".join(
        part
        for part in (clip_id, f"frame {frame_index}" if frame_index is not None else None)
        if part
    )
    if corner:
        _label(canvas, corner, (6, 16), (40, 40, 40), 0.4)

    return canvas
