"""Convert between this project's schema and ``supervision.Detections``.

Two different shapes meet here. This project uses one pydantic object per
detection; ``supervision`` uses parallel NumPy arrays (``xyxy``, ``confidence``,
``class_id``, ``tracker_id``). Every conversion lives in this module so a
column-order or dtype mistake is made once, in one place, rather than spreading.

``supervision`` is imported lazily inside the functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from panaf_ape_detection.types import BoundingBox, Detection, TrackedDetection

if TYPE_CHECKING:  # pragma: no cover - typing only
    import supervision as sv

__all__ = ["from_supervision", "to_supervision"]

_DEFAULT_CATEGORY = "animal"


def to_supervision(detections: Sequence[Detection]) -> sv.Detections:
    """Pack detections into a ``supervision.Detections``.

    Args:
        detections: Detections to convert. May be empty.

    Returns:
        An equivalent ``sv.Detections``. An empty input gives
        ``sv.Detections.empty()``, which ByteTrack accepts -- frames with no
        detection still have to be fed to the tracker, or it never learns that a
        track went missing.
    """
    import numpy as np
    import supervision as sv

    if not detections:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(
            [[d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max] for d in detections],
            dtype=np.float32,
        ),
        confidence=np.array([d.confidence for d in detections], dtype=np.float32),
        class_id=np.array([d.category_id for d in detections], dtype=int),
    )


def from_supervision(
    detections: sv.Detections,
    *,
    category_names: dict[int, str] | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> list[TrackedDetection]:
    """Unpack a tracked ``supervision.Detections`` into this project's schema.

    Only rows carrying a ``tracker_id`` are returned. An untracked row is a
    detection the tracker declined to confirm, and promoting it to a
    :class:`~panaf_ape_detection.types.TrackedDetection` would mean inventing an
    identity.

    Args:
        detections: Output of ``ByteTrack.update_with_detections``.
        category_names: Maps class ids to names. Defaults to ``animal`` for
            everything, which is right for this pipeline because
            ``keep_animals`` has already filtered the rest out.
        frame_width: Clamp boxes to this width, when given.
        frame_height: Clamp boxes to this height, when given.

    Returns:
        Tracked detections, in the order ``supervision`` returned them.
    """
    names = category_names or {}
    boxes = getattr(detections, "xyxy", None)
    if boxes is None or len(boxes) == 0:
        return []

    tracker_ids = getattr(detections, "tracker_id", None)
    if tracker_ids is None:
        return []

    confidences = getattr(detections, "confidence", None)
    class_ids = getattr(detections, "class_id", None)

    tracked: list[TrackedDetection] = []
    for index, row in enumerate(boxes):
        track_id = tracker_ids[index]
        if track_id is None:
            continue

        x_min, y_min, x_max, y_max = (float(v) for v in row[:4])
        if frame_width is not None:
            x_min, x_max = max(0.0, x_min), min(float(frame_width), x_max)
        if frame_height is not None:
            y_min, y_max = max(0.0, y_min), min(float(frame_height), y_max)
        if x_max <= x_min or y_max <= y_min:
            continue

        category = int(class_ids[index]) if class_ids is not None else 0
        tracked.append(
            TrackedDetection(
                box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                confidence=(
                    min(1.0, max(0.0, float(confidences[index])))
                    if confidences is not None
                    else 1.0
                ),
                category_id=category,
                category_name=names.get(category, _DEFAULT_CATEGORY),
                # ByteTrack counts from 1; the schema requires >= 0, so this is
                # carried through unchanged rather than renumbered. Track ids are
                # only meaningful within one clip anyway.
                track_id=int(track_id),
            )
        )
    return tracked
