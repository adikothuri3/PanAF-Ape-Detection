"""Read PanAf500 frame-level ground-truth annotations.

Schema, **verified against real deposit files** rather than assumed:

.. code-block:: json

    {"video": "035FQHMNRk",
     "annotations": [{"frame_id": 1,
                      "detections": [{"bbox": [144.0, 0.0, 491.0, 397.0],
                                      "ape_id": 0,
                                      "species": "chimpanzee",
                                      "behaviour": "climbing_up"}]}]}

Two traps live in this module, deliberately, so they are fixed in exactly one
place and covered by tests:

1. **``frame_id`` is 1-based** (1..360). Everything else in this project indexes
   frames from 0. An unconverted join silently misaligns every box by one frame,
   which looks like a mediocre detector rather than a bug.
2. **Behaviour labels use underscores on disk** (``climbing_up``), while the
   project's onboarding documentation writes them as prose (``climbing up``). A
   naive lookup against the prose form matches nothing, silently.

The file carries no frame dimensions, so they are supplied by the caller from
the decoded video — which is also what makes the resulting records
self-describing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from panaf_ape_detection.types import BoundingBox

__all__ = [
    "BEHAVIOURS",
    "GroundTruthDetection",
    "GroundTruthFrame",
    "canonical_behaviour",
    "load_ground_truth",
    "normalise_behaviour",
]

BEHAVIOURS: Final[tuple[str, ...]] = (
    "sitting",
    "standing",
    "walking",
    "running",
    "climbing_up",
    "climbing_down",
    "hanging",
    "sitting_on_back",
    "camera_interaction",
)
"""The nine action labels, in the on-disk underscore form.

Confirmed present in the deposit's own annotation files. The onboarding document
lists the same nine in prose form; :func:`normalise_behaviour` maps between them.
"""


def normalise_behaviour(label: str) -> str:
    """Return *label* in the canonical on-disk form.

    Accepts either the prose form used in documentation (``"climbing up"``) or
    the underscore form used in the files (``"climbing_up"``), so a lookup never
    silently misses because of which spelling the caller happened to have.

    Args:
        label: A behaviour label in either spelling.

    Returns:
        The underscore form, lower-cased and stripped.
    """
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def canonical_behaviour(label: str) -> str:
    """Return the human-readable prose form, for captions and figures."""
    return normalise_behaviour(label).replace("_", " ")


class GroundTruthDetection(BaseModel):
    """One annotated ape in one frame.

    Distinct from :class:`~panaf_ape_detection.types.Detection`: this is what a
    human recorded, so it has no confidence score, and it carries information no
    detector can produce -- species and an individual identity.

    Attributes:
        box: Annotated region, ``xyxy`` in absolute pixels.
        ape_id: Identity of the individual **within this video only**.
        species: ``chimpanzee`` or ``gorilla``, as recorded by the dataset.
        behaviour: One of :data:`BEHAVIOURS`, in underscore form.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    box: BoundingBox
    ape_id: int = Field(ge=0)
    species: str = Field(min_length=1)
    behaviour: str = Field(min_length=1)


class GroundTruthFrame(BaseModel):
    """Every annotated ape in one frame, indexed from zero.

    Attributes:
        clip_id: Clip identifier, matching the manifest.
        frame_index: **Zero-based** index, converted from the file's 1-based
            ``frame_id``.
        frame_width: Frame width in pixels, from the decoded video.
        frame_height: Frame height in pixels, from the decoded video.
        detections: Annotated apes. May be empty -- some frames genuinely
            contain no ape, which is what makes false positives measurable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    clip_id: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    detections: tuple[GroundTruthDetection, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Whether this frame contains no annotated ape."""
        return not self.detections


def load_ground_truth(
    path: Path | str,
    *,
    frame_width: int,
    frame_height: int,
    clip_id: str | None = None,
    clamp_to_frame: bool = True,
) -> dict[int, GroundTruthFrame]:
    """Load a PanAf500 annotation file, keyed by **zero-based** frame index.

    Args:
        path: The ``<clip>.json`` annotation file.
        frame_width: Decoded frame width, in pixels.
        frame_height: Decoded frame height, in pixels.
        clip_id: Override the clip id. Defaults to the file's ``video`` field.
        clamp_to_frame: Clamp box edges to the frame. Annotated boxes sometimes
            extend a pixel or two beyond the frame; clamping keeps them valid
            without discarding a real annotation. Set ``False`` to see the raw
            values.

    Returns:
        ``{frame_index: GroundTruthFrame}`` with **zero-based** keys.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file is not valid JSON or lacks the expected keys.
    """
    annotation_path = Path(path)
    try:
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{annotation_path} is not valid JSON: {exc}"
        raise ValueError(msg) from exc

    if not isinstance(document, dict) or "annotations" not in document:
        msg = f"{annotation_path} has no 'annotations' key; is this a PanAf500 annotation file?"
        raise ValueError(msg)

    resolved_clip = clip_id or document.get("video") or annotation_path.stem

    frames: dict[int, GroundTruthFrame] = {}
    for entry in document["annotations"]:
        raw_id = entry.get("frame_id")
        if raw_id is None:
            msg = f"{annotation_path}: an annotation entry has no 'frame_id'"
            raise ValueError(msg)

        # THE conversion. `frame_id` counts from 1; everything else here counts
        # from 0. Doing this anywhere else, or twice, misaligns the whole clip.
        frame_index = int(raw_id) - 1
        if frame_index < 0:
            msg = (
                f"{annotation_path}: frame_id {raw_id} is below 1, so it is not 1-based as expected"
            )
            raise ValueError(msg)

        detections: list[GroundTruthDetection] = []
        for box in entry.get("detections", []):
            x_min, y_min, x_max, y_max = (float(v) for v in box["bbox"])
            if clamp_to_frame:
                x_min, x_max = max(0.0, x_min), min(float(frame_width), x_max)
                y_min, y_max = max(0.0, y_min), min(float(frame_height), y_max)
            if x_max <= x_min or y_max <= y_min:
                # Degenerate after clamping: an annotation entirely outside the
                # frame. Skipping is right, but it should not pass unnoticed.
                continue
            detections.append(
                GroundTruthDetection(
                    box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    ape_id=int(box["ape_id"]),
                    species=str(box["species"]),
                    behaviour=normalise_behaviour(str(box["behaviour"])),
                )
            )

        frames[frame_index] = GroundTruthFrame(
            clip_id=resolved_clip,
            frame_index=frame_index,
            frame_width=frame_width,
            frame_height=frame_height,
            detections=tuple(detections),
        )

    return frames
