"""Typed structures shared across the (future) pipeline stages.

Nothing in this module performs inference. The detection, track and run-metadata
models are defined up front so that the pipeline stages implemented in later
tasks all agree on one schema, and so that the reproducibility contract in
``docs/reproducibility.md`` has a concrete representation.

.. warning::
   The pipeline does not yet exist. No code in this repository currently
   produces :class:`RunMetadata`; it is a design artefact, not evidence that a
   run has been executed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BoundingBox",
    "Detection",
    "DeviceKind",
    "FrameDetections",
    "InputFileRecord",
    "RunMetadata",
    "TrackedDetection",
    "TrackedFrameDetections",
    "TrackingBackend",
]

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
"""A detector score constrained to the closed unit interval."""


class DeviceKind(StrEnum):
    """Compute device selection understood by the project configuration."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class TrackingBackend(StrEnum):
    """Tracker implementations the project may adopt.

    The concrete backend is deliberately *not* chosen yet. Phase 1 establishes
    the detection baseline first; the tracker is selected once detections are
    known to be reasonable.
    """

    NONE = "none"
    BYTETRACK = "bytetrack"
    SORT = "sort"


class _StrictModel(BaseModel):
    """Base model that rejects unknown fields and validates on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False)


class BoundingBox(_StrictModel):
    """An axis-aligned box in absolute pixel coordinates.

    Coordinates follow the ``xyxy`` convention with the origin at the top-left
    of the frame, matching both MegaDetector output and ``supervision``.

    Attributes:
        x_min: Left edge, in pixels.
        y_min: Top edge, in pixels.
        x_max: Right edge, in pixels; must exceed ``x_min``.
        y_max: Bottom edge, in pixels; must exceed ``y_min``.
    """

    x_min: float = Field(ge=0.0)
    y_min: float = Field(ge=0.0)
    x_max: float = Field(gt=0.0)
    y_max: float = Field(gt=0.0)

    def model_post_init(self, _context: object, /) -> None:
        """Validate that the box has strictly positive width and height."""
        if self.x_max <= self.x_min:
            msg = f"x_max ({self.x_max}) must be greater than x_min ({self.x_min})"
            raise ValueError(msg)
        if self.y_max <= self.y_min:
            msg = f"y_max ({self.y_max}) must be greater than y_min ({self.y_min})"
            raise ValueError(msg)

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    def relative_area(self, frame_width: int, frame_height: int) -> float:
        """Return the box area as a fraction of the frame area.

        The scale-free measure of subject size. ``docs/model.md`` names
        "distant / small subjects" as an expected failure condition, and testing
        that requires comparing box size *relative to the frame* — absolute
        pixel area is meaningless across clips of different resolutions.

        Args:
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            Area fraction in ``(0, 1]`` for a box inside the frame.

        Raises:
            ValueError: If either dimension is not positive.
        """
        if frame_width <= 0 or frame_height <= 0:
            msg = f"frame dimensions must be positive, got {frame_width}x{frame_height}"
            raise ValueError(msg)
        return self.area / (frame_width * frame_height)

    def is_within(self, frame_width: int, frame_height: int) -> bool:
        """Return whether the box lies entirely inside the given frame.

        Args:
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            ``True`` when every edge is inside the frame bounds.
        """
        return self.x_max <= frame_width and self.y_max <= frame_height


class Detection(_StrictModel):
    """A single detector output for one frame.

    Attributes:
        box: Detected region.
        confidence: Detector score in ``[0, 1]``.
        category_id: Integer class id as emitted by the detector.
        category_name: Detector class name. For MegaDetector this is one of
            ``animal``, ``person`` or ``vehicle`` -- *not* a species.
    """

    box: BoundingBox
    confidence: Confidence
    category_id: int = Field(ge=0)
    category_name: str = Field(min_length=1)


class TrackedDetection(Detection):
    """A detection that a tracker has associated with an identity.

    Attributes:
        track_id: Identity assigned by the tracker; stable within one clip only.
        behavior_label: Dataset-provided behaviour label to display alongside the
            box, when one is available. This is never predicted by the detector.
    """

    track_id: int = Field(ge=0)
    behavior_label: str | None = None


class FrameDetections(_StrictModel):
    """All detections for one frame of one clip.

    Carries the frame dimensions so that a serialized record is **self-describing**:
    boxes are absolute pixels, so without the frame size a saved result cannot be
    normalised, bounds-checked, or compared against a clip of another resolution
    without re-opening the source video.

    Attributes:
        clip_id: Manifest identifier of the source clip.
        frame_index: Zero-based index into the decoded frame sequence.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        timestamp_seconds: Presentation timestamp of the frame, when known.
        detections: Detections surviving the configured confidence threshold.
    """

    clip_id: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    timestamp_seconds: float | None = Field(default=None, ge=0.0)
    # `Sequence`, not `list`: `list` is invariant, so a `list[TrackedDetection]`
    # override in the subclass below would be unsound -- code holding this as a
    # `FrameDetections` could append a plain `Detection` into a tracked frame,
    # and pydantic does not validate `.append()`. The read-only interface makes
    # the override type-safe and blocks that mutation.
    detections: Sequence[Detection] = Field(default_factory=list)

    def model_post_init(self, _context: object, /) -> None:
        """Reject any detection whose box escapes the declared frame.

        A box outside the frame means whatever produced it is wrong — a bad
        coordinate convention, a resize that was not applied, a stale frame size.
        Catching that at construction is far cheaper than finding it in a figure
        weeks later.
        """
        for index, detection in enumerate(self.detections):
            if not detection.box.is_within(self.frame_width, self.frame_height):
                msg = (
                    f"detection {index} of clip {self.clip_id!r} frame {self.frame_index} "
                    f"has a box extending to ({detection.box.x_max}, {detection.box.y_max}), "
                    f"outside the declared {self.frame_width}x{self.frame_height} frame"
                )
                raise ValueError(msg)

    def relative_areas(self) -> list[float]:
        """Return each detection's box area as a fraction of the frame."""
        return [
            detection.box.relative_area(self.frame_width, self.frame_height)
            for detection in self.detections
        ]


class TrackedFrameDetections(FrameDetections):
    """One frame's detections after tracking, with identities preserved.

    This class exists because of a real data-loss trap. pydantic serializes to
    the **declared** field type: storing ``TrackedDetection`` objects in
    :class:`FrameDetections`, whose field is ``list[Detection]``, keeps the
    subclass in memory but silently drops ``track_id`` and ``behavior_label`` on
    ``model_dump()``. The identities and behaviour labels — the Phase 1d/1e
    deliverable — would vanish the moment results were written to disk, with no
    error.

    Anything holding tracked results must use this type, not its parent.

    Attributes:
        detections: Detections carrying tracker identities and, where the
            dataset provides one, a behaviour label.
    """

    detections: Sequence[TrackedDetection] = Field(default_factory=list)

    def track_ids(self) -> set[int]:
        """Return the distinct tracker identities present in this frame."""
        return {detection.track_id for detection in self.detections}


class InputFileRecord(_StrictModel):
    """Provenance record for one input file consumed by a run.

    Attributes:
        filename: File name only; absolute paths are deliberately excluded so
            that run metadata does not leak local directory structure.
        sha256: Hex-encoded SHA-256 digest of the file contents.
        size_bytes: File size in bytes.
    """

    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class RunMetadata(_StrictModel):
    """Everything needed to describe and re-run one pipeline execution.

    This is the schema referenced by ``docs/reproducibility.md``. It is defined
    here so that the inference implementation added in the next phase writes a
    stable, reviewable record next to every generated artefact.

    Attributes:
        experiment_name: Human-readable experiment identifier.
        started_at_utc: Timezone-aware UTC start timestamp.
        git_commit: Full commit SHA of the checkout, when available.
        git_dirty: Whether the working tree had uncommitted changes.
        config_snapshot: The fully resolved configuration used for the run.
        python_version: Interpreter version string.
        dependency_versions: Mapping of distribution name to installed version.
        platform: Operating-system description.
        device: Compute device actually selected at runtime.
        model_framework: Framework that loaded the detector.
        model_name: Detector family, e.g. ``MegaDetectorV6``.
        model_variant: Exact weight variant, e.g. ``MDV6-yolov9-c``.
        confidence_threshold: Score threshold applied to detections.
        seed: Random seed used for any stochastic component.
        inputs: Checksummed provenance records for every input file.
        output_paths: Repository-relative paths of the produced artefacts.
        elapsed_seconds: Wall-clock duration of the run.
    """

    experiment_name: str = Field(min_length=1)
    started_at_utc: datetime
    git_commit: str | None = None
    git_dirty: bool | None = None
    config_snapshot: dict[str, object] = Field(default_factory=dict)
    python_version: str = ""
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    platform: str = ""
    device: DeviceKind = DeviceKind.CPU
    model_framework: str = ""
    model_name: str = ""
    model_variant: str = ""
    confidence_threshold: Confidence = 0.0
    seed: int | None = None
    inputs: list[InputFileRecord] = Field(default_factory=list)
    output_paths: list[Path] = Field(default_factory=list)
    elapsed_seconds: float | None = Field(default=None, ge=0.0)
