"""Re-run tracking over saved detections, and sweep tracker settings.

**The whole point is that this costs no GPU.** ``artifacts/detections/`` already
holds every box the detector produced, so association can be replayed with
different settings as many times as it takes. Detection is the expensive stage
and it is finished; tracking is arithmetic over boxes.

Two consequences shape the module:

* It never imports a detector, never opens a video, and never needs an
  accelerator. A sweep runs on a laptop.
* It must reproduce the pipeline exactly when given the pipeline's settings,
  otherwise a sweep optimises something the pipeline will not do. The stage
  order here and in :mod:`panaf_ape_detection.pipeline.runner` is the same, and
  both narrow to :class:`~panaf_ape_detection.types.TrackedDetection` the same
  way.

Track ids stored in the saved documents are **discarded and recomputed**, so
results always match the settings reported alongside them.

.. warning::
   The saved detections were filtered at the detector's confidence threshold
   before being written, so boxes below it were never recorded.
   :attr:`RetrackSettings.detection_floor` can therefore only raise that
   threshold, never lower it. Exploring below it needs a fresh detection run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

from panaf_ape_detection.config import Config
from panaf_ape_detection.data.annotations import GroundTruthFrame, load_ground_truth
from panaf_ape_detection.evaluation.tracking import ClipTrackEvaluation, evaluate_tracking
from panaf_ape_detection.reporting import detection_fields
from panaf_ape_detection.types import Detection, TrackedDetection

__all__ = [
    "TRACKING_SWEEP_SCHEMA",
    "ClipSource",
    "RetrackSettings",
    "SweepArm",
    "expand_grid",
    "finalise_tracks",
    "load_clip",
    "retrack_document",
    "sweep",
    "track_clip",
    "track_clips",
]

logger = logging.getLogger(__name__)

TRACKING_SWEEP_SCHEMA = "panaf.tracking-sweep/v1"
"""Written into every sweep record, so one can be identified in isolation."""


@dataclass(frozen=True, slots=True)
class RetrackSettings:
    """Every knob a re-tracking run can vary.

    Deliberately a flat record of plain numbers rather than a
    :class:`~panaf_ape_detection.config.Config`: a sweep arm has to be
    serialisable into its result file, comparable against every other arm, and
    cheap to send to a worker process.

    Attributes:
        activation_threshold: Score at which a detection participates fully in
            association.
        lost_track_buffer: Frames a track survives unmatched, at 30 fps.
        minimum_matching_threshold: IoU-distance threshold for association.
        minimum_consecutive_frames: Frames a new track must be seen in before it
            is reported.
        minimum_track_length: Tracks shorter than this are dropped afterwards.
        detection_floor: Discard saved detections scoring below this before
            tracking. ``0.0`` uses every box on disk. Can only ever raise the
            threshold the detections were written at.
        score_floor: Rescale scores into ``[score_floor, 1]`` before association
            so ByteTrack's hardcoded 0.1 floor discards nothing. ``0.0`` is off.
        stitch_max_gap: Frames a gap may span when joining two fragments.
        stitch_max_distance: Positional tolerance for that join, in diagonals.
        interpolate_max_gap: Interior gap length to fill with marked boxes.
        smooth_window: Odd window to average boxes over; ``1`` is off.
    """

    activation_threshold: float = 0.2
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 1
    minimum_track_length: int = 5
    detection_floor: float = 0.0
    score_floor: float = 0.0
    stitch_max_gap: int = 0
    stitch_max_distance: float = 1.0
    interpolate_max_gap: int = 0
    smooth_window: int = 1

    @classmethod
    def from_config(cls, config: Config) -> RetrackSettings:
        """Build the settings a configured pipeline run would use.

        Args:
            config: A loaded configuration.

        Returns:
            The equivalent settings, so a sweep's baseline arm is the pipeline.
        """
        return cls(
            activation_threshold=config.tracking.resolved_activation_threshold(
                config.model.confidence_threshold
            ),
            lost_track_buffer=config.tracking.lost_track_buffer,
            minimum_matching_threshold=config.tracking.minimum_matching_threshold,
            minimum_consecutive_frames=config.tracking.minimum_consecutive_frames,
            minimum_track_length=config.tracking.minimum_track_length,
            score_floor=config.tracking.score_floor or 0.0,
            stitch_max_gap=config.tracking.stitch_max_gap,
            stitch_max_distance=config.tracking.stitch_max_distance,
            interpolate_max_gap=config.tracking.interpolate_max_gap,
            smooth_window=config.tracking.smooth_window,
        )

    def with_values(self, values: Mapping[str, float | int]) -> RetrackSettings:
        """Return a copy with *values* applied.

        Args:
            values: Field names mapped to new values.

        Returns:
            The updated settings.

        Raises:
            ValueError: If a name is not a settable field. A typo in a sweep
                grid would otherwise sweep nothing and look like a null result.
        """
        unknown = set(values) - set(self.as_dict())
        if unknown:
            msg = (
                f"unknown tracker setting(s): {sorted(unknown)}. "
                f"Valid names are {sorted(self.as_dict())}."
            )
            raise ValueError(msg)
        return replace(self, **values)  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly record of every setting."""
        return {
            "activation_threshold": self.activation_threshold,
            "lost_track_buffer": self.lost_track_buffer,
            "minimum_matching_threshold": self.minimum_matching_threshold,
            "minimum_consecutive_frames": self.minimum_consecutive_frames,
            "minimum_track_length": self.minimum_track_length,
            "detection_floor": self.detection_floor,
            "score_floor": self.score_floor,
            "stitch_max_gap": self.stitch_max_gap,
            "stitch_max_distance": self.stitch_max_distance,
            "interpolate_max_gap": self.interpolate_max_gap,
            "smooth_window": self.smooth_window,
        }


@dataclass(frozen=True, slots=True)
class ClipSource:
    """Where one clip's saved detections and annotations live.

    Paths rather than loaded data, so a sweep can hand this to a worker process
    without pickling every box in the dataset.

    Attributes:
        clip_id: Manifest identifier.
        detections_path: ``artifacts/detections/<clip>.json``.
        annotation_path: The dataset's annotation JSON for this clip.
    """

    clip_id: str
    detections_path: Path
    annotation_path: Path


def load_clip(source: ClipSource) -> tuple[dict[str, Any], dict[int, GroundTruthFrame]]:
    """Read one clip's detections document and its ground truth.

    Frame dimensions come from the detections document rather than from the
    video, so no clip file has to be present.

    Args:
        source: Where to read from.

    Returns:
        ``(document, ground truth by frame index)``.
    """
    document = json.loads(source.detections_path.read_text(encoding="utf-8"))
    video = document["video"]
    truth = load_ground_truth(
        source.annotation_path,
        frame_width=int(video["width"]),
        frame_height=int(video["height"]),
        clip_id=source.clip_id,
    )
    return document, truth


def retrack_document(
    document: Mapping[str, Any], settings: RetrackSettings
) -> dict[int, list[TrackedDetection]]:
    """Track one clip's saved detections under *settings*.

    Args:
        document: A parsed ``artifacts/detections/<clip>.json``.
        settings: Tracker settings for this arm.

    Returns:
        ``{frame_index: tracked detections}``, after short tracks are dropped.
    """
    from panaf_ape_detection.tracking.bytetrack import ByteTrackTracker

    video = document["video"]
    tracker = ByteTrackTracker(
        activation_threshold=settings.activation_threshold,
        frame_rate=float(video["fps"]) or 24.0,
        lost_track_buffer=settings.lost_track_buffer,
        minimum_matching_threshold=settings.minimum_matching_threshold,
        minimum_consecutive_frames=settings.minimum_consecutive_frames,
        score_floor=settings.score_floor or None,
        frame_width=int(video["width"]),
        frame_height=int(video["height"]),
    )

    # Frame order matters: a tracker fed frames out of order produces
    # meaningless motion estimates, and JSON preserves file order only.
    tracked: dict[int, list[TrackedDetection]] = {}
    for frame in sorted(document["frames"], key=lambda f: int(f["frame_index"])):
        detections = [
            detection
            # Rebuilt as plain Detections, so any stored track_id is ignored and
            # the result matches the settings rather than an earlier run.
            for detection in (Detection(**detection_fields(d)) for d in frame["detections"])
            if detection.confidence >= settings.detection_floor
        ]
        # Frames with no detections must still reach the tracker, or its
        # lost-track buffer never expires.
        tracked[int(frame["frame_index"])] = tracker.update(detections)

    return finalise_tracks(tracked, settings)


def finalise_tracks(
    tracked: Mapping[int, Sequence[TrackedDetection]], settings: RetrackSettings
) -> dict[int, list[TrackedDetection]]:
    """Turn raw tracker output into the pipeline's final tracks.

    ``stitch -> drop short -> interpolate -> smooth``. **The single place this
    chain lives**, called by ``detect`` and by ``track`` alike.

    That matters more than it looks. This chain was originally applied only on
    the re-tracking path, so every measurement went through it while every
    artifact -- the detections cache, the annotated video -- did not. The
    refinement settings in ``configs/base.yaml`` were accepted, validated, and
    silently ignored by the command that produces the deliverables, and the two
    paths disagreed about what the pipeline even was.

    Stitching runs *before* the length filter deliberately: the 2026-07-26
    findings record ``minimum_track_length: 5`` deleting 286 detections, 169 of
    them true positives, and a fragment that joins into a long track should not
    be judged by the length it had alone.

    Args:
        tracked: Raw per-frame output from the tracker.
        settings: The settings this run is using.

    Returns:
        The finished tracks.
    """
    from panaf_ape_detection.tracking.bytetrack import drop_short_tracks
    from panaf_ape_detection.tracking.refine import refine, stitch_tracks

    stitched = stitch_tracks(
        tracked,
        max_gap=settings.stitch_max_gap,
        max_distance=settings.stitch_max_distance,
    )
    shortened = dict(drop_short_tracks(stitched, settings.minimum_track_length))
    return refine(
        shortened,
        stitch_max_gap=0,  # already applied, before the length filter
        interpolate_max_gap=settings.interpolate_max_gap,
        smooth_window=settings.smooth_window,
    )


def track_clip(
    clip_id: str,
    document: Mapping[str, Any],
    truth: Mapping[int, GroundTruthFrame],
    settings: RetrackSettings,
) -> tuple[ClipTrackEvaluation, dict[int, list[TrackedDetection]]]:
    """Track one clip and evaluate it, returning the tracks as well as the score.

    The tracks are returned because some callers need to *write* them -- the
    refined boxes are what Phase 2 pose will consume, and evaluating detection
    accuracy on the tracked output needs them too. A sweep ignores them.

    Args:
        clip_id: Manifest identifier.
        document: A parsed ``artifacts/detections/<clip>.json``.
        truth: Ground truth by frame index.
        settings: Tracker settings for this arm.

    Returns:
        ``(evaluation, tracked frames)``.
    """
    tracked = retrack_document(document, settings)
    return evaluate_tracking(clip_id, tracked, truth), tracked


def track_clips(
    clips: Iterable[tuple[str, Mapping[str, Any], Mapping[int, GroundTruthFrame]]],
    settings: RetrackSettings,
) -> list[ClipTrackEvaluation]:
    """Track and evaluate several clips under one set of settings.

    Args:
        clips: ``(clip_id, detections document, ground truth)`` per clip.
        settings: Tracker settings for this arm.

    Returns:
        One evaluation per clip, in the order given.
    """
    return [track_clip(clip_id, document, truth, settings)[0] for clip_id, document, truth in clips]


@dataclass(frozen=True, slots=True)
class SweepArm:
    """One point in a sweep, with what it scored.

    Attributes:
        settings: The settings used.
        evaluations: Per-clip track quality under them.
    """

    settings: RetrackSettings
    evaluations: tuple[ClipTrackEvaluation, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly record of the arm and its per-clip results."""
        return {
            "settings": self.settings.as_dict(),
            "clips": [evaluation.as_dict() for evaluation in self.evaluations],
        }


def expand_grid(
    base: RetrackSettings, axes: Mapping[str, Sequence[float | int]]
) -> list[RetrackSettings]:
    """Expand a mapping of axes into every combination, applied to *base*.

    Args:
        base: Settings every arm starts from; axes override its fields.
        axes: Field name mapped to the values to try.

    Returns:
        One settings object per combination, in a deterministic order. An empty
        *axes* yields ``[base]`` -- a sweep of one arm, not of none.

    Raises:
        ValueError: If an axis names an unknown field, or lists no values.
    """
    if not axes:
        return [base]

    empty = sorted(name for name, values in axes.items() if not values)
    if empty:
        msg = f"sweep axis/axes {empty} list no values, so nothing would be tried."
        raise ValueError(msg)

    names = sorted(axes)
    return [
        base.with_values(dict(zip(names, combination, strict=True)))
        for combination in product(*(axes[name] for name in names))
    ]


# Set once per worker process by `_init_worker`, so the detection documents are
# parsed once rather than once per arm, and never pickled between processes.
_WORKER_CLIPS: list[tuple[str, dict[str, Any], dict[int, GroundTruthFrame]]] = []


def _init_worker(sources: Sequence[ClipSource]) -> None:
    """Load every clip into this worker, once."""
    global _WORKER_CLIPS
    _WORKER_CLIPS = [(source.clip_id, *load_clip(source)) for source in sources]


def _run_arm(settings: RetrackSettings) -> SweepArm:
    """Evaluate one arm against this worker's preloaded clips."""
    return SweepArm(settings=settings, evaluations=tuple(track_clips(_WORKER_CLIPS, settings)))


def sweep(
    sources: Sequence[ClipSource],
    settings: Sequence[RetrackSettings],
    *,
    jobs: int = 1,
) -> Iterator[SweepArm]:
    """Evaluate every arm over every clip, yielding results as they finish.

    Args:
        sources: Clips to track.
        settings: One entry per arm, as :func:`expand_grid` produces.
        jobs: Worker processes. ``1`` runs in this process, which keeps
            tracebacks readable; more parallelises across arms, each worker
            holding its own copy of the parsed clips.

    Yields:
        One :class:`SweepArm` per entry in *settings*. Order is preserved.
    """
    if jobs <= 1:
        clips = [(source.clip_id, *load_clip(source)) for source in sources]
        for index, arm_settings in enumerate(settings, start=1):
            logger.info("arm %d/%d: %s", index, len(settings), arm_settings.as_dict())
            yield SweepArm(
                settings=arm_settings, evaluations=tuple(track_clips(clips, arm_settings))
            )
        return

    with ProcessPoolExecutor(
        max_workers=jobs, initializer=_init_worker, initargs=(list(sources),)
    ) as pool:
        yield from pool.map(_run_arm, list(settings))
