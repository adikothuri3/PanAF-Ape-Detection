"""Track quality against PanAf500's ``ape_id`` ground truth.

**A deliberately small, hand-checkable set of metrics.** MOTA and IDF1 are the
standard summaries, and both are easy to get subtly wrong — off-by-one in the
matching, or a different convention for counting switches, changes the number
without changing anything visible. A metric nobody can verify by hand is not a
measurement, so they are **not implemented here** rather than half-implemented.

What is measured, per ground-truth individual:

* **ID switches** — the individual was matched to predicted track A, then later
  to track B. Each change counts once. This is the metric the Phase 1 write-up
  asks for by name.
* **Fragmentation** — how many distinct predicted tracks that one individual was
  spread across. With detection recall around 0.4, this is where the damage
  shows: an ape the detector loses and re-finds becomes two tracks.
* **Coverage, mostly-tracked, mostly-lost** — what fraction of the individual's
  annotated frames any track covered, and the standard ≥80% / ≤20% buckets.

Frames are associated by IoU, reusing
:func:`~panaf_ape_detection.evaluation.detection.match_frame`, so detection and
tracking metrics agree about what counts as the same animal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

from panaf_ape_detection.data.annotations import GroundTruthFrame
from panaf_ape_detection.evaluation.detection import DEFAULT_IOU_THRESHOLD, intersection_over_union
from panaf_ape_detection.reporting import TRACKING_METRICS_SCHEMA
from panaf_ape_detection.types import TrackedDetection

__all__ = [
    "ClipTrackEvaluation",
    "IndividualTrackEvaluation",
    "evaluate_tracking",
]

MOSTLY_TRACKED = 0.8
MOSTLY_LOST = 0.2


@dataclass(frozen=True, slots=True)
class IndividualTrackEvaluation:
    """How well one annotated ape was followed.

    Attributes:
        ape_id: The dataset's identity for this individual, within this clip.
        annotated_frames: Frames in which the individual was annotated.
        covered_frames: Frames in which some predicted track matched it.
        id_switches: Times the matched track id changed.
        track_ids: Distinct predicted track ids it was ever matched to.
    """

    ape_id: int
    annotated_frames: int
    covered_frames: int
    id_switches: int
    track_ids: frozenset[int] = frozenset()

    @property
    def coverage(self) -> float:
        """Fraction of annotated frames some track covered."""
        return self.covered_frames / self.annotated_frames if self.annotated_frames else 0.0

    @property
    def fragmentation(self) -> int:
        """Distinct predicted tracks this one individual was split across.

        ``1`` is ideal. ``0`` means it was never tracked at all.
        """
        return len(self.track_ids)

    @property
    def is_mostly_tracked(self) -> bool:
        """Whether at least 80% of its annotated frames were covered."""
        return self.coverage >= MOSTLY_TRACKED

    @property
    def is_mostly_lost(self) -> bool:
        """Whether at most 20% of its annotated frames were covered."""
        return self.coverage <= MOSTLY_LOST

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly summary."""
        return {
            "ape_id": self.ape_id,
            "annotated_frames": self.annotated_frames,
            "covered_frames": self.covered_frames,
            "coverage": round(self.coverage, 4),
            "id_switches": self.id_switches,
            "fragmentation": self.fragmentation,
        }


@dataclass(frozen=True, slots=True)
class ClipTrackEvaluation:
    """Track quality for one clip.

    Attributes:
        clip_id: Manifest identifier.
        iou_threshold: IoU used to associate a prediction with an annotation.
        individuals: Per-ape results.
        predicted_tracks: Distinct track ids the tracker produced.
        frames_evaluated: Frames compared.
    """

    clip_id: str
    iou_threshold: float
    individuals: tuple[IndividualTrackEvaluation, ...] = ()
    predicted_tracks: int = 0
    frames_evaluated: int = 0

    @property
    def total_id_switches(self) -> int:
        """ID switches summed over every individual."""
        return sum(i.id_switches for i in self.individuals)

    @property
    def mean_fragmentation(self) -> float:
        """Mean predicted tracks per annotated individual."""
        if not self.individuals:
            return 0.0
        return sum(i.fragmentation for i in self.individuals) / len(self.individuals)

    @property
    def mean_coverage(self) -> float:
        """Mean fraction of each individual's frames that was tracked."""
        if not self.individuals:
            return 0.0
        return sum(i.coverage for i in self.individuals) / len(self.individuals)

    @property
    def mostly_tracked(self) -> int:
        """Individuals covered in at least 80% of their frames."""
        return sum(1 for i in self.individuals if i.is_mostly_tracked)

    @property
    def mostly_lost(self) -> int:
        """Individuals covered in at most 20% of their frames."""
        return sum(1 for i in self.individuals if i.is_mostly_lost)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly summary for ``artifacts/metrics/tracking/``.

        Kept in its own directory, and self-identifying via ``schema``, because
        this payload and :class:`~panaf_ape_detection.evaluation.detection.ClipEvaluation`
        share just enough keys to be confused for one another.
        """
        return {
            "schema": TRACKING_METRICS_SCHEMA,
            "clip_id": self.clip_id,
            "iou_threshold": self.iou_threshold,
            "frames_evaluated": self.frames_evaluated,
            "annotated_individuals": len(self.individuals),
            "predicted_tracks": self.predicted_tracks,
            "total_id_switches": self.total_id_switches,
            "mean_fragmentation": round(self.mean_fragmentation, 3),
            "mean_coverage": round(self.mean_coverage, 4),
            "mostly_tracked": self.mostly_tracked,
            "mostly_lost": self.mostly_lost,
            "individuals": [i.as_dict() for i in self.individuals],
        }


def evaluate_tracking(
    clip_id: str,
    tracked: Mapping[int, Sequence[TrackedDetection]],
    truth: Mapping[int, GroundTruthFrame],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> ClipTrackEvaluation:
    """Measure how well predicted tracks followed the annotated individuals.

    Only frames present in **both** inputs are evaluated, for the same reason as
    detection evaluation: a frame the tracker never saw is not a failure to
    track, and counting it would report a coverage that is really just the
    fraction of frames processed.

    Args:
        clip_id: Clip identifier.
        tracked: ``{frame_index: tracked detections}``, zero-based.
        truth: ``{frame_index: GroundTruthFrame}``, zero-based.
        iou_threshold: Minimum IoU to associate a prediction with an annotation.

    Returns:
        The clip's :class:`ClipTrackEvaluation`.
    """
    # {ape_id: [track_id per covered frame, in frame order]}
    history: dict[int, list[int]] = {}
    annotated: dict[int, int] = {}
    predicted_track_ids: set[int] = set()
    evaluated = 0

    for frame_index in sorted(set(tracked) & set(truth)):
        ground_truth = truth[frame_index]
        predictions = list(tracked.get(frame_index, ()))
        evaluated += 1
        predicted_track_ids.update(p.track_id for p in predictions)

        unclaimed = list(range(len(predictions)))
        for annotation in ground_truth.detections:
            annotated[annotation.ape_id] = annotated.get(annotation.ape_id, 0) + 1

            best_index: int | None = None
            best_iou = 0.0
            for position in unclaimed:
                overlap = intersection_over_union(predictions[position].box, annotation.box)
                if overlap > best_iou:
                    best_index, best_iou = position, overlap

            if best_index is not None and best_iou >= iou_threshold:
                unclaimed.remove(best_index)
                history.setdefault(annotation.ape_id, []).append(predictions[best_index].track_id)

    individuals = tuple(
        IndividualTrackEvaluation(
            ape_id=ape_id,
            annotated_frames=frames,
            covered_frames=len(history.get(ape_id, [])),
            id_switches=_count_switches(history.get(ape_id, [])),
            track_ids=frozenset(history.get(ape_id, [])),
        )
        for ape_id, frames in sorted(annotated.items())
    )

    return ClipTrackEvaluation(
        clip_id=clip_id,
        iou_threshold=iou_threshold,
        individuals=individuals,
        predicted_tracks=len(predicted_track_ids),
        frames_evaluated=evaluated,
    )


def _count_switches(track_ids: Sequence[int]) -> int:
    """Count how many times the matched track id changed.

    Consecutive frames only, in order. A gap where the individual was not
    tracked is **not** itself a switch — resuming with the same id afterwards is
    a success for the tracker's lost-track buffer, and counting it as a switch
    would penalise exactly the behaviour that is wanted.

    Args:
        track_ids: The matched track id for each covered frame, in frame order.

    Returns:
        The number of changes.
    """
    return sum(1 for previous, current in pairwise(track_ids) if previous != current)
