"""Detection accuracy against ground truth.

Greedy, confidence-ordered IoU matching -- the standard single-threshold scheme,
chosen because it is easy to state exactly and therefore easy to trust:

1. Sort predictions by descending confidence.
2. Each prediction claims the unclaimed ground-truth box it overlaps most, if
   that overlap reaches ``iou_threshold``.
3. Claimed pairs are true positives, unclaimed predictions are false positives,
   unclaimed ground truth are false negatives.

This is **not** mAP. A single threshold answers "at the confidence and IoU we
actually ran, how often was it right?", which is the question that decides
whether fine-tuning is worth it. mAP sweeps thresholds and would hide exactly
the operating point being evaluated.

A caveat that must travel with every number this produces: MegaDetector emits a
generic ``animal`` class, so a detection is counted as correct when it localises
an annotated ape. **No claim about species is made or implied.**
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from panaf_ape_detection.data.annotations import GroundTruthDetection, GroundTruthFrame
from panaf_ape_detection.reporting import DETECTION_METRICS_SCHEMA
from panaf_ape_detection.types import BoundingBox, Detection

__all__ = [
    "ClipEvaluation",
    "FrameMatch",
    "MatchCounts",
    "SizeBand",
    "evaluate_clip",
    "intersection_over_union",
    "match_frame",
    "size_band",
]

DEFAULT_IOU_THRESHOLD = 0.5
"""Standard detection IoU threshold. Recorded with every result."""

_SMALL_MAX = 0.02
_MEDIUM_MAX = 0.10


class SizeBand(StrEnum):
    """Coarse subject-size band, as a fraction of frame area.

    ``docs/obsidian/05 Technical/model.md`` names "distant / small subjects" as
    an expected failure condition. Banding by *relative* area is what makes that
    testable across clips of different resolutions.
    """

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


def size_band(box: BoundingBox, frame_width: int, frame_height: int) -> SizeBand:
    """Return the size band of *box* within a frame.

    Bands are ``small`` below 2% of frame area, ``medium`` below 10%, else
    ``large``.
    """
    fraction = box.relative_area(frame_width, frame_height)
    if fraction < _SMALL_MAX:
        return SizeBand.SMALL
    if fraction < _MEDIUM_MAX:
        return SizeBand.MEDIUM
    return SizeBand.LARGE


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Return the IoU of two boxes, in ``[0, 1]``.

    Args:
        first: A box.
        second: Another box.

    Returns:
        Intersection area divided by union area; ``0.0`` when they do not
        overlap.
    """
    x_min = max(first.x_min, second.x_min)
    y_min = max(first.y_min, second.y_min)
    x_max = min(first.x_max, second.x_max)
    y_max = min(first.y_max, second.y_max)

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class FrameMatch:
    """The outcome of matching one frame's predictions to its ground truth.

    Attributes:
        frame_index: Zero-based frame index.
        true_positives: ``(prediction, ground_truth, iou)`` for matched pairs.
        false_positives: Predictions that matched nothing.
        false_negatives: Ground-truth boxes nothing matched.
    """

    frame_index: int
    true_positives: tuple[tuple[Detection, GroundTruthDetection, float], ...] = ()
    false_positives: tuple[Detection, ...] = ()
    false_negatives: tuple[GroundTruthDetection, ...] = ()


def match_frame(
    predictions: Sequence[Detection],
    truth: Sequence[GroundTruthDetection],
    *,
    frame_index: int = 0,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> FrameMatch:
    """Match one frame's predictions against its ground truth.

    Args:
        predictions: Detections that survived confidence filtering.
        truth: Annotated apes in the same frame.
        frame_index: Zero-based index, carried into the result.
        iou_threshold: Minimum IoU for a pair to count as a match.

    Returns:
        The :class:`FrameMatch` for this frame.
    """
    unclaimed = list(range(len(truth)))
    matched: list[tuple[Detection, GroundTruthDetection, float]] = []
    unmatched_predictions: list[Detection] = []

    for prediction in sorted(predictions, key=lambda d: d.confidence, reverse=True):
        best_index: int | None = None
        best_iou = 0.0
        for index in unclaimed:
            overlap = intersection_over_union(prediction.box, truth[index].box)
            if overlap > best_iou:
                best_index, best_iou = index, overlap

        if best_index is not None and best_iou >= iou_threshold:
            unclaimed.remove(best_index)
            matched.append((prediction, truth[best_index], best_iou))
        else:
            unmatched_predictions.append(prediction)

    return FrameMatch(
        frame_index=frame_index,
        true_positives=tuple(matched),
        false_positives=tuple(unmatched_predictions),
        false_negatives=tuple(truth[i] for i in unclaimed),
    )


@dataclass(frozen=True, slots=True)
class MatchCounts:
    """Counts and the rates derived from them.

    Attributes:
        true_positives: Predictions matched to ground truth.
        false_positives: Predictions matching nothing.
        false_negatives: Ground-truth boxes nothing matched.
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Fraction of predictions that were right. ``0.0`` if none were made."""
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        """Fraction of annotated apes found. ``0.0`` if there were none."""
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        """Return a flat, JSON-friendly summary."""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }

    def __add__(self, other: MatchCounts) -> MatchCounts:
        """Sum two count sets, so per-clip results roll up to an overall one."""
        return MatchCounts(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )


@dataclass(frozen=True, slots=True)
class ClipEvaluation:
    """Everything measured for one clip.

    Attributes:
        clip_id: Manifest clip identifier.
        model_variant: The exact weights that produced the predictions. Recorded
            because a metrics directory is updated clip by clip: a run that stops
            part-way leaves a mixture of models, and pooling that mixture yields a
            number describing no model at all. Empty for records written before
            this field existed.
        iou_threshold: The IoU threshold used.
        confidence_threshold: The detector score threshold the predictions had
            already been filtered at.
        overall: Counts across every evaluated frame.
        by_behaviour: Counts split by the ground-truth behaviour label. Only
            recall is meaningful here -- a false positive has no behaviour.
        by_size: Counts split by ground-truth subject size band.
        frames_evaluated: How many frames were compared.
        empty_frames: Frames whose ground truth contained no ape.
        false_positives_on_empty_frames: Detections on frames with no ape at
            all -- the cleanest read on false-positive behaviour.
        mean_iou: Mean IoU over matched pairs.
    """

    clip_id: str
    iou_threshold: float
    confidence_threshold: float
    overall: MatchCounts
    model_variant: str = ""
    by_behaviour: Mapping[str, MatchCounts] = field(default_factory=dict)
    by_size: Mapping[str, MatchCounts] = field(default_factory=dict)
    frames_evaluated: int = 0
    empty_frames: int = 0
    false_positives_on_empty_frames: int = 0
    mean_iou: float = 0.0

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly summary for ``artifacts/metrics/<clip>.json``.

        The ``schema`` field is not decoration: track metrics share ``clip_id``,
        ``frames_evaluated`` and ``iou_threshold`` with this payload, which is
        enough for a consumer to mistake one for the other. It did, and crashed.
        """
        return {
            "schema": DETECTION_METRICS_SCHEMA,
            "clip_id": self.clip_id,
            "model_variant": self.model_variant,
            "iou_threshold": self.iou_threshold,
            "confidence_threshold": self.confidence_threshold,
            "frames_evaluated": self.frames_evaluated,
            "empty_frames": self.empty_frames,
            "false_positives_on_empty_frames": self.false_positives_on_empty_frames,
            "mean_iou": round(self.mean_iou, 4),
            "overall": self.overall.as_dict(),
            "by_behaviour": {k: v.as_dict() for k, v in sorted(self.by_behaviour.items())},
            "by_size": {k: v.as_dict() for k, v in sorted(self.by_size.items())},
        }


def evaluate_clip(
    clip_id: str,
    predictions: Mapping[int, Sequence[Detection]],
    truth: Mapping[int, GroundTruthFrame],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    confidence_threshold: float = 0.0,
    model_variant: str = "",
) -> ClipEvaluation:
    """Evaluate one clip's detections against its ground truth.

    Only frames present in *truth* are evaluated, so a strided run compares just
    the frames it actually processed rather than counting skipped frames as
    misses.

    Args:
        clip_id: Clip identifier.
        predictions: ``{frame_index: detections}``, zero-based.
        truth: ``{frame_index: GroundTruthFrame}``, zero-based.
        iou_threshold: Minimum IoU for a match.
        confidence_threshold: The score threshold already applied, recorded so
            the result is interpretable on its own.
        model_variant: The weights that produced *predictions*, recorded for the
            same reason.

    Returns:
        The clip's :class:`ClipEvaluation`.
    """
    overall = MatchCounts()
    by_behaviour: Counter[str] = Counter()
    behaviour_found: Counter[str] = Counter()
    by_size_found: Counter[str] = Counter()
    by_size_total: Counter[str] = Counter()
    ious: list[float] = []
    empty_frames = 0
    false_positives_on_empty = 0
    evaluated = 0

    # Only frames that were BOTH processed and annotated. A frame the detector
    # never saw -- because of `frame_stride`, or a capped run -- must not be
    # counted as a miss; that would silently report a recall that is really just
    # the fraction of frames processed.
    for frame_index in sorted(set(predictions) & set(truth)):
        ground_truth_frame = truth[frame_index]
        evaluated += 1
        frame_predictions = list(predictions.get(frame_index, ()))
        result = match_frame(
            frame_predictions,
            ground_truth_frame.detections,
            frame_index=frame_index,
            iou_threshold=iou_threshold,
        )

        overall = overall + MatchCounts(
            true_positives=len(result.true_positives),
            false_positives=len(result.false_positives),
            false_negatives=len(result.false_negatives),
        )
        ious.extend(iou for _, _, iou in result.true_positives)

        if ground_truth_frame.is_empty:
            empty_frames += 1
            false_positives_on_empty += len(result.false_positives)

        width = ground_truth_frame.frame_width
        height = ground_truth_frame.frame_height

        for _, ground_truth, _ in result.true_positives:
            behaviour_found[ground_truth.behaviour] += 1
            by_behaviour[ground_truth.behaviour] += 1
            band = size_band(ground_truth.box, width, height).value
            by_size_found[band] += 1
            by_size_total[band] += 1
        for missed in result.false_negatives:
            by_behaviour[missed.behaviour] += 1
            by_size_total[size_band(missed.box, width, height).value] += 1

    return ClipEvaluation(
        clip_id=clip_id,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
        model_variant=model_variant,
        overall=overall,
        # Only recall is defined per behaviour and per size: a false positive
        # has no ground-truth label to attribute it to.
        by_behaviour={
            behaviour: MatchCounts(
                true_positives=behaviour_found[behaviour],
                false_negatives=total - behaviour_found[behaviour],
            )
            for behaviour, total in by_behaviour.items()
        },
        by_size={
            band: MatchCounts(
                true_positives=by_size_found[band],
                false_negatives=total - by_size_found[band],
            )
            for band, total in by_size_total.items()
        },
        frames_evaluated=evaluated,
        empty_frames=empty_frames,
        false_positives_on_empty_frames=false_positives_on_empty,
        mean_iou=sum(ious) / len(ious) if ious else 0.0,
    )
