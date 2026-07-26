"""Tests for detection accuracy measurement.

Every metric here is checked against a **hand-computed** answer. A metric that
has only ever been compared against itself is not a measurement, and these
numbers are what a fine-tuning decision will rest on.
"""

from __future__ import annotations

import pytest

from panaf_ape_detection.data.annotations import GroundTruthDetection, GroundTruthFrame
from panaf_ape_detection.evaluation.detection import (
    MatchCounts,
    SizeBand,
    evaluate_clip,
    intersection_over_union,
    match_frame,
    size_band,
)
from panaf_ape_detection.types import BoundingBox, Detection

WIDTH, HEIGHT = 100, 100


def box(x_min: float, y_min: float, x_max: float, y_max: float) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def prediction(b: BoundingBox, confidence: float = 0.9) -> Detection:
    return Detection(box=b, confidence=confidence, category_id=0, category_name="animal")


def truth(b: BoundingBox, behaviour: str = "sitting", ape_id: int = 0) -> GroundTruthDetection:
    return GroundTruthDetection(box=b, ape_id=ape_id, species="chimpanzee", behaviour=behaviour)


def frame(*detections: GroundTruthDetection, index: int = 0) -> GroundTruthFrame:
    return GroundTruthFrame(
        clip_id="clip-a",
        frame_index=index,
        frame_width=WIDTH,
        frame_height=HEIGHT,
        detections=detections,
    )


# --------------------------------------------------------------------------- #
# IoU, against arithmetic done by hand
# --------------------------------------------------------------------------- #


def test_identical_boxes_have_iou_one():
    b = box(0, 0, 10, 10)

    assert intersection_over_union(b, b) == pytest.approx(1.0)


def test_disjoint_boxes_have_iou_zero():
    assert intersection_over_union(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0.0


def test_touching_boxes_have_iou_zero():
    """Edge contact is not overlap."""
    assert intersection_over_union(box(0, 0, 10, 10), box(10, 0, 20, 10)) == 0.0


def test_half_overlap():
    """10x10 and 10x10 offset by 5 in x: intersection 50, union 150, IoU 1/3."""
    result = intersection_over_union(box(0, 0, 10, 10), box(5, 0, 15, 10))

    assert result == pytest.approx(50 / 150)


def test_contained_box():
    """A 5x5 inside a 10x10: intersection 25, union 100, IoU 0.25."""
    result = intersection_over_union(box(0, 0, 10, 10), box(0, 0, 5, 5))

    assert result == pytest.approx(0.25)


def test_iou_is_symmetric():
    first, second = box(0, 0, 10, 10), box(3, 3, 13, 13)

    assert intersection_over_union(first, second) == pytest.approx(
        intersection_over_union(second, first)
    )


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def test_perfect_match():
    b = box(0, 0, 10, 10)

    result = match_frame([prediction(b)], [truth(b)])

    assert len(result.true_positives) == 1
    assert result.false_positives == ()
    assert result.false_negatives == ()
    assert result.true_positives[0][2] == pytest.approx(1.0)


def test_prediction_below_threshold_is_a_false_positive():
    """IoU 1/3 is under the 0.5 threshold, so nothing matches."""
    result = match_frame([prediction(box(5, 0, 15, 10))], [truth(box(0, 0, 10, 10))])

    assert result.true_positives == ()
    assert len(result.false_positives) == 1
    assert len(result.false_negatives) == 1


def test_detection_with_no_ground_truth_is_a_false_positive():
    result = match_frame([prediction(box(0, 0, 10, 10))], [])

    assert len(result.false_positives) == 1


def test_missed_ape_is_a_false_negative():
    result = match_frame([], [truth(box(0, 0, 10, 10))])

    assert len(result.false_negatives) == 1


def test_each_ground_truth_is_claimed_once():
    """Two predictions on one ape: one true positive, one false positive."""
    b = box(0, 0, 10, 10)

    result = match_frame([prediction(b, 0.9), prediction(b, 0.8)], [truth(b)])

    assert len(result.true_positives) == 1
    assert len(result.false_positives) == 1


def test_higher_confidence_claims_first():
    """The better-scoring prediction should win the box it overlaps."""
    exact = box(0, 0, 10, 10)
    offset = box(1, 1, 11, 11)

    result = match_frame([prediction(offset, 0.6), prediction(exact, 0.95)], [truth(exact)])

    matched_prediction = result.true_positives[0][0]
    assert matched_prediction.confidence == pytest.approx(0.95)


def test_two_apes_two_predictions():
    left, right = box(0, 0, 10, 10), box(50, 50, 60, 60)

    result = match_frame([prediction(left), prediction(right)], [truth(left), truth(right)])

    assert len(result.true_positives) == 2
    assert result.false_positives == ()


def test_threshold_is_configurable():
    """IoU 1/3 passes at 0.3 and fails at 0.5."""
    predictions = [prediction(box(5, 0, 15, 10))]
    ground_truth = [truth(box(0, 0, 10, 10))]

    assert len(match_frame(predictions, ground_truth, iou_threshold=0.3).true_positives) == 1
    assert len(match_frame(predictions, ground_truth, iou_threshold=0.5).true_positives) == 0


# --------------------------------------------------------------------------- #
# Counts and rates
# --------------------------------------------------------------------------- #


def test_rates_from_known_counts():
    """TP 3, FP 1, FN 1 -> P 0.75, R 0.75, F1 0.75."""
    counts = MatchCounts(true_positives=3, false_positives=1, false_negatives=1)

    assert counts.precision == pytest.approx(0.75)
    assert counts.recall == pytest.approx(0.75)
    assert counts.f1 == pytest.approx(0.75)


def test_asymmetric_rates():
    """TP 1, FP 3, FN 0 -> P 0.25, R 1.0, F1 0.4."""
    counts = MatchCounts(true_positives=1, false_positives=3)

    assert counts.precision == pytest.approx(0.25)
    assert counts.recall == pytest.approx(1.0)
    assert counts.f1 == pytest.approx(0.4)


def test_no_predictions_gives_zero_rather_than_dividing_by_zero():
    counts = MatchCounts(false_negatives=5)

    assert counts.precision == 0.0
    assert counts.recall == 0.0
    assert counts.f1 == 0.0


def test_counts_add():
    total = MatchCounts(1, 2, 3) + MatchCounts(10, 20, 30)

    assert (total.true_positives, total.false_positives, total.false_negatives) == (11, 22, 33)


# --------------------------------------------------------------------------- #
# Size bands
# --------------------------------------------------------------------------- #


def test_size_bands_by_relative_area():
    # 10x10 of 100x100 = 1% -> small; 20x20 = 4% -> medium; 50x50 = 25% -> large.
    assert size_band(box(0, 0, 10, 10), WIDTH, HEIGHT) is SizeBand.SMALL
    assert size_band(box(0, 0, 20, 20), WIDTH, HEIGHT) is SizeBand.MEDIUM
    assert size_band(box(0, 0, 50, 50), WIDTH, HEIGHT) is SizeBand.LARGE


def test_size_band_is_resolution_independent():
    """The same fraction of a bigger frame lands in the same band."""
    assert size_band(box(0, 0, 10, 10), 100, 100) is size_band(box(0, 0, 100, 100), 1000, 1000)


# --------------------------------------------------------------------------- #
# Whole-clip evaluation
# --------------------------------------------------------------------------- #


def test_clip_with_everything_found():
    b = box(0, 0, 50, 50)
    evaluation = evaluate_clip(
        "clip-a",
        {0: [prediction(b)], 1: [prediction(b)]},
        {0: frame(truth(b)), 1: frame(truth(b), index=1)},
    )

    assert evaluation.overall.precision == pytest.approx(1.0)
    assert evaluation.overall.recall == pytest.approx(1.0)
    assert evaluation.frames_evaluated == 2


def test_unprocessed_frames_are_not_counted_as_misses():
    """Regression guard.

    A capped or strided run once compared its handful of processed frames
    against every annotated frame, reporting a recall that was really just the
    fraction of frames processed -- on a 5-of-360 run that read as 0.013.
    """
    b = box(0, 0, 50, 50)
    ground_truth = {i: frame(truth(b), index=i) for i in range(100)}
    processed = {0: [prediction(b)], 1: [prediction(b)]}

    evaluation = evaluate_clip("clip-a", processed, ground_truth)

    assert evaluation.frames_evaluated == 2
    assert evaluation.overall.recall == pytest.approx(1.0)
    assert evaluation.overall.false_negatives == 0


def test_false_positives_on_empty_frames_are_counted():
    """The cleanest read on false-positive behaviour."""
    evaluation = evaluate_clip("clip-a", {0: [prediction(box(0, 0, 10, 10))]}, {0: frame()})

    assert evaluation.empty_frames == 1
    assert evaluation.false_positives_on_empty_frames == 1
    assert evaluation.overall.precision == 0.0


def test_recall_is_broken_down_by_behaviour():
    found, missed = box(0, 0, 50, 50), box(60, 60, 70, 70)
    evaluation = evaluate_clip(
        "clip-a",
        {0: [prediction(found)]},
        {0: frame(truth(found, behaviour="walking"), truth(missed, behaviour="hanging"))},
    )

    assert evaluation.by_behaviour["walking"].recall == pytest.approx(1.0)
    assert evaluation.by_behaviour["hanging"].recall == pytest.approx(0.0)


def test_recall_is_broken_down_by_size():
    large, small = box(0, 0, 50, 50), box(80, 80, 85, 85)
    evaluation = evaluate_clip(
        "clip-a", {0: [prediction(large)]}, {0: frame(truth(large), truth(small))}
    )

    assert evaluation.by_size["large"].recall == pytest.approx(1.0)
    assert evaluation.by_size["small"].recall == pytest.approx(0.0)


def test_mean_iou_is_over_matched_pairs_only():
    exact = box(0, 0, 10, 10)
    evaluation = evaluate_clip("clip-a", {0: [prediction(exact)]}, {0: frame(truth(exact))})

    assert evaluation.mean_iou == pytest.approx(1.0)


def test_thresholds_are_recorded_with_the_result():
    """A number without its thresholds cannot be compared with anything."""
    evaluation = evaluate_clip("clip-a", {}, {0: frame()}, confidence_threshold=0.35)

    assert evaluation.confidence_threshold == pytest.approx(0.35)
    assert evaluation.iou_threshold == pytest.approx(0.5)
    assert evaluation.as_dict()["confidence_threshold"] == pytest.approx(0.35)


# --------------------------------------------------------------------------- #
# Pooling across clips
# --------------------------------------------------------------------------- #


def test_pooled_mean_iou_weights_by_matched_pairs():
    """A clip with no matches must not be averaged in as though it scored 0.0.

    `ClipEvaluation.mean_iou` is 0.0 by construction when nothing matched, so an
    unweighted mean over clips reports bad *localisation* where there was really
    no localisation at all -- and lets a two-match clip outvote a 600-match one.
    """
    from panaf_ape_detection.pipeline.runner import ClipResult, summarise

    exact = box(0, 0, 10, 10)
    matched = evaluate_clip("clip-a", {0: [prediction(exact)]}, {0: frame(truth(exact))})
    unmatched = evaluate_clip(
        "clip-b", {0: [prediction(box(80, 80, 90, 90))]}, {0: frame(truth(exact))}
    )
    assert matched.mean_iou == pytest.approx(1.0)
    assert unmatched.mean_iou == pytest.approx(0.0)
    assert unmatched.overall.true_positives == 0

    summary = summarise(
        [
            ClipResult(clip_id="clip-a", frames_processed=1, detections_kept=1, evaluation=matched),
            ClipResult(
                clip_id="clip-b", frames_processed=1, detections_kept=1, evaluation=unmatched
            ),
        ]
    )

    # Unweighted this would be 0.5; the one clip that matched anything scored 1.0.
    assert summary["mean_iou"] == pytest.approx(1.0)
    assert summary["matched_pairs"] == 1
