"""Tests for tracking and track-quality measurement.

The ID-switch and fragmentation arithmetic is checked against worked examples, in
the same spirit as the detection metrics: a number nobody has verified by hand is
not a measurement.

The `supervision` converters are exercised separately, and skip when the
``inference`` extra is absent.
"""

from __future__ import annotations

import pytest

from panaf_ape_detection.data.annotations import GroundTruthDetection, GroundTruthFrame
from panaf_ape_detection.evaluation.tracking import evaluate_tracking
from panaf_ape_detection.tracking.bytetrack import drop_short_tracks
from panaf_ape_detection.types import BoundingBox, TrackedDetection

WIDTH, HEIGHT = 100, 100


def box(x_min: float = 0, y_min: float = 0, x_max: float = 20, y_max: float = 20) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def tracked(track_id: int, b: BoundingBox | None = None) -> TrackedDetection:
    return TrackedDetection(
        box=b or box(),
        confidence=0.9,
        category_id=0,
        category_name="animal",
        track_id=track_id,
    )


def truth(ape_id: int, b: BoundingBox | None = None) -> GroundTruthDetection:
    return GroundTruthDetection(
        box=b or box(), ape_id=ape_id, species="chimpanzee", behaviour="walking"
    )


def frame(index: int, *detections: GroundTruthDetection) -> GroundTruthFrame:
    return GroundTruthFrame(
        clip_id="clip-a",
        frame_index=index,
        frame_width=WIDTH,
        frame_height=HEIGHT,
        detections=detections,
    )


# --------------------------------------------------------------------------- #
# ID switches -- worked examples
# --------------------------------------------------------------------------- #


def test_one_individual_one_track_has_no_switches():
    tracked_frames = {i: [tracked(1)] for i in range(5)}
    ground_truth = {i: frame(i, truth(0)) for i in range(5)}

    result = evaluate_tracking("clip-a", tracked_frames, ground_truth)

    assert result.total_id_switches == 0
    assert result.individuals[0].fragmentation == 1
    assert result.individuals[0].coverage == pytest.approx(1.0)


def test_a_single_id_change_counts_exactly_one_switch():
    """Track 1 for frames 0-2, then track 2 for 3-4: one switch, two tracks."""
    tracked_frames = {
        0: [tracked(1)],
        1: [tracked(1)],
        2: [tracked(1)],
        3: [tracked(2)],
        4: [tracked(2)],
    }
    ground_truth = {i: frame(i, truth(0)) for i in range(5)}

    result = evaluate_tracking("clip-a", tracked_frames, ground_truth)

    assert result.total_id_switches == 1
    assert result.individuals[0].fragmentation == 2


def test_two_id_changes_count_two_switches():
    """1, 1, 2, 2, 3 -> two changes."""
    ids = [1, 1, 2, 2, 3]
    tracked_frames = {i: [tracked(t)] for i, t in enumerate(ids)}
    ground_truth = {i: frame(i, truth(0)) for i in range(len(ids))}

    result = evaluate_tracking("clip-a", tracked_frames, ground_truth)

    assert result.total_id_switches == 2
    assert result.individuals[0].fragmentation == 3


def test_resuming_the_same_id_after_a_gap_is_not_a_switch():
    """The lost-track buffer working correctly must not be penalised.

    Frames 0-1 tracked as 1, frame 2 the detector missed it, frames 3-4 tracked
    as 1 again. That is one continuous track, not a switch.
    """
    tracked_frames = {0: [tracked(1)], 1: [tracked(1)], 2: [], 3: [tracked(1)], 4: [tracked(1)]}
    ground_truth = {i: frame(i, truth(0)) for i in range(5)}

    result = evaluate_tracking("clip-a", tracked_frames, ground_truth)

    assert result.total_id_switches == 0
    assert result.individuals[0].fragmentation == 1
    assert result.individuals[0].covered_frames == 4
    assert result.individuals[0].annotated_frames == 5


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def test_an_individual_never_tracked_has_zero_coverage():
    ground_truth = {i: frame(i, truth(0)) for i in range(4)}

    result = evaluate_tracking("clip-a", {i: [] for i in range(4)}, ground_truth)

    individual = result.individuals[0]
    assert individual.coverage == 0.0
    assert individual.fragmentation == 0
    assert individual.is_mostly_lost
    assert not individual.is_mostly_tracked


def test_mostly_tracked_and_mostly_lost_thresholds():
    """4 of 5 frames = 0.8 -> mostly tracked; 1 of 5 = 0.2 -> mostly lost."""
    good = {i: ([tracked(1)] if i < 4 else []) for i in range(5)}
    poor = {i: ([tracked(1)] if i < 1 else []) for i in range(5)}
    ground_truth = {i: frame(i, truth(0)) for i in range(5)}

    assert evaluate_tracking("c", good, ground_truth).mostly_tracked == 1
    assert evaluate_tracking("c", poor, ground_truth).mostly_lost == 1


def test_unprocessed_frames_are_not_counted_against_coverage():
    """Same guard as detection: only frames in both inputs are evaluated."""
    ground_truth = {i: frame(i, truth(0)) for i in range(100)}
    processed = {0: [tracked(1)], 1: [tracked(1)]}

    result = evaluate_tracking("clip-a", processed, ground_truth)

    assert result.frames_evaluated == 2
    assert result.individuals[0].coverage == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Multiple individuals
# --------------------------------------------------------------------------- #


def test_two_individuals_tracked_separately():
    left, right = box(0, 0, 20, 20), box(60, 60, 80, 80)
    tracked_frames = {i: [tracked(1, left), tracked(2, right)] for i in range(3)}
    ground_truth = {i: frame(i, truth(0, left), truth(1, right)) for i in range(3)}

    result = evaluate_tracking("clip-a", tracked_frames, ground_truth)

    assert len(result.individuals) == 2
    assert result.total_id_switches == 0
    assert result.mean_fragmentation == pytest.approx(1.0)
    assert result.predicted_tracks == 2


def test_swapped_identities_are_counted_as_switches():
    """Two apes whose track ids trade places: one switch each."""
    left, right = box(0, 0, 20, 20), box(60, 60, 80, 80)
    tracked_frames = {
        0: [tracked(1, left), tracked(2, right)],
        1: [tracked(1, left), tracked(2, right)],
        2: [tracked(2, left), tracked(1, right)],
    }
    ground_truth = {i: frame(i, truth(0, left), truth(1, right)) for i in range(3)}

    result = evaluate_tracking("clip-a", tracked_frames, ground_truth)

    assert result.total_id_switches == 2


def test_a_prediction_matching_nothing_does_not_create_an_individual():
    far_away = box(80, 80, 95, 95)
    ground_truth = {0: frame(0, truth(0, box(0, 0, 20, 20)))}

    result = evaluate_tracking("clip-a", {0: [tracked(9, far_away)]}, ground_truth)

    assert len(result.individuals) == 1
    assert result.individuals[0].coverage == 0.0
    # The stray track still counts toward what the tracker produced.
    assert result.predicted_tracks == 1


def test_empty_clip_reports_zeroes_rather_than_dividing_by_zero():
    result = evaluate_tracking("clip-a", {}, {})

    assert result.total_id_switches == 0
    assert result.mean_fragmentation == 0.0
    assert result.mean_coverage == 0.0
    assert result.as_dict()["annotated_individuals"] == 0


# --------------------------------------------------------------------------- #
# drop_short_tracks
# --------------------------------------------------------------------------- #


def test_short_tracks_are_dropped():
    per_frame = {0: [tracked(1), tracked(2)], 1: [tracked(1)], 2: [tracked(1)]}

    kept = drop_short_tracks(per_frame, minimum_length=2)

    assert [d.track_id for d in kept[0]] == [1]
    assert all(d.track_id == 1 for frame_dets in kept.values() for d in frame_dets)


def test_frames_emptied_by_dropping_are_kept_as_empty():
    """Losing the frame would lose the fact that it was processed."""
    per_frame = {0: [tracked(7)], 1: [], 2: []}

    kept = drop_short_tracks(per_frame, minimum_length=2)

    assert set(kept) == {0, 1, 2}
    assert kept[0] == []


def test_minimum_length_one_keeps_everything():
    per_frame = {0: [tracked(1)], 1: [tracked(2)]}

    assert drop_short_tracks(per_frame, minimum_length=1) == per_frame


def test_minimum_length_below_one_is_rejected():
    with pytest.raises(ValueError, match="minimum_length"):
        drop_short_tracks({}, minimum_length=0)


# --------------------------------------------------------------------------- #
# supervision converters -- need the inference extra
# --------------------------------------------------------------------------- #


def test_converter_round_trip():
    pytest.importorskip("supervision", reason="requires the inference extra")
    import numpy as np

    from panaf_ape_detection.tracking.convert import from_supervision, to_supervision
    from panaf_ape_detection.types import Detection

    original = [
        Detection(box=box(1, 2, 31, 42), confidence=0.75, category_id=0, category_name="animal"),
        Detection(box=box(50, 50, 70, 90), confidence=0.5, category_id=0, category_name="animal"),
    ]

    packed = to_supervision(original)
    assert packed.xyxy.shape == (2, 4)
    assert packed.confidence is not None

    # `from_supervision` only returns rows carrying a tracker_id, so assign some.
    packed.tracker_id = np.array([11, 12], dtype=int)
    restored = from_supervision(packed)

    assert len(restored) == 2
    for before, after in zip(original, restored, strict=True):
        assert after.box.x_min == pytest.approx(before.box.x_min)
        assert after.box.y_max == pytest.approx(before.box.y_max)
        assert after.confidence == pytest.approx(before.confidence, abs=1e-6)
        assert after.category_name == "animal"
    assert [d.track_id for d in restored] == [11, 12]


def test_empty_detections_convert_cleanly():
    pytest.importorskip("supervision", reason="requires the inference extra")

    from panaf_ape_detection.tracking.convert import from_supervision, to_supervision

    packed = to_supervision([])
    assert len(packed) == 0
    assert from_supervision(packed) == []


def test_untracked_rows_are_not_promoted_to_tracked_detections():
    """A row with no tracker_id has no identity, so inventing one is wrong."""
    pytest.importorskip("supervision", reason="requires the inference extra")

    from panaf_ape_detection.tracking.convert import from_supervision, to_supervision
    from panaf_ape_detection.types import Detection

    packed = to_supervision(
        [Detection(box=box(), confidence=0.9, category_id=0, category_name="animal")]
    )
    assert packed.tracker_id is None

    assert from_supervision(packed) == []
