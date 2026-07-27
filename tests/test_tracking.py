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
from panaf_ape_detection.evaluation.tracking import evaluate_tracking, measure_jitter
from panaf_ape_detection.tracking.bytetrack import (
    UPSTREAM_SCORE_FLOOR,
    ScoreFloor,
    drop_short_tracks,
)
from panaf_ape_detection.types import BoundingBox, Detection, TrackedDetection

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
# Identity coverage -- the metric fragmentation and switches cannot express
# --------------------------------------------------------------------------- #


def test_one_track_throughout_gives_identity_coverage_equal_to_coverage():
    result = evaluate_tracking(
        "clip-a",
        {index: [tracked(1)] for index in range(4)},
        {index: frame(index, truth(0)) for index in range(4)},
    )

    individual = result.individuals[0]
    assert individual.coverage == 1.0
    assert individual.identity_coverage == 1.0


def test_splitting_an_ape_across_two_tracks_halves_identity_coverage():
    """Coverage stays at 1.0 while the ape is followed by two different ids.

    Four frames, ids 1, 1, 2, 2: every frame is covered, so `coverage` cannot
    see the problem. The dominant track holds only two of the four.
    """
    per_frame = {0: [tracked(1)], 1: [tracked(1)], 2: [tracked(2)], 3: [tracked(2)]}

    result = evaluate_tracking(
        "clip-a", per_frame, {index: frame(index, truth(0)) for index in range(4)}
    )

    individual = result.individuals[0]
    assert individual.coverage == 1.0
    assert individual.identity_coverage == 0.5
    assert individual.fragmentation == 2


def test_identity_coverage_is_zero_for_an_ape_never_tracked():
    result = evaluate_tracking(
        "clip-a", {0: [], 1: []}, {0: frame(0, truth(0)), 1: frame(1, truth(0))}
    )

    assert result.individuals[0].identity_coverage == 0.0


# --------------------------------------------------------------------------- #
# Track purity and merges -- the guard against cheating by merging apes
# --------------------------------------------------------------------------- #

LEFT = box(0, 0, 20, 20)
RIGHT = box(60, 60, 80, 80)


def test_one_track_per_ape_is_perfectly_pure():
    per_frame = {index: [tracked(1, LEFT), tracked(2, RIGHT)] for index in range(3)}
    ground_truth = {index: frame(index, truth(0, LEFT), truth(1, RIGHT)) for index in range(3)}

    result = evaluate_tracking("clip-a", per_frame, ground_truth)

    assert result.mean_track_purity == 1.0
    assert result.id_merges == 0


def test_one_track_shared_by_two_apes_is_caught_as_a_merge():
    """The failure every other metric on this class rates as perfect.

    Track 1 covers ape 0 for two frames, then ape 1 for two frames. Neither ape
    ever sees its id change *while it is covered*, so there are no switches;
    each is spread over exactly one track, so fragmentation is the ideal 1.0.
    Only purity registers that the track changed animal.
    """
    per_frame = {
        0: [tracked(1, LEFT)],
        1: [tracked(1, LEFT)],
        2: [tracked(1, RIGHT)],
        3: [tracked(1, RIGHT)],
    }
    ground_truth = {
        0: frame(0, truth(0, LEFT)),
        1: frame(1, truth(0, LEFT)),
        2: frame(2, truth(1, RIGHT)),
        3: frame(3, truth(1, RIGHT)),
    }

    result = evaluate_tracking("clip-a", per_frame, ground_truth)

    # The metrics that look clean.
    assert result.total_id_switches == 0
    assert result.mean_fragmentation == 1.0
    # The one that does not: 2 of the track's 4 matched frames were the wrong ape.
    assert result.mean_track_purity == 0.5
    assert result.id_merges == 1


def test_tracks_matching_nothing_do_not_count_against_purity():
    """A track over empty ground truth is a false positive, not an impure track."""
    result = evaluate_tracking(
        "clip-a",
        {0: [tracked(1, LEFT), tracked(9, RIGHT)]},
        {0: frame(0, truth(0, LEFT))},
    )

    assert result.mean_track_purity == 1.0
    assert result.predicted_tracks == 2


def test_purity_is_one_when_nothing_matched_at_all():
    result = evaluate_tracking("clip-a", {0: []}, {0: frame(0, truth(0))})

    assert result.mean_track_purity == 1.0
    assert result.id_merges == 0


# --------------------------------------------------------------------------- #
# Jitter -- makes "smooth" a measurement
# --------------------------------------------------------------------------- #


def test_a_stationary_box_has_no_jitter():
    assert measure_jitter({index: [tracked(1, box(0, 0, 20, 20))] for index in range(5)}) == 0.0


def test_constant_velocity_has_no_jitter():
    """Panning steadily across the frame is smooth, however fast it moves."""
    per_frame = {index: [tracked(1, box(index * 10, 0, index * 10 + 20, 20))] for index in range(5)}

    assert measure_jitter(per_frame) == pytest.approx(0.0)


def test_a_shaking_box_has_jitter():
    """Same start and end, but alternating one step forward and back."""
    offsets = [0, 10, 0, 10, 0]
    per_frame = {
        index: [tracked(1, box(offset, 0, offset + 20, 20))] for index, offset in enumerate(offsets)
    }

    assert measure_jitter(per_frame) > 0.0


def test_jitter_is_resolution_independent():
    """Doubling every coordinate must not change how shaky the track looks."""
    small = {
        index: [tracked(1, box(offset, 0, offset + 20, 20))]
        for index, offset in enumerate([0, 10, 0, 10, 0])
    }
    large = {
        index: [tracked(1, box(offset, 0, offset + 40, 40))]
        for index, offset in enumerate([0, 20, 0, 20, 0])
    }

    assert measure_jitter(small) == pytest.approx(measure_jitter(large))


def test_jitter_ignores_gaps_in_a_track():
    """A box that vanishes and returns did not travel that distance in one frame."""
    per_frame = {
        0: [tracked(1, box(0, 0, 20, 20))],
        1: [tracked(1, box(0, 0, 20, 20))],
        2: [tracked(1, box(0, 0, 20, 20))],
        # frame 3 missing
        4: [tracked(1, box(500, 0, 520, 20))],
    }

    assert measure_jitter(per_frame) == 0.0


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


# --------------------------------------------------------------------------- #
# ScoreFloor -- working around supervision's hardcoded discard floor
# --------------------------------------------------------------------------- #


def test_score_floor_fixes_one_and_lifts_zero():
    floor = ScoreFloor(0.12)

    assert floor.forward(1.0) == pytest.approx(1.0)
    assert floor.forward(0.0) == pytest.approx(0.12)


def test_score_floor_lifts_everything_clear_of_the_upstream_discard():
    """The point of the class: nothing may land at or below the 0.1 literal.

    18.6% of this project's detections at confidence 0.05 score at or below
    0.10 and are dropped by supervision before either association pass.
    """
    floor = ScoreFloor(0.12)

    for score in (0.0, 0.001, 0.05, 0.0999, 0.1):
        assert floor.forward(score) > UPSTREAM_SCORE_FLOOR


def test_score_floor_is_strictly_monotone():
    """Ranking must survive: the map decides what is seen, not what wins."""
    floor = ScoreFloor(0.2)
    scores = [0.0, 0.05, 0.1, 0.3, 0.55, 0.9, 1.0]

    lifted = [floor.forward(s) for s in scores]

    assert lifted == sorted(lifted)
    assert len(set(lifted)) == len(lifted)


def test_score_floor_round_trips_exactly():
    """Verified by observing the value, not by the absence of an exception."""
    floor = ScoreFloor(0.15)

    for score in (0.0, 0.05, 0.2, 0.5, 0.87, 1.0):
        assert floor.inverse(floor.forward(score)) == pytest.approx(score)


def test_score_floor_inverse_is_clamped_to_a_valid_confidence():
    """`Confidence` is constrained to [0, 1]; a stray value must not raise later."""
    floor = ScoreFloor(0.3)

    assert floor.inverse(0.0) == 0.0
    assert floor.inverse(2.0) == 1.0


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_an_unusable_score_floor_is_rejected(bad: float):
    with pytest.raises(ValueError, match="score floor"):
        ScoreFloor(bad)


def test_tracked_detections_carry_the_original_score_not_the_lifted_one():
    """Reporting a rescaled score would be inventing a measurement."""
    pytest.importorskip("supervision", reason="requires the inference extra")
    from panaf_ape_detection.tracking.bytetrack import ByteTrackTracker

    tracker = ByteTrackTracker(
        activation_threshold=0.2, score_floor=0.15, frame_width=WIDTH, frame_height=HEIGHT
    )
    detection = Detection(box=box(), confidence=0.42, category_id=0, category_name="animal")

    seen: list[float] = []
    for _ in range(10):
        seen.extend(d.confidence for d in tracker.update([detection]))

    assert seen, "the tracker never reported the detection"
    assert all(value == pytest.approx(0.42, abs=1e-6) for value in seen)
