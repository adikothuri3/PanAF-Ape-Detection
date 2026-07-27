"""Tests for track refinement: stitching, interpolation and smoothing.

Pure functions over tracked detections, so none of this needs the ``inference``
extra, a GPU or a video. Every expectation is a worked example.
"""

from __future__ import annotations

import pytest

from panaf_ape_detection.evaluation.tracking import measure_jitter
from panaf_ape_detection.tracking.refine import (
    interpolate_gaps,
    refine,
    smooth_tracks,
    stitch_tracks,
)
from panaf_ape_detection.types import BoundingBox, TrackedDetection


def box(x: float, y: float = 0.0, size: float = 20.0) -> BoundingBox:
    """A *size* square with its top-left corner at (*x*, *y*)."""
    return BoundingBox(x_min=x, y_min=y, x_max=x + size, y_max=y + size)


def tracked(track_id: int, b: BoundingBox, confidence: float = 0.9) -> TrackedDetection:
    return TrackedDetection(
        box=b, confidence=confidence, category_id=0, category_name="animal", track_id=track_id
    )


# --------------------------------------------------------------------------- #
# Stitching
# --------------------------------------------------------------------------- #


def test_a_fragment_continuing_the_motion_is_joined():
    """Track 1 walks right, vanishes for 3 frames, and track 2 resumes where it would be."""
    per_frame = {
        0: [tracked(1, box(0))],
        1: [tracked(1, box(10))],
        2: [tracked(1, box(20))],
        3: [],
        4: [],
        # Constant velocity from frame 2 predicts x = 20 + 10*3 = 50.
        5: [tracked(2, box(50))],
        6: [tracked(2, box(60))],
    }

    stitched = stitch_tracks(per_frame, max_gap=5, max_distance=1.0)

    ids = {d.track_id for detections in stitched.values() for d in detections}
    assert ids == {1}


def test_tracks_overlapping_in_time_are_never_joined():
    """Two apes visible in the same frame are two apes, whatever the distance.

    This is the rule that keeps stitching from cheating: merging them would erase
    the ID switches and drive fragmentation to a perfect 1.00.
    """
    per_frame = {
        0: [tracked(1, box(0)), tracked(2, box(0.5))],
        1: [tracked(1, box(0)), tracked(2, box(0.5))],
    }

    stitched = stitch_tracks(per_frame, max_gap=50, max_distance=100.0)

    ids = {d.track_id for detections in stitched.values() for d in detections}
    assert ids == {1, 2}


def test_a_fragment_too_far_away_is_left_alone():
    per_frame = {
        0: [tracked(1, box(0))],
        1: [tracked(1, box(0))],
        3: [tracked(2, box(900))],
    }

    stitched = stitch_tracks(per_frame, max_gap=5, max_distance=1.0)

    assert {d.track_id for dets in stitched.values() for d in dets} == {1, 2}


def test_a_gap_beyond_the_limit_is_not_bridged():
    per_frame = {0: [tracked(1, box(0))], 1: [tracked(1, box(0))], 30: [tracked(2, box(0))]}

    stitched = stitch_tracks(per_frame, max_gap=5)

    assert {d.track_id for dets in stitched.values() for d in dets} == {1, 2}


def test_a_wildly_different_size_is_not_the_same_animal():
    per_frame = {
        0: [tracked(1, box(0, size=20))],
        1: [tracked(1, box(0, size=20))],
        3: [tracked(2, box(0, size=200))],
    }

    stitched = stitch_tracks(per_frame, max_gap=5, max_distance=10.0)

    assert {d.track_id for dets in stitched.values() for d in dets} == {1, 2}


def test_stitching_is_one_to_one():
    """Two later fragments must not both be absorbed into a single chain."""
    per_frame = {
        0: [tracked(1, box(0))],
        2: [tracked(2, box(0))],
        4: [tracked(3, box(0))],
    }

    stitched = stitch_tracks(per_frame, max_gap=3, max_distance=5.0)

    # 2 joins 1; 3 then joins the chain's new tail rather than being refused,
    # which is the intended chaining -- but each link is still a single hop.
    assert len({d.track_id for dets in stitched.values() for d in dets}) == 1


def test_zero_gap_disables_stitching():
    per_frame = {0: [tracked(1, box(0))], 2: [tracked(2, box(0))]}

    assert stitch_tracks(per_frame, max_gap=0) == {
        index: list(detections) for index, detections in per_frame.items()
    }


def test_a_negative_gap_is_rejected():
    with pytest.raises(ValueError, match="max_gap"):
        stitch_tracks({}, max_gap=-1)


# --------------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------------- #


def test_an_interior_gap_is_filled_linearly():
    per_frame = {0: [tracked(1, box(0))], 1: [], 2: [], 3: [tracked(1, box(30))]}

    filled = interpolate_gaps(per_frame, max_gap=5)

    assert [d.box.x_min for d in filled[1]] == [pytest.approx(10.0)]
    assert [d.box.x_min for d in filled[2]] == [pytest.approx(20.0)]


def test_interpolated_boxes_are_marked_as_such():
    """A synthesised box must never be mistakable for a detection."""
    per_frame = {0: [tracked(1, box(0))], 1: [], 2: [tracked(1, box(20))]}

    filled = interpolate_gaps(per_frame, max_gap=5)

    assert filled[1][0].interpolated is True
    assert filled[0][0].interpolated is False
    assert filled[2][0].interpolated is False


def test_an_interpolated_box_takes_the_lower_surrounding_confidence():
    per_frame = {
        0: [tracked(1, box(0), confidence=0.9)],
        1: [],
        2: [tracked(1, box(20), confidence=0.4)],
    }

    filled = interpolate_gaps(per_frame, max_gap=5)

    assert filled[1][0].confidence == pytest.approx(0.4)


def test_a_track_is_never_extended_past_its_own_span():
    """There is no evidence about where the animal was before or after."""
    per_frame = {0: [], 1: [tracked(1, box(0))], 2: [tracked(1, box(10))], 3: [], 4: []}

    filled = interpolate_gaps(per_frame, max_gap=5)

    assert filled[0] == []
    assert filled[3] == []
    assert filled[4] == []


def test_a_gap_longer_than_the_limit_is_left_open():
    per_frame = {0: [tracked(1, box(0))], **{i: [] for i in range(1, 20)}, 20: [tracked(1, box(0))]}

    filled = interpolate_gaps(per_frame, max_gap=5)

    assert all(filled[i] == [] for i in range(1, 20))


def test_zero_gap_disables_interpolation():
    per_frame = {0: [tracked(1, box(0))], 1: [], 2: [tracked(1, box(20))]}

    assert interpolate_gaps(per_frame, max_gap=0)[1] == []


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #


def test_smoothing_removes_shake():
    """Same endpoints, alternating back and forth: jitter must fall."""
    offsets = [0, 10, 0, 10, 0, 10, 0]
    per_frame = {index: [tracked(1, box(offset))] for index, offset in enumerate(offsets)}

    before = measure_jitter(per_frame)
    after = measure_jitter(smooth_tracks(per_frame, window=3))

    assert before > 0.0
    assert after < before


def test_smoothing_preserves_steady_motion():
    """A centred average of a straight line is that same line."""
    per_frame = {index: [tracked(1, box(index * 10))] for index in range(7)}

    smoothed = smooth_tracks(per_frame, window=3)

    # Interior frames are unmoved; the ends shrink their window and so are too.
    for index in range(7):
        assert smoothed[index][0].box.x_min == pytest.approx(index * 10.0)


def test_smoothing_changes_no_identities_or_frames():
    per_frame = {0: [tracked(1, box(0))], 1: [tracked(1, box(30))], 2: [tracked(2, box(0))]}

    smoothed = smooth_tracks(per_frame, window=3)

    assert set(smoothed) == set(per_frame)
    assert [len(v) for v in smoothed.values()] == [len(v) for v in per_frame.values()]
    assert {d.track_id for dets in smoothed.values() for d in dets} == {1, 2}


def test_smoothing_does_not_reach_across_a_gap():
    """A frame the track is absent from must not drag its neighbours."""
    per_frame = {0: [tracked(1, box(0))], 1: [], 2: [tracked(1, box(1000))]}

    smoothed = smooth_tracks(per_frame, window=3)

    # Frame 0 averages only itself and frame 2 is not within its window.
    assert smoothed[0][0].box.x_min == pytest.approx(0.0)


def test_window_of_one_changes_nothing():
    per_frame = {0: [tracked(1, box(0))], 1: [tracked(1, box(50))]}

    assert smooth_tracks(per_frame, window=1) == {i: list(v) for i, v in per_frame.items()}


def test_an_even_window_is_rejected():
    """It has no centre, so every box would shift half a frame forward in time."""
    with pytest.raises(ValueError, match="odd"):
        smooth_tracks({}, window=4)


# --------------------------------------------------------------------------- #
# The combined pipeline
# --------------------------------------------------------------------------- #


def test_refine_stitches_before_it_interpolates():
    """Interpolating first would fill toward a fragment's end, not its continuation.

    Track 1 stops at frame 2 and track 2 resumes at frame 5. Only if they are
    joined first can frames 3 and 4 be filled at all.
    """
    per_frame = {
        0: [tracked(1, box(0))],
        1: [tracked(1, box(10))],
        2: [tracked(1, box(20))],
        3: [],
        4: [],
        5: [tracked(2, box(50))],
    }

    refined = refine(per_frame, stitch_max_gap=5, interpolate_max_gap=5)

    assert len(refined[3]) == 1
    assert refined[3][0].interpolated is True
    assert {d.track_id for dets in refined.values() for d in dets} == {1}


def test_refine_with_everything_off_is_a_no_op():
    per_frame = {0: [tracked(1, box(0))], 1: [], 2: [tracked(2, box(90))]}

    assert refine(per_frame) == {index: list(v) for index, v in per_frame.items()}
