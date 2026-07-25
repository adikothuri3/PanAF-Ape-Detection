"""Tests for the shared detection, track and run-metadata schemas.

These structures are a design contract for the pipeline stages implemented in
later phases. Nothing here runs inference.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from panaf_ape_detection.types import (
    BoundingBox,
    Detection,
    DeviceKind,
    FrameDetections,
    InputFileRecord,
    RunMetadata,
    TrackedDetection,
    TrackedFrameDetections,
    TrackingBackend,
)

SHA = "0" * 64


def box(**overrides: float) -> BoundingBox:
    values: dict[str, float] = {"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 220}
    values.update(overrides)
    return BoundingBox(**values)  # type: ignore[arg-type]


def test_bounding_box_geometry():
    b = box()

    assert b.width == pytest.approx(100.0)
    assert b.height == pytest.approx(200.0)
    assert b.area == pytest.approx(20000.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"x_max": 5.0}, "x_max"),
        ({"y_max": 1.0}, "y_max"),
    ],
)
def test_bounding_box_rejects_inverted_edges(overrides: dict[str, float], message: str):
    with pytest.raises(ValidationError, match=message):
        box(**overrides)


def test_bounding_box_rejects_negative_origin():
    with pytest.raises(ValidationError):
        box(x_min=-1.0)


def test_detection_confidence_is_bounded():
    with pytest.raises(ValidationError, match="confidence"):
        Detection(box=box(), confidence=1.4, category_id=0, category_name="animal")


def test_detection_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="species"):
        Detection(
            box=box(),
            confidence=0.5,
            category_id=0,
            category_name="animal",
            species="chimpanzee",  # type: ignore[call-arg]
        )


def test_tracked_detection_carries_identity_and_dataset_label():
    tracked = TrackedDetection(
        box=box(),
        confidence=0.7,
        category_id=0,
        category_name="animal",
        track_id=3,
        behavior_label="walking",
    )

    assert tracked.track_id == 3
    # Behaviour labels come from the dataset; the detector never predicts them.
    assert tracked.behavior_label == "walking"


def test_tracked_detection_behavior_label_defaults_to_none():
    tracked = TrackedDetection(
        box=box(), confidence=0.7, category_id=0, category_name="animal", track_id=0
    )

    assert tracked.behavior_label is None


def frame(**overrides: object) -> FrameDetections:
    values: dict[str, object] = {
        "clip_id": "clip-a",
        "frame_index": 0,
        "frame_width": 1280,
        "frame_height": 720,
    }
    values.update(overrides)
    return FrameDetections(**values)  # type: ignore[arg-type]


def test_frame_detections_defaults_to_empty():
    detections = frame()

    assert detections.detections == []
    assert detections.timestamp_seconds is None


def test_frame_detections_rejects_negative_index():
    with pytest.raises(ValidationError, match="frame_index"):
        frame(frame_index=-1)


# --------------------------------------------------------------------------- #
# Frame dimensions -- a serialized record must be interpretable on its own,
# without re-opening the source video.
# --------------------------------------------------------------------------- #


def test_frame_dimensions_are_required():
    """Without them, boxes cannot be normalised or bounds-checked after the fact."""
    with pytest.raises(ValidationError, match="frame_width"):
        FrameDetections(clip_id="clip-a", frame_index=0)  # type: ignore[call-arg]


@pytest.mark.parametrize(("width", "height"), [(0, 720), (1280, 0), (-1, 720)])
def test_frame_dimensions_must_be_positive(width: int, height: int):
    with pytest.raises(ValidationError):
        frame(frame_width=width, frame_height=height)


def test_relative_area_is_scale_free():
    """The measure that makes 'small distant subjects' testable across resolutions."""
    small = box(x_min=0, y_min=0, x_max=128, y_max=72)

    assert small.relative_area(1280, 720) == pytest.approx(0.01)
    # The same fraction of a smaller frame gives the same answer.
    assert box(x_min=0, y_min=0, x_max=64, y_max=36).relative_area(640, 360) == pytest.approx(0.01)


def test_relative_area_rejects_nonpositive_frames():
    with pytest.raises(ValueError, match="must be positive"):
        box().relative_area(0, 720)


def test_is_within_frame():
    assert box(x_min=0, y_min=0, x_max=100, y_max=100).is_within(1280, 720)
    assert not box(x_min=0, y_min=0, x_max=2000, y_max=100).is_within(1280, 720)


def test_relative_areas_for_a_frame():
    detections = frame(
        detections=[
            Detection(
                box=box(x_min=0, y_min=0, x_max=128, y_max=72),
                confidence=0.5,
                category_id=0,
                category_name="animal",
            )
        ]
    )

    assert detections.relative_areas() == [pytest.approx(0.01)]


def test_box_outside_the_declared_frame_is_rejected():
    """A box escaping the frame means the producer is wrong -- fail at construction."""
    oversized = Detection(
        box=box(x_min=0, y_min=0, x_max=2000, y_max=100),
        confidence=0.5,
        category_id=0,
        category_name="animal",
    )

    with pytest.raises(ValidationError, match="outside the declared"):
        frame(detections=[oversized])


# --------------------------------------------------------------------------- #
# TrackedFrameDetections
#
# Regression guard for a verified data-loss bug: pydantic serializes to the
# *declared* field type, so tracked detections stored in a plain
# FrameDetections (list[Detection]) lost track_id and behavior_label on write.
# --------------------------------------------------------------------------- #


def tracked(**overrides: object) -> TrackedDetection:
    values: dict[str, object] = {
        "box": box(),
        "confidence": 0.8,
        "category_id": 0,
        "category_name": "animal",
        "track_id": 7,
        "behavior_label": "climbing up",
    }
    values.update(overrides)
    return TrackedDetection(**values)  # type: ignore[arg-type]


def test_track_identity_survives_serialization():
    """The exact assertion that fails against a plain FrameDetections."""
    frame_detections = TrackedFrameDetections(
        clip_id="clip-a",
        frame_index=3,
        frame_width=1280,
        frame_height=720,
        detections=[tracked()],
    )

    dumped = frame_detections.model_dump()["detections"][0]

    assert "track_id" in dumped
    assert "behavior_label" in dumped
    assert dumped["track_id"] == 7
    assert dumped["behavior_label"] == "climbing up"


def test_track_identity_survives_a_json_round_trip():
    original = TrackedFrameDetections(
        clip_id="clip-a",
        frame_index=3,
        frame_width=1280,
        frame_height=720,
        detections=[tracked(track_id=11, behavior_label="hanging")],
    )

    restored = TrackedFrameDetections.model_validate_json(original.model_dump_json())

    assert restored == original
    assert restored.detections[0].track_id == 11
    assert restored.detections[0].behavior_label == "hanging"


def test_plain_frame_detections_still_drops_identity():
    """Documents *why* TrackedFrameDetections exists, so nobody 'simplifies' it away."""
    frame_detections = FrameDetections(
        clip_id="clip-a",
        frame_index=0,
        frame_width=1280,
        frame_height=720,
        detections=[tracked()],
    )

    dumped = frame_detections.model_dump()["detections"][0]

    assert "track_id" not in dumped
    assert "behavior_label" not in dumped


def test_track_ids_collects_distinct_identities():
    frame_detections = TrackedFrameDetections(
        clip_id="clip-a",
        frame_index=0,
        frame_width=1280,
        frame_height=720,
        detections=[tracked(track_id=1), tracked(track_id=2), tracked(track_id=1)],
    )

    assert frame_detections.track_ids() == {1, 2}


def test_tracked_frame_rejects_plain_detections():
    """A detection with no identity must not slip into a tracked record."""
    plain = Detection(box=box(), confidence=0.5, category_id=0, category_name="animal")

    with pytest.raises(ValidationError, match="TrackedDetection"):
        TrackedFrameDetections(
            clip_id="clip-a",
            frame_index=0,
            frame_width=1280,
            frame_height=720,
            detections=[plain],  # type: ignore[list-item]
        )


def test_input_file_record_requires_a_sha256():
    record = InputFileRecord(filename="clip.mp4", sha256=SHA, size_bytes=1024)

    assert record.sha256 == SHA

    with pytest.raises(ValidationError, match="sha256"):
        InputFileRecord(filename="clip.mp4", sha256="not-a-digest", size_bytes=1024)


def test_run_metadata_round_trips():
    metadata = RunMetadata(
        experiment_name="phase1-baseline",
        started_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
        git_commit="a" * 40,
        git_dirty=False,
        python_version="3.11.15",
        platform="Linux",
        device=DeviceKind.CPU,
        model_framework="pytorch-wildlife",
        model_name="MegaDetectorV6",
        model_variant="MDV6-yolov9-c",
        confidence_threshold=0.2,
        seed=42,
        inputs=[InputFileRecord(filename="clip.mp4", sha256=SHA, size_bytes=1)],
        output_paths=[Path("artifacts/videos/clip.mp4")],
        elapsed_seconds=12.5,
    )

    restored = RunMetadata.model_validate(metadata.model_dump())

    assert restored == metadata
    assert restored.model_variant == "MDV6-yolov9-c"


def test_run_metadata_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="map"):
        RunMetadata(
            experiment_name="x",
            started_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
            map=0.87,  # type: ignore[call-arg]
        )


def test_enums_have_the_documented_members():
    assert {member.value for member in DeviceKind} == {"auto", "cpu", "cuda", "mps"}
    assert {member.value for member in TrackingBackend} == {"none", "bytetrack", "sort"}
