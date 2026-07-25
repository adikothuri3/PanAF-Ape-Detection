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


def test_frame_detections_defaults_to_empty():
    frame = FrameDetections(clip_id="clip-a", frame_index=0)

    assert frame.detections == []
    assert frame.timestamp_seconds is None


def test_frame_detections_rejects_negative_index():
    with pytest.raises(ValidationError, match="frame_index"):
        FrameDetections(clip_id="clip-a", frame_index=-1)


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
