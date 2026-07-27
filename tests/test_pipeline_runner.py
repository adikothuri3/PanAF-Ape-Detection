"""Tests for the pipeline runner's saved-detection helpers.

These cover the two functions a resumed run depends on. ``run_clip`` itself
needs a decodable video and a detector, so it is exercised through the CLI; the
logic that made a resumed run report nothing lives here, where it can be tested
without either.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from panaf_ape_detection.data.annotations import GroundTruthDetection, GroundTruthFrame
from panaf_ape_detection.pipeline.runner import evaluate_and_write, restore_frames
from panaf_ape_detection.reporting import TRACK_METRICS_SUBDIR
from panaf_ape_detection.types import BoundingBox, TrackedDetection

WIDTH, HEIGHT = 720, 404
BOX = {"x_min": 10.0, "y_min": 10.0, "x_max": 110.0, "y_max": 110.0}


def _document(*, tracked: bool, frames: int = 3) -> dict[str, Any]:
    """A detections document in the shape ``run_clip`` writes."""
    return {
        "clip_id": "clip-a",
        "model": {
            "name": "MegaDetectorV6",
            "variant": "MDV6-yolov10-e",
            "confidence_threshold": 0.2,
        },
        "tracking": {
            "enabled": tracked,
            "backend": "bytetrack" if tracked else None,
            "minimum_track_length": 1,
        },
        "video": {"width": WIDTH, "height": HEIGHT, "fps": 24.0, "frame_count": frames},
        "frames": [
            {
                "clip_id": "clip-a",
                "frame_index": index,
                "frame_width": WIDTH,
                "frame_height": HEIGHT,
                "timestamp_seconds": index / 24.0,
                "detections": [
                    {
                        "box": dict(BOX),
                        "confidence": 0.9,
                        "category_id": 0,
                        "category_name": "animal",
                        **({"track_id": 7, "behavior_label": None} if tracked else {}),
                    }
                ],
            }
            for index in range(frames)
        ],
    }


def _truth(frames: int = 3) -> dict[int, GroundTruthFrame]:
    """One ape, holding still, annotated in every frame."""
    return {
        index: GroundTruthFrame(
            clip_id="clip-a",
            frame_index=index,
            frame_width=WIDTH,
            frame_height=HEIGHT,
            detections=(
                GroundTruthDetection(
                    box=BoundingBox(**BOX), ape_id=0, species="chimpanzee", behaviour="walking"
                ),
            ),
        )
        for index in range(frames)
    }


# --------------------------------------------------------------------------- #
# restore_frames
# --------------------------------------------------------------------------- #


def test_restoring_an_untracked_document_yields_no_tracks():
    per_frame, tracked = restore_frames(_document(tracked=False))

    assert len(per_frame) == 3
    assert all(len(detections) == 1 for detections in per_frame.values())
    # Empty, not a mapping of empty lists: "tracking was off" and "tracking ran
    # and found nothing" are different claims.
    assert tracked == {}


def test_restoring_a_tracked_document_preserves_track_ids():
    per_frame, tracked = restore_frames(_document(tracked=True))

    assert len(tracked) == 3
    assert all(
        isinstance(d, TrackedDetection) for detections in tracked.values() for d in detections
    )
    assert {d.track_id for detections in tracked.values() for d in detections} == {7}
    # The plain view drops the track field rather than carrying it along.
    assert not hasattr(per_frame[0][0], "track_id")


def test_track_fields_do_not_leak_into_plain_detections():
    """Regression: ``Detection`` forbids extras, so splatting a record raised."""
    per_frame, _ = restore_frames(_document(tracked=True))

    assert per_frame[0][0].confidence == 0.9
    assert per_frame[0][0].category_name == "animal"


# --------------------------------------------------------------------------- #
# evaluate_and_write
# --------------------------------------------------------------------------- #


def test_both_metrics_files_are_written(tmp_path: Path):
    per_frame, tracked = restore_frames(_document(tracked=True))

    evaluation, track_evaluation = evaluate_and_write(
        "clip-a",
        per_frame,
        tracked,
        _truth(),
        artifacts=tmp_path,
        confidence_threshold=0.2,
        model_variant="MDV6-yolov10-e",
    )

    assert evaluation is not None
    assert track_evaluation is not None
    assert evaluation.overall.true_positives == 3
    assert track_evaluation.total_id_switches == 0

    detection_metrics = json.loads((tmp_path / "metrics" / "clip-a.json").read_text())
    assert detection_metrics["overall"]["true_positives"] == 3

    track_metrics = tmp_path / "metrics" / TRACK_METRICS_SUBDIR / "clip-a.json"
    assert json.loads(track_metrics.read_text())["predicted_tracks"] == 1


def test_no_ground_truth_writes_nothing(tmp_path: Path):
    per_frame, tracked = restore_frames(_document(tracked=True))

    evaluation, track_evaluation = evaluate_and_write(
        "clip-a",
        per_frame,
        tracked,
        {},
        artifacts=tmp_path,
        confidence_threshold=0.2,
        model_variant="MDV6-yolov10-e",
    )

    assert evaluation is None
    assert track_evaluation is None
    assert not (tmp_path / "metrics").exists()


def test_untracked_run_writes_detection_metrics_only(tmp_path: Path):
    per_frame, tracked = restore_frames(_document(tracked=False))

    evaluation, track_evaluation = evaluate_and_write(
        "clip-a",
        per_frame,
        tracked,
        _truth(),
        artifacts=tmp_path,
        confidence_threshold=0.2,
        model_variant="MDV6-yolov10-e",
    )

    assert evaluation is not None
    assert track_evaluation is None
    assert (tmp_path / "metrics" / "clip-a.json").is_file()
    assert not (tmp_path / "metrics" / TRACK_METRICS_SUBDIR).exists()
