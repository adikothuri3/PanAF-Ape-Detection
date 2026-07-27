"""Tests for re-tracking saved detections and sweeping tracker settings.

Everything that does not need a tracker runs without the ``inference`` extra;
the parts that actually associate boxes are gated behind ``importorskip``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from panaf_ape_detection.config import load_config
from panaf_ape_detection.pipeline.retrack import (
    ClipSource,
    RetrackSettings,
    expand_grid,
    load_clip,
)

WIDTH, HEIGHT = 720, 404


def _detections_document(scores: list[float]) -> dict[str, Any]:
    """One clip, one box per frame, at the given scores."""
    return {
        "clip_id": "clip-a",
        "model": {"name": "MegaDetectorV6", "confidence_threshold": 0.05},
        "tracking": {"enabled": False, "backend": None, "minimum_track_length": 5},
        "video": {"width": WIDTH, "height": HEIGHT, "fps": 24.0, "frame_count": len(scores)},
        "frames": [
            {
                "clip_id": "clip-a",
                "frame_index": index,
                "frame_width": WIDTH,
                "frame_height": HEIGHT,
                "detections": [
                    {
                        "box": {"x_min": 10.0, "y_min": 10.0, "x_max": 110.0, "y_max": 110.0},
                        "confidence": score,
                        "category_id": 0,
                        "category_name": "animal",
                    }
                ],
            }
            for index, score in enumerate(scores)
        ],
    }


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def test_settings_from_config_match_the_pipeline(valid_config_file: Path):
    """A sweep's baseline arm must be exactly what `detect` would do."""
    loaded = load_config(valid_config_file, use_env_overrides=False)

    settings = RetrackSettings.from_config(loaded)

    assert settings.activation_threshold == loaded.model.confidence_threshold
    assert settings.lost_track_buffer == loaded.tracking.lost_track_buffer
    assert settings.minimum_matching_threshold == loaded.tracking.minimum_matching_threshold
    assert settings.minimum_consecutive_frames == loaded.tracking.minimum_consecutive_frames
    assert settings.minimum_track_length == loaded.tracking.minimum_track_length
    # Nothing is filtered out unless asked for.
    assert settings.detection_floor == 0.0


def test_settings_from_config_honour_an_explicit_activation_threshold(
    write_config, config_data: dict[str, Any]
):
    config_data["model"]["confidence_threshold"] = 0.05
    config_data["tracking"]["activation_threshold"] = 0.25
    loaded = load_config(write_config(config_data), use_env_overrides=False)

    assert RetrackSettings.from_config(loaded).activation_threshold == 0.25


def test_with_values_rejects_an_unknown_setting():
    """A typo in a sweep grid would otherwise sweep nothing and look like a null result."""
    with pytest.raises(ValueError, match="unknown tracker setting"):
        RetrackSettings().with_values({"lost_frame_buffer": 30})


def test_with_values_leaves_the_original_untouched():
    original = RetrackSettings(lost_track_buffer=30)

    updated = original.with_values({"lost_track_buffer": 90})

    assert original.lost_track_buffer == 30
    assert updated.lost_track_buffer == 90


# --------------------------------------------------------------------------- #
# Grid expansion
# --------------------------------------------------------------------------- #


def test_expanding_a_grid_takes_every_combination():
    arms = expand_grid(
        RetrackSettings(),
        {"lost_track_buffer": [30, 60], "minimum_matching_threshold": [0.7, 0.8, 0.9]},
    )

    assert len(arms) == 6
    assert len({(a.lost_track_buffer, a.minimum_matching_threshold) for a in arms}) == 6


def test_expanding_a_grid_is_deterministic():
    axes = {"minimum_track_length": [1, 5], "lost_track_buffer": [30, 60]}

    assert expand_grid(RetrackSettings(), axes) == expand_grid(RetrackSettings(), axes)


def test_an_empty_grid_is_a_sweep_of_one_arm():
    """Not a sweep of none -- the baseline is still worth measuring."""
    assert expand_grid(RetrackSettings(), {}) == [RetrackSettings()]


def test_axes_carry_over_settings_not_being_swept():
    arms = expand_grid(RetrackSettings(minimum_track_length=7), {"lost_track_buffer": [30, 60]})

    assert {arm.minimum_track_length for arm in arms} == {7}


def test_an_axis_with_no_values_is_rejected():
    with pytest.raises(ValueError, match="list no values"):
        expand_grid(RetrackSettings(), {"lost_track_buffer": []})


def test_an_unknown_axis_is_rejected():
    with pytest.raises(ValueError, match="unknown tracker setting"):
        expand_grid(RetrackSettings(), {"buffer_size": [30]})


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_loading_a_clip_needs_no_video_file(tmp_path: Path):
    """Frame size comes from the detections document, not from decoding."""
    detections = tmp_path / "clip-a.json"
    detections.write_text(json.dumps(_detections_document([0.9, 0.9])), encoding="utf-8")
    annotation = tmp_path / "clip-a-annotations.json"
    annotation.write_text(
        json.dumps(
            {
                "video": "clip-a",
                "annotations": [
                    {
                        "frame_id": 1,
                        "detections": [
                            {
                                "bbox": [10.0, 10.0, 110.0, 110.0],
                                "ape_id": 0,
                                "species": "chimpanzee",
                                "behaviour": "walking",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    document, truth = load_clip(
        ClipSource(clip_id="clip-a", detections_path=detections, annotation_path=annotation)
    )

    assert document["clip_id"] == "clip-a"
    assert truth[0].frame_width == WIDTH


# --------------------------------------------------------------------------- #
# Tracking -- needs the inference extra
# --------------------------------------------------------------------------- #


def test_detection_floor_filters_before_tracking():
    """The floor must remove boxes, not merely be recorded in the settings."""
    pytest.importorskip("supervision", reason="requires the inference extra")
    from panaf_ape_detection.pipeline.retrack import retrack_document

    document = _detections_document([0.9] * 10)
    settings = RetrackSettings(minimum_track_length=1, activation_threshold=0.2)

    kept = retrack_document(document, settings)
    dropped = retrack_document(document, settings.with_values({"detection_floor": 0.95}))

    assert sum(len(d) for d in kept.values()) > 0
    assert sum(len(d) for d in dropped.values()) == 0
    # Frames still reach the tracker even when everything in them was filtered,
    # otherwise its lost-track buffer would never expire.
    assert len(dropped) == 10


def test_stored_track_ids_are_recomputed_not_reused():
    """Otherwise a sweep would report the settings of whichever run wrote the file."""
    pytest.importorskip("supervision", reason="requires the inference extra")
    from panaf_ape_detection.pipeline.retrack import retrack_document

    document = _detections_document([0.9] * 10)
    for frame in document["frames"]:
        for detection in frame["detections"]:
            detection["track_id"] = 999
            detection["behavior_label"] = None
    document["tracking"]["enabled"] = True

    tracked = retrack_document(document, RetrackSettings(minimum_track_length=1))

    found = {d.track_id for detections in tracked.values() for d in detections}
    assert found
    assert 999 not in found
