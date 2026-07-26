"""Tests for PanAf500 annotation loading.

Two traps are guarded here, both verified against real deposit files:

* ``frame_id`` is 1-based while everything else in this project is 0-based;
* behaviour labels use underscores on disk but prose in the documentation.

Both fail *silently* if unhandled -- misaligned boxes look like a mediocre
detector, and a label lookup that never matches looks like missing data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from panaf_ape_detection.data.annotations import (
    BEHAVIOURS,
    canonical_behaviour,
    load_ground_truth,
    normalise_behaviour,
)

WIDTH, HEIGHT = 720, 404


def annotation_file(tmp_path: Path, entries: list[dict[str, Any]], video: str = "clip-a") -> Path:
    path = tmp_path / f"{video}.json"
    path.write_text(json.dumps({"video": video, "annotations": entries}), encoding="utf-8")
    return path


def box(
    x_min: float = 10, y_min: float = 10, x_max: float = 110, y_max: float = 110
) -> list[float]:
    return [x_min, y_min, x_max, y_max]


def detection(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "bbox": box(),
        "ape_id": 0,
        "species": "chimpanzee",
        "behaviour": "climbing_up",
    }
    record.update(overrides)
    return record


def load(path: Path, **kwargs: Any):
    return load_ground_truth(path, frame_width=WIDTH, frame_height=HEIGHT, **kwargs)


# --------------------------------------------------------------------------- #
# Trap 1: frame_id is 1-based
# --------------------------------------------------------------------------- #


def test_frame_id_one_becomes_index_zero(tmp_path: Path):
    """The whole clip shifts by one frame if this conversion is missed."""
    path = annotation_file(tmp_path, [{"frame_id": 1, "detections": [detection()]}])

    frames = load(path)

    assert set(frames) == {0}
    assert frames[0].frame_index == 0


def test_frame_ids_map_across_the_whole_range(tmp_path: Path):
    entries = [{"frame_id": n, "detections": []} for n in range(1, 361)]

    frames = load(annotation_file(tmp_path, entries))

    assert min(frames) == 0
    assert max(frames) == 359
    assert len(frames) == 360


def test_frame_id_below_one_is_rejected(tmp_path: Path):
    """A 0-based file would silently shift everything the other way."""
    path = annotation_file(tmp_path, [{"frame_id": 0, "detections": []}])

    with pytest.raises(ValueError, match="not 1-based"):
        load(path)


def test_missing_frame_id_is_rejected(tmp_path: Path):
    path = annotation_file(tmp_path, [{"detections": []}])

    with pytest.raises(ValueError, match="no 'frame_id'"):
        load(path)


# --------------------------------------------------------------------------- #
# Trap 2: underscore vs prose behaviour labels
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("climbing_up", "climbing_up"),
        ("climbing up", "climbing_up"),
        ("Climbing Up", "climbing_up"),
        ("  hanging  ", "hanging"),
        ("sitting-on-back", "sitting_on_back"),
        ("camera interaction", "camera_interaction"),
    ],
)
def test_behaviour_normalisation(given: str, expected: str):
    assert normalise_behaviour(given) == expected


def test_every_documented_behaviour_round_trips():
    """The nine labels survive prose <-> underscore conversion both ways."""
    for behaviour in BEHAVIOURS:
        assert normalise_behaviour(canonical_behaviour(behaviour)) == behaviour


def test_loader_normalises_labels(tmp_path: Path):
    path = annotation_file(
        tmp_path, [{"frame_id": 1, "detections": [detection(behaviour="Climbing Up")]}]
    )

    frames = load(path)

    assert frames[0].detections[0].behaviour == "climbing_up"


def test_documented_label_set_matches_the_nine():
    assert len(BEHAVIOURS) == 9
    assert "camera_interaction" in BEHAVIOURS
    assert "sitting_on_back" in BEHAVIOURS


# --------------------------------------------------------------------------- #
# Structure and edge cases
# --------------------------------------------------------------------------- #


def test_fields_are_carried_through(tmp_path: Path):
    path = annotation_file(
        tmp_path,
        [
            {
                "frame_id": 7,
                "detections": [detection(ape_id=3, species="gorilla", bbox=box(1, 2, 51, 62))],
            }
        ],
    )

    truth = load(path)[6].detections[0]

    assert truth.ape_id == 3
    assert truth.species == "gorilla"
    assert (truth.box.x_min, truth.box.y_min) == (1.0, 2.0)
    assert truth.box.width == 50.0


def test_empty_frames_are_preserved(tmp_path: Path):
    """Frames with no ape are what make false positives measurable."""
    path = annotation_file(tmp_path, [{"frame_id": 1, "detections": []}])

    frames = load(path)

    assert frames[0].detections == ()
    assert frames[0].is_empty


def test_frame_dimensions_are_recorded(tmp_path: Path):
    path = annotation_file(tmp_path, [{"frame_id": 1, "detections": [detection()]}])

    frame = load(path)[0]

    assert (frame.frame_width, frame.frame_height) == (WIDTH, HEIGHT)


def test_boxes_are_clamped_to_the_frame(tmp_path: Path):
    """Annotations sometimes overshoot by a pixel; clamping beats discarding."""
    path = annotation_file(
        tmp_path,
        [{"frame_id": 1, "detections": [detection(bbox=box(-5, -5, WIDTH + 40, HEIGHT + 40))]}],
    )

    truth = load(path)[0].detections[0]

    assert truth.box.x_min == 0.0
    assert truth.box.x_max == float(WIDTH)
    assert truth.box.y_max == float(HEIGHT)


def test_boxes_entirely_outside_the_frame_are_dropped(tmp_path: Path):
    path = annotation_file(
        tmp_path,
        [{"frame_id": 1, "detections": [detection(bbox=box(WIDTH + 10, 10, WIDTH + 90, 90))]}],
    )

    assert load(path)[0].detections == ()


def test_clip_id_defaults_to_the_video_field(tmp_path: Path):
    path = annotation_file(tmp_path, [{"frame_id": 1, "detections": []}], video="XYZ123")

    assert load(path)[0].clip_id == "XYZ123"


def test_clip_id_can_be_overridden(tmp_path: Path):
    path = annotation_file(tmp_path, [{"frame_id": 1, "detections": []}])

    assert load(path, clip_id="override")[0].clip_id == "override"


def test_malformed_json_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load(path)


def test_file_without_annotations_key_is_rejected(tmp_path: Path):
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"video": "x"}), encoding="utf-8")

    with pytest.raises(ValueError, match="no 'annotations' key"):
        load(path)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "absent.json")
