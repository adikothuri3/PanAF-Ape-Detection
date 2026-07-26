"""Tests for frame decoding, overlays and video stitching.

These need OpenCV, which only the ``inference`` extra installs, so the whole
module skips when it is absent. That keeps CI weights-free and offline while
still exercising the code whenever the extra is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2", reason="requires the inference extra")
np = pytest.importorskip("numpy", reason="requires the inference extra")

from panaf_ape_detection.data.annotations import GroundTruthDetection  # noqa: E402
from panaf_ape_detection.data.video import iter_frames, read_video_properties  # noqa: E402
from panaf_ape_detection.inference.filtering import (  # noqa: E402
    filter_by_confidence,
    keep_animals,
)
from panaf_ape_detection.types import BoundingBox, Detection  # noqa: E402
from panaf_ape_detection.visualization.overlays import draw_frame  # noqa: E402
from panaf_ape_detection.visualization.video import VideoWriter, write_gif  # noqa: E402

WIDTH, HEIGHT, FRAMES = 160, 120, 12


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    """Write a short synthetic video. Never uses dataset footage."""
    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 24.0, (WIDTH, HEIGHT))
    rng = np.random.default_rng(0)
    for _ in range(FRAMES):
        writer.write(rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8))
    writer.release()
    return path


def box(x_min: float = 10, y_min: float = 10, x_max: float = 60, y_max: float = 60) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def test_properties_are_read(synthetic_clip: Path):
    properties = read_video_properties(synthetic_clip)

    assert (properties.width, properties.height) == (WIDTH, HEIGHT)
    assert properties.fps == pytest.approx(24.0, abs=0.5)


def test_all_frames_are_yielded(synthetic_clip: Path):
    frames = list(iter_frames(synthetic_clip))

    assert len(frames) == FRAMES
    assert [index for index, _ in frames] == list(range(FRAMES))


def test_stride_yields_positions_not_a_running_count(synthetic_clip: Path):
    """The index must stay the position in the video, or ground truth misaligns."""
    frames = list(iter_frames(synthetic_clip, frame_stride=3))

    assert [index for index, _ in frames] == [0, 3, 6, 9]


def test_limit_stops_early(synthetic_clip: Path):
    assert len(list(iter_frames(synthetic_clip, limit=4))) == 4


def test_zero_stride_is_rejected(synthetic_clip: Path):
    with pytest.raises(ValueError, match="frame_stride"):
        list(iter_frames(synthetic_clip, frame_stride=0))


def test_missing_video_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_video_properties(tmp_path / "absent.mp4")


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


def detection(confidence: float, name: str = "animal") -> Detection:
    return Detection(
        box=box(),
        confidence=confidence,
        category_id=0 if name == "animal" else 1,
        category_name=name,
    )


def test_confidence_filtering_is_inclusive_at_the_threshold():
    kept = filter_by_confidence([detection(0.2), detection(0.19)], 0.2)

    assert len(kept) == 1


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        filter_by_confidence([], 1.5)


def test_only_animals_are_kept():
    kept = keep_animals([detection(0.9, "animal"), detection(0.9, "person")])

    assert len(kept) == 1
    assert kept[0].category_name == "animal"


# --------------------------------------------------------------------------- #
# Overlays
# --------------------------------------------------------------------------- #


def blank():
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def test_drawing_does_not_mutate_the_source_frame():
    """A bug in the overlay must never be able to alter recorded pixels."""
    frame = blank()
    original = frame.copy()

    draw_frame(frame, detections=[detection(0.9)])

    assert np.array_equal(frame, original)


def test_drawing_changes_the_copy():
    result = draw_frame(blank(), detections=[detection(0.9)])

    assert not np.array_equal(result, blank())


def test_predictions_and_ground_truth_use_different_colours():
    """A viewer must never mistake a prediction for ground truth."""
    truth = GroundTruthDetection(
        box=box(80, 80, 120, 110), ape_id=0, species="chimpanzee", behaviour="sitting"
    )

    predicted_only = draw_frame(blank(), detections=[detection(0.9)])
    truth_only = draw_frame(blank(), ground_truth=[truth])

    assert not np.array_equal(predicted_only, truth_only)


def test_drawing_survives_an_empty_frame():
    result = draw_frame(blank(), frame_index=0, clip_id="clip-a")

    assert result.shape == (HEIGHT, WIDTH, 3)


# --------------------------------------------------------------------------- #
# Stitching
# --------------------------------------------------------------------------- #


def test_written_video_reads_back(tmp_path: Path):
    target = tmp_path / "out.mp4"

    with VideoWriter(target, width=WIDTH, height=HEIGHT, fps=24.0) as writer:
        for _ in range(FRAMES):
            writer.write(blank())

    assert target.is_file()
    assert len(list(iter_frames(target))) == FRAMES


def test_mismatched_frame_size_is_rejected(tmp_path: Path):
    """OpenCV drops mismatched frames silently, truncating the video."""
    with (
        VideoWriter(tmp_path / "out.mp4", width=WIDTH, height=HEIGHT, fps=24.0) as writer,
        pytest.raises(ValueError, match="would drop it silently"),
    ):
        writer.write(np.zeros((HEIGHT * 2, WIDTH, 3), dtype=np.uint8))


def test_writer_rejects_nonpositive_geometry(tmp_path: Path):
    with pytest.raises(ValueError, match="dimensions"):
        VideoWriter(tmp_path / "x.mp4", width=0, height=HEIGHT, fps=24.0)
    with pytest.raises(ValueError, match="fps"):
        VideoWriter(tmp_path / "y.mp4", width=WIDTH, height=HEIGHT, fps=0)


def test_gif_export(tmp_path: Path):
    target = write_gif(tmp_path / "out.gif", [blank() for _ in range(4)], fps=8.0)

    assert target.is_file()
    assert target.stat().st_size > 0


def test_gif_with_no_frames_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="no frames"):
        write_gif(tmp_path / "out.gif", [])
