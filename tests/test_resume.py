"""End-to-end proof that an interrupted run resumes correctly.

A 500-clip Colab run **will** be interrupted, so resuming is not a nicety: it is
the difference between two hours of GPU time being kept or thrown away. That
makes it worth testing against a real decode path rather than by inspection.

Everything here uses a synthetic video and a stub detector, so it needs no
dataset, no weights, no GPU and no network -- but it exercises the same
``run_manifest`` the pipeline uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

cv2 = pytest.importorskip("cv2", reason="requires the inference extra")

from panaf_ape_detection.config import load_config  # noqa: E402
from panaf_ape_detection.pipeline.runner import run_manifest  # noqa: E402
from panaf_ape_detection.provenance import file_sha256  # noqa: E402
from panaf_ape_detection.types import (  # noqa: E402
    BoundingBox,
    Detection,
    DeviceKind,
)

WIDTH, HEIGHT, FRAMES = 160, 120, 6
BOX = (20.0, 20.0, 90.0, 90.0)


class StubDetector:
    """A detector that always finds one box, and counts how often it ran.

    The count is the whole point: after a resume it must not have moved for
    clips that were already done.
    """

    name = "StubDetector"
    variant = "stub-v1"
    device = DeviceKind.CPU

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, _frame: Any) -> list[Detection]:
        """Return one fixed box, and record that the detector actually ran."""
        self.calls += 1
        return [
            Detection(
                box=BoundingBox(x_min=BOX[0], y_min=BOX[1], x_max=BOX[2], y_max=BOX[3]),
                confidence=0.9,
                category_id=0,
                category_name="animal",
            )
        ]


def _write_clip(path: Path) -> None:
    """Write a short synthetic video. Never uses dataset footage."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 24.0, (WIDTH, HEIGHT))
    try:
        for index in range(FRAMES):
            frame = cv2.rectangle(
                cv2.UMat(HEIGHT, WIDTH, cv2.CV_8UC3).get() * 0,
                (20, 20),
                (90, 90),
                (200, 200, 200),
                -1,
            )
            writer.write(frame + index)
    finally:
        writer.release()


@pytest.fixture
def repository(tmp_path: Path, config_data: dict[str, Any], write_config) -> tuple[Path, Path]:
    """A miniature repository with two clips, annotations and a manifest."""
    from panaf_ape_detection.manifest import MANIFEST_COLUMNS

    raw = tmp_path / "data" / "raw" / "panaf500"
    (raw / "videos").mkdir(parents=True)
    (raw / "annotations").mkdir(parents=True)

    rows = []
    for clip_id in ("clip-a", "clip-b"):
        video = raw / "videos" / f"{clip_id}.mp4"
        _write_clip(video)
        (raw / "annotations" / f"{clip_id}.json").write_text(
            json.dumps(
                {
                    "video": clip_id,
                    "annotations": [
                        {
                            "frame_id": index + 1,
                            "detections": [
                                {
                                    "bbox": list(BOX),
                                    "ape_id": 0,
                                    "species": "chimpanzee",
                                    "behaviour": "walking",
                                }
                            ],
                        }
                        for index in range(FRAMES)
                    ],
                }
            ),
            encoding="utf-8",
        )
        digest = file_sha256(video)
        rows.append(
            f"{clip_id},test,{clip_id}.mp4,{clip_id}.json,chimpanzee,,unit test,{digest},{digest},"
        )

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        ",".join(MANIFEST_COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )

    config_data["data"]["manifest_path"] = str(manifest)
    config_data["data"]["max_clips"] = 2
    config_data["paths"]["raw_data_dir"] = str(tmp_path / "data" / "raw")
    config_data["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    config_data["tracking"] = {"enabled": False, "backend": "none", "minimum_track_length": 1}
    # The whole-dataset config turns annotated video off; the resume path has to
    # work in exactly that mode, which is where it previously did not.
    config_data["video"]["write_annotated"] = False

    return write_config(config_data), tmp_path / "artifacts"


def test_a_resumed_run_does_not_re_detect_finished_clips(repository: tuple[Path, Path]):
    """The point of resuming: GPU work already paid for is not paid for twice."""
    config_path, _artifacts = repository
    config = load_config(config_path, use_env_overrides=False)

    first = StubDetector()
    run_manifest(config, first, verify=True)
    assert first.calls == 2 * FRAMES

    second = StubDetector()
    results = run_manifest(config, second, verify=True)

    assert second.calls == 0, "a resumed run re-ran the detector"
    assert len(results) == 2
    assert {r.clip_id for r in results} == {"clip-a", "clip-b"}


def test_a_resumed_run_still_reports_metrics(repository: tuple[Path, Path]):
    """Regression: the skip branch used to return `evaluation=None`.

    A resumed 500-clip run would then have reported nothing at all for every
    clip it had already finished -- the exact failure a dropped Colab session
    produces, and invisible until the summary came out empty.
    """
    config_path, artifacts = repository
    config = load_config(config_path, use_env_overrides=False)

    run_manifest(config, StubDetector(), verify=True)
    resumed = run_manifest(config, StubDetector(), verify=True)

    assert all(r.evaluation is not None for r in resumed)
    assert all(r.evaluation.overall.true_positives == FRAMES for r in resumed if r.evaluation)
    # And the metrics files are still on disk, not merely returned.
    assert (artifacts / "metrics" / "clip-a.json").is_file()


def test_resuming_a_partial_run_completes_the_rest(repository: tuple[Path, Path]):
    """Half-finished is the normal state of an interrupted run."""
    config_path, artifacts = repository
    config = load_config(config_path, use_env_overrides=False)

    run_manifest(config, StubDetector(), verify=True)
    # Simulate an interruption *after* clip-a and before clip-b.
    (artifacts / "detections" / "clip-b.json").unlink()
    (artifacts / "metrics" / "clip-b.json").unlink()

    detector = StubDetector()
    run_manifest(config, detector, verify=True)

    assert detector.calls == FRAMES, "expected exactly the missing clip to be re-detected"
    assert (artifacts / "detections" / "clip-b.json").is_file()
    assert (artifacts / "metrics" / "clip-b.json").is_file()


def test_no_annotated_video_is_written_when_it_is_turned_off(repository: tuple[Path, Path]):
    """500 MP4s is gigabytes nobody watches, and a second decode of every clip."""
    config_path, artifacts = repository
    config = load_config(config_path, use_env_overrides=False)

    results = run_manifest(config, StubDetector(), verify=True)

    assert not list((artifacts / "videos").glob("*.mp4"))
    assert all(r.video_path is None for r in results)


def test_progress_is_logged_for_every_clip(
    repository: tuple[Path, Path], caplog: pytest.LogCaptureFixture
):
    """A run that prints nothing for an hour is indistinguishable from a hang."""
    config_path, _ = repository
    config = load_config(config_path, use_env_overrides=False)

    with caplog.at_level("INFO", logger="panaf_ape_detection.pipeline.runner"):
        run_manifest(config, StubDetector(), verify=True)

    # getMessage(), not .message: the latter is only populated once a handler
    # has formatted the record, so filtering on it silently matches nothing.
    progress = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[")]

    assert len(progress) == 2
    assert progress[0].startswith("[1/2] clip-a")
    assert progress[1].startswith("[2/2] clip-b")
    # The counter is what makes a resumed run legible, so it must be in the line.
    assert "100% done" in progress[1]


def test_the_detect_path_applies_the_configured_refinement(
    repository: tuple[Path, Path], config_data: dict[str, Any], write_config
):
    """`detect` must run the same finishing chain `track` does.

    It did not. Stitching, interpolation and smoothing lived only on the
    re-tracking path, so every *measurement* went through them while every
    *artifact* -- the detections cache, the annotated video -- did not. The
    refinement settings in the config were accepted, validated, and silently
    ignored by the command that produces the deliverables.
    """
    import json

    config_path, artifacts = repository
    config = load_config(config_path, use_env_overrides=False)
    data = json.loads(config_path.read_text()) if config_path.suffix == ".json" else None
    assert data is None  # the fixture writes YAML; keep mypy and readers honest

    # Re-write the config with tracking on and a gap worth filling.
    config_data["data"]["manifest_path"] = str(config.data.manifest_path)
    config_data["data"]["max_clips"] = 2
    config_data["paths"]["raw_data_dir"] = str(config.paths.raw_data_dir)
    config_data["paths"]["artifacts_dir"] = str(artifacts)
    config_data["video"]["write_annotated"] = False
    config_data["tracking"] = {
        "enabled": True,
        "backend": "bytetrack",
        "minimum_track_length": 1,
        "activation_threshold": 0.1,
        "lost_track_buffer": 30,
        "minimum_matching_threshold": 0.8,
        "minimum_consecutive_frames": 1,
        "score_floor": None,
        "stitch_max_gap": 0,
        "stitch_max_distance": 1.0,
        "interpolate_max_gap": 5,
        "smooth_window": 3,
    }
    refined = load_config(write_config(config_data, "refined.yaml"), use_env_overrides=False)

    class _Blinking(StubDetector):
        """Finds the ape except on frame 2, so there is a gap to interpolate."""

        def __init__(self) -> None:
            super().__init__()
            self._frame = -1

        def detect(self, _frame: Any) -> list[Detection]:
            self._frame = (self._frame + 1) % FRAMES
            if self._frame == 2:
                self.calls += 1
                return []
            return super().detect(_frame)

    run_manifest(refined, _Blinking(), verify=True)

    document = json.loads((artifacts / "detections" / "clip-a.json").read_text())
    boxes = [d for f in document["frames"] for d in f["detections"]]
    assert any(d.get("interpolated") for d in boxes), (
        "detect produced no interpolated boxes, so the configured interpolate_max_gap was ignored"
    )
