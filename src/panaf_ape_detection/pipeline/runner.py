"""Run the Phase 1 pipeline over the manifest's clips.

Per clip: decode → detect → filter → compare against ground truth → draw →
stitch → record. Each clip is independent and its outputs are skipped if already
present, so a dropped Colab session resumes rather than restarting.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from panaf_ape_detection.config import Config
from panaf_ape_detection.data.annotations import load_ground_truth
from panaf_ape_detection.data.video import iter_frames, read_video_properties
from panaf_ape_detection.evaluation.detection import ClipEvaluation, evaluate_clip
from panaf_ape_detection.evaluation.tracking import ClipTrackEvaluation, evaluate_tracking
from panaf_ape_detection.inference.filtering import filter_by_confidence, keep_animals
from panaf_ape_detection.manifest import ManifestRow
from panaf_ape_detection.provenance import build_run_metadata, file_sha256, write_run_metadata
from panaf_ape_detection.reporting import TRACK_METRICS_SUBDIR
from panaf_ape_detection.types import (
    Detection,
    FrameDetections,
    TrackedDetection,
    TrackedFrameDetections,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from panaf_ape_detection.inference.base import Detector
    from panaf_ape_detection.tracking.base import Tracker

__all__ = ["ClipResult", "build_tracker", "load_manifest", "run_clip", "run_manifest"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClipResult:
    """What one clip's run produced.

    Attributes:
        clip_id: Manifest identifier.
        frames_processed: Frames actually run through the detector.
        detections_kept: Detections surviving confidence and class filtering.
        evaluation: Accuracy against ground truth, when annotations were found.
        track_evaluation: Track quality against ``ape_id``, when tracking ran.
        detections_path: Per-frame detection records.
        video_path: The annotated clip.
        elapsed_seconds: Wall-clock time for this clip.
    """

    clip_id: str
    frames_processed: int
    detections_kept: int
    evaluation: ClipEvaluation | None
    track_evaluation: ClipTrackEvaluation | None = None
    detections_path: Path | None = None
    video_path: Path | None = None
    elapsed_seconds: float = 0.0


def load_manifest(path: Path | str) -> list[ManifestRow]:
    """Read and validate the clip manifest.

    Args:
        path: CSV manifest, as written by ``scripts/fetch_panaf500.py``.

    Returns:
        Validated rows, in file order.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: If a row fails validation.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        msg = (
            f"manifest not found: {manifest_path}. "
            "Run `panaf-phase1 fetch-clips` to select and download a sample."
        )
        raise FileNotFoundError(msg)

    rows: list[ManifestRow] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for number, record in enumerate(csv.DictReader(handle), start=2):
            try:
                rows.append(ManifestRow(**{k: (v or None) for k, v in record.items()}))
            except Exception as exc:
                msg = f"{manifest_path} line {number} is invalid: {exc}"
                raise ValueError(msg) from exc
    return rows


def verify_checksums(rows: list[ManifestRow], raw_root: Path) -> None:
    """Check that every input still matches its recorded digest.

    A mismatch means the inputs changed underneath a result, so every number
    derived from them is suspect. Failing loudly here is much cheaper than
    discovering it in a write-up.

    Raises:
        ValueError: On the first mismatch or missing file.
    """
    for row in rows:
        video = raw_root / "videos" / row.video_filename
        if not video.is_file():
            msg = f"{video} is in the manifest but missing on disk"
            raise ValueError(msg)
        actual = file_sha256(video)
        if actual != row.video_sha256:
            msg = (
                f"{row.clip_id}: checksum mismatch. Manifest says {row.video_sha256[:12]}..., "
                f"file is {actual[:12]}.... The input changed; results would not be reproducible."
            )
            raise ValueError(msg)


def build_tracker(
    config: Config, *, frame_rate: float, frame_width: int, frame_height: int
) -> Tracker | None:
    """Construct the configured tracker, or ``None`` when tracking is off.

    A fresh tracker per clip: track ids mean nothing across videos, so carrying
    state between them would invent continuity that does not exist. The clip's
    geometry is passed explicitly rather than as a
    :class:`~panaf_ape_detection.data.video.VideoProperties`, so the CLI can
    build the same tracker from a saved detection document with no video file
    present.

    Args:
        config: The loaded configuration.
        frame_rate: The clip's frame rate; scales ByteTrack's motion model.
        frame_width: Clip width, used to clamp output boxes.
        frame_height: Clip height, used to clamp output boxes.

    Returns:
        A tracker, or ``None`` if ``tracking.enabled`` is false.

    Raises:
        ValueError: If the configured backend is not implemented. SORT is a
            valid config value but has no implementation, and silently running
            without a tracker would be worse than failing.
    """
    if not config.tracking.enabled:
        return None

    backend = config.tracking.backend.value
    if backend == "bytetrack":
        from panaf_ape_detection.tracking.bytetrack import ByteTrackTracker

        return ByteTrackTracker(
            confidence_threshold=config.model.confidence_threshold,
            frame_rate=frame_rate or 24.0,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    msg = (
        f"tracking.backend {backend!r} is not implemented. "
        "Use 'bytetrack', or set tracking.enabled to false."
    )
    raise ValueError(msg)


def run_clip(
    row: ManifestRow,
    config: Config,
    detector: Detector,
    *,
    raw_root: Path,
    artifacts: Path,
    overwrite: bool = False,
    limit_frames: int | None = None,
) -> ClipResult:
    """Run detection, evaluation and annotation for one clip.

    Args:
        row: The manifest row.
        config: Resolved configuration.
        detector: Anything satisfying the detector protocol.
        raw_root: ``data/raw/panaf500``.
        artifacts: Root artifacts directory.
        overwrite: Re-run even if outputs already exist.
        limit_frames: Stop after this many frames. For smoke runs.

    Returns:
        The clip's :class:`ClipResult`.
    """
    from panaf_ape_detection.tracking.bytetrack import drop_short_tracks
    from panaf_ape_detection.visualization.overlays import draw_frame
    from panaf_ape_detection.visualization.video import VideoWriter

    started = time.perf_counter()
    video_path = raw_root / "videos" / row.video_filename
    properties = read_video_properties(video_path)

    detections_path = artifacts / "detections" / f"{row.clip_id}.json"
    annotated_path = artifacts / "videos" / f"{row.clip_id}_annotated.mp4"

    if not overwrite and detections_path.is_file() and annotated_path.is_file():
        logger.info("%s: outputs already present, skipping", row.clip_id)
        stored = json.loads(detections_path.read_text(encoding="utf-8"))
        return ClipResult(
            clip_id=row.clip_id,
            frames_processed=len(stored.get("frames", [])),
            detections_kept=sum(len(f["detections"]) for f in stored.get("frames", [])),
            evaluation=None,
            detections_path=detections_path,
            video_path=annotated_path,
        )

    ground_truth = {}
    if row.annotation_filename:
        annotation_path = raw_root / "annotations" / row.annotation_filename
        if annotation_path.is_file():
            ground_truth = load_ground_truth(
                annotation_path,
                frame_width=properties.width,
                frame_height=properties.height,
                clip_id=row.clip_id,
            )

    tracker = build_tracker(
        config,
        frame_rate=properties.fps,
        frame_width=properties.width,
        frame_height=properties.height,
    )

    # Pass 1: detect, and track if enabled. The video is not written yet --
    # `drop_short_tracks` needs the whole clip before it can know which tracks
    # are too short to keep, and drawing tracks that are about to be discarded
    # would leave the video disagreeing with the metrics.
    per_frame: dict[int, list[Detection]] = {}
    kept_total = 0

    for frame_index, frame in iter_frames(
        video_path, frame_stride=config.data.frame_stride, limit=limit_frames
    ):
        raw = detector.detect(frame)
        kept: Sequence[Detection] = keep_animals(
            filter_by_confidence(raw, config.model.confidence_threshold)
        )
        if tracker is not None:
            # Frames with no detections must still reach the tracker, or its
            # lost-track buffer never expires.
            kept = tracker.update(kept)
        per_frame[frame_index] = list(kept)
        kept_total += len(kept)

    # Narrow once, explicitly, and reuse for both the video and the metrics --
    # so what is drawn and what is measured cannot diverge.
    tracked_frames: dict[int, list[TrackedDetection]] = {}
    if tracker is not None:
        tracked_frames = drop_short_tracks(
            {
                index: [d for d in detections if isinstance(d, TrackedDetection)]
                for index, detections in per_frame.items()
            },
            config.tracking.minimum_track_length,
        )
        per_frame = {index: list(dets) for index, dets in tracked_frames.items()}

    records: list[FrameDetections] = []
    for frame_index, detections in sorted(per_frame.items()):
        common = {
            "clip_id": row.clip_id,
            "frame_index": frame_index,
            "frame_width": properties.width,
            "frame_height": properties.height,
            "timestamp_seconds": frame_index / properties.fps if properties.fps else None,
        }
        if tracker is not None:
            # TrackedFrameDetections, not FrameDetections: pydantic serialises to
            # the *declared* field type, so the wrong container here silently
            # drops track_id on write -- the exact bug the type exists to prevent.
            records.append(TrackedFrameDetections(**common, detections=tracked_frames[frame_index]))
        else:
            records.append(FrameDetections(**common, detections=detections))

    # Pass 2: decode again and draw the finalised results. Decoding costs a few
    # milliseconds a frame against ~200 ms of inference, so this is far cheaper
    # than holding 360 uncompressed frames in memory.
    with VideoWriter(
        annotated_path,
        width=properties.width,
        height=properties.height,
        fps=config.video.output_fps,
        codec=config.video.codec,
    ) as writer:
        for frame_index, frame in iter_frames(
            video_path, frame_stride=config.data.frame_stride, limit=limit_frames
        ):
            truth_frame = ground_truth.get(frame_index)
            writer.write(
                draw_frame(
                    frame,
                    detections=per_frame.get(frame_index, []),
                    ground_truth=truth_frame.detections if truth_frame else (),
                    frame_index=frame_index,
                    clip_id=row.clip_id,
                    draw_confidence=config.video.draw_confidence,
                    draw_behaviour=config.video.draw_behavior_label,
                    draw_track_id=config.video.draw_track_id,
                )
            )

    detections_path.parent.mkdir(parents=True, exist_ok=True)
    detections_path.write_text(
        json.dumps(
            {
                "clip_id": row.clip_id,
                "model": {
                    "name": detector.name,
                    "variant": detector.variant,
                    "device": detector.device.value,
                    "confidence_threshold": config.model.confidence_threshold,
                },
                "tracking": {
                    "enabled": tracker is not None,
                    "backend": tracker.name if tracker is not None else None,
                    "minimum_track_length": config.tracking.minimum_track_length,
                },
                "video": {
                    "width": properties.width,
                    "height": properties.height,
                    "fps": properties.fps,
                    "frame_count": properties.frame_count,
                },
                "frames": [record.model_dump(mode="json") for record in records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    track_evaluation = None
    if ground_truth and tracker is not None:
        track_evaluation = evaluate_tracking(row.clip_id, tracked_frames, ground_truth)
        # Its own directory, so a glob of metrics/ returns one schema only.
        track_metrics_path = artifacts / "metrics" / TRACK_METRICS_SUBDIR / f"{row.clip_id}.json"
        track_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        track_metrics_path.write_text(
            json.dumps(track_evaluation.as_dict(), indent=2) + "\n", encoding="utf-8"
        )

    evaluation = None
    if ground_truth:
        evaluation = evaluate_clip(
            row.clip_id,
            per_frame,
            ground_truth,
            confidence_threshold=config.model.confidence_threshold,
        )
        metrics_path = artifacts / "metrics" / f"{row.clip_id}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(evaluation.as_dict(), indent=2) + "\n", encoding="utf-8")

    return ClipResult(
        clip_id=row.clip_id,
        frames_processed=len(records),
        detections_kept=kept_total,
        evaluation=evaluation,
        track_evaluation=track_evaluation,
        detections_path=detections_path,
        video_path=annotated_path,
        elapsed_seconds=time.perf_counter() - started,
    )


def run_manifest(
    config: Config,
    detector: Detector,
    *,
    max_clips: int | None = None,
    overwrite: bool = False,
    limit_frames: int | None = None,
    verify: bool = True,
) -> list[ClipResult]:
    """Run every manifest clip and write run metadata.

    Args:
        config: Resolved configuration.
        detector: The detector to run.
        max_clips: Cap the number of clips. Defaults to ``data.max_clips``.
        overwrite: Re-run clips whose outputs already exist.
        limit_frames: Frames per clip, for smoke runs.
        verify: Verify manifest checksums before running.

    Returns:
        One :class:`ClipResult` per clip processed.
    """
    paths = config.repository_paths()
    paths.ensure_artifact_dirs()
    raw_root = paths.raw_data_dir / "panaf500"

    rows = load_manifest(config.data.manifest_path)[: max_clips or config.data.max_clips]
    if verify:
        verify_checksums(rows, raw_root)
        logger.info("verified checksums for %d clips", len(rows))

    started = time.perf_counter()
    results = [
        run_clip(
            row,
            config,
            detector,
            raw_root=raw_root,
            artifacts=paths.artifacts_dir,
            overwrite=overwrite,
            limit_frames=limit_frames,
        )
        for row in rows
    ]
    elapsed = time.perf_counter() - started

    metadata = build_run_metadata(
        config,
        device=detector.device,
        inputs=[raw_root / "videos" / row.video_filename for row in rows],
        outputs=[r.video_path for r in results if r.video_path],
        elapsed_seconds=elapsed,
    )
    write_run_metadata(
        metadata,
        paths.artifacts_dir / "metadata" / f"{config.project.experiment_name}_{int(started)}.json",
    )
    return results


def summarise(results: list[ClipResult]) -> dict[str, object]:
    """Aggregate per-clip evaluations into an overall summary."""
    from panaf_ape_detection.evaluation.detection import MatchCounts

    evaluated = [r.evaluation for r in results if r.evaluation is not None]
    overall = MatchCounts()
    for evaluation in evaluated:
        overall = overall + evaluation.overall
    matched_pairs = sum(e.overall.true_positives for e in evaluated)

    return {
        "clips": len(results),
        "clips_evaluated": len(evaluated),
        "frames_processed": sum(r.frames_processed for r in results),
        "detections_kept": sum(r.detections_kept for r in results),
        "overall": overall.as_dict(),
        # Weighted by matched pairs, not averaged over clips. Each clip's
        # mean_iou is a mean over its true positives, so an unweighted average
        # lets a clip with two matches count as much as one with six hundred --
        # and a clip with *no* matches, whose mean_iou is 0.0 by construction,
        # drag the figure down as though it had localised badly rather than not
        # at all.
        "mean_iou": (
            round(sum(e.mean_iou * e.overall.true_positives for e in evaluated) / matched_pairs, 4)
            if matched_pairs
            else 0.0
        ),
        "matched_pairs": matched_pairs,
        "false_positives_on_empty_frames": sum(
            e.false_positives_on_empty_frames for e in evaluated
        ),
    }
