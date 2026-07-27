"""Tests for reading and aggregating a run's artifacts.

**The first test here is the bug that prompted this module.** ``artifacts/metrics/``
held two incompatible schemas distinguished only by a filename suffix, and the
notebook globbed the directory and read ``m["overall"]``. It crashed with
``KeyError: 'overall'`` for every user who enabled tracking, and no test could
reach it because the code lived in a notebook cell.

Everything here runs offline with no artifacts of its own: the fixtures build a
miniature artifacts tree in ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from panaf_ape_detection.reporting import (
    TRACK_METRICS_SUBDIR,
    PooledCounts,
    format_recall_table,
    latest_run_metadata,
    load_detection_metrics,
    load_track_metrics,
    pooled_counts,
    pooled_track_metrics,
    recall_by_behaviour,
    recall_by_size,
    score_bands,
    variants_in,
)


def counts(tp: int = 0, fp: int = 0, fn: int = 0) -> dict[str, Any]:
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn}


def detection_metrics(clip_id: str, *, tp: int = 10, fp: int = 2, fn: int = 5, iou: float = 0.8):
    return {
        "clip_id": clip_id,
        "iou_threshold": 0.5,
        "confidence_threshold": 0.2,
        "frames_evaluated": 360,
        "empty_frames": 4,
        "false_positives_on_empty_frames": 1,
        "mean_iou": iou,
        "overall": counts(tp, fp, fn),
        "by_behaviour": {"walking": counts(tp, 0, fn)},
        "by_size": {"small": counts(tp, 0, fn)},
    }


def track_metrics(clip_id: str) -> dict[str, Any]:
    """Note the overlap with detection metrics: clip_id, frames_evaluated, iou_threshold.

    Enough shared keys to look like the same thing to a careless reader, which is
    exactly how the original bug survived review.
    """
    return {
        "clip_id": clip_id,
        "iou_threshold": 0.5,
        "frames_evaluated": 360,
        "annotated_individuals": 2,
        "predicted_tracks": 4,
        "total_id_switches": 2,
        "mean_fragmentation": 2.0,
        "mean_coverage": 0.5,
        "mean_identity_coverage": 0.375,
        "mean_track_purity": 0.9,
        "id_merges": 1,
        "mean_jitter": 0.02,
        "mostly_tracked": 1,
        "mostly_lost": 1,
        # 100 annotated frames each; ape 0 fully covered by one track, ape 1
        # covered half the time and split so its best track holds a quarter.
        "individuals": [
            {
                "ape_id": 0,
                "annotated_frames": 100,
                "covered_frames": 100,
                "coverage": 1.0,
                "identity_coverage": 1.0,
                "id_switches": 0,
                "fragmentation": 1,
            },
            {
                "ape_id": 1,
                "annotated_frames": 100,
                "covered_frames": 50,
                "coverage": 0.5,
                "identity_coverage": 0.25,
                "id_switches": 1,
                "fragmentation": 2,
            },
        ],
    }


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    """A run with two clips, tracking enabled, in the current layout."""
    root = tmp_path / "artifacts"
    for clip in ("clip-a", "clip-b"):
        write(root / "metrics" / f"{clip}.json", detection_metrics(clip))
        write(root / "metrics" / TRACK_METRICS_SUBDIR / f"{clip}.json", track_metrics(clip))
    return root


# --------------------------------------------------------------------------- #
# The regression: two schemas must never be confused
# --------------------------------------------------------------------------- #


def test_track_metrics_are_not_loaded_as_detection_metrics(artifacts: Path):
    """The exact failure: KeyError 'overall' from globbing metrics/*.json."""
    loaded = load_detection_metrics(artifacts)

    assert len(loaded) == 2
    assert all("overall" in document for document in loaded)
    assert [d["clip_id"] for d in loaded] == ["clip-a", "clip-b"]


def test_legacy_suffixed_track_metrics_are_skipped(tmp_path: Path):
    """Artifact trees written before the split must still load.

    The user's Drive copies use `metrics/<clip>_tracking.json`; refusing to read
    them would mean re-running 30 minutes of inference for a layout change.
    """
    root = tmp_path / "artifacts"
    write(root / "metrics" / "clip-a.json", detection_metrics("clip-a"))
    write(root / "metrics" / "clip-a_tracking.json", track_metrics("clip-a"))

    loaded = load_detection_metrics(root)

    assert len(loaded) == 1
    assert loaded[0]["clip_id"] == "clip-a"


def test_legacy_track_metrics_are_still_readable(tmp_path: Path):
    root = tmp_path / "artifacts"
    write(root / "metrics" / "clip-a.json", detection_metrics("clip-a"))
    write(root / "metrics" / "clip-a_tracking.json", track_metrics("clip-a"))

    tracks = load_track_metrics(root)

    assert [t["clip_id"] for t in tracks] == ["clip-a"]
    assert tracks[0]["total_id_switches"] == 2


def test_an_unrecognised_file_in_metrics_is_an_error_not_a_silent_skip(tmp_path: Path):
    """Skipping quietly is how a partial result gets reported as a whole one."""
    root = tmp_path / "artifacts"
    write(root / "metrics" / "clip-a.json", detection_metrics("clip-a"))
    write(root / "metrics" / "something-else.json", {"clip_id": "x", "unexpected": True})

    with pytest.raises(ValueError, match="not detection metrics"):
        load_detection_metrics(root)


def test_missing_metrics_directory_says_what_to_run(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="panaf-phase1 detect"):
        load_detection_metrics(tmp_path / "artifacts")


def test_track_metrics_are_empty_when_tracking_was_off(tmp_path: Path):
    root = tmp_path / "artifacts"
    write(root / "metrics" / "clip-a.json", detection_metrics("clip-a"))

    assert load_track_metrics(root) == []


# --------------------------------------------------------------------------- #
# Pooling -- hand-computed
# --------------------------------------------------------------------------- #


def test_pooled_counts_sum_across_clips():
    metrics = [
        detection_metrics("a", tp=10, fp=2, fn=5),
        detection_metrics("b", tp=30, fp=8, fn=15),
    ]

    pooled = pooled_counts(metrics)

    assert (pooled.true_positives, pooled.false_positives, pooled.false_negatives) == (40, 10, 20)
    assert pooled.precision == pytest.approx(40 / 50)
    assert pooled.recall == pytest.approx(40 / 60)
    assert pooled.f1 == pytest.approx(2 * (0.8 * (2 / 3)) / (0.8 + 2 / 3))
    assert pooled.detections == 50
    assert pooled.annotated == 60


def test_pooled_mean_iou_weights_by_matched_pairs():
    """A clip that matched nothing has mean_iou 0.0 by construction.

    Averaging that in reports bad localisation where there was none at all --
    the bug that once made tracking look as though it had wrecked box quality.
    """
    metrics = [
        detection_metrics("a", tp=100, fp=0, fn=0, iou=0.9),
        detection_metrics("b", tp=0, fp=3, fn=50, iou=0.0),
    ]

    assert pooled_counts(metrics).mean_iou == pytest.approx(0.9)


def test_empty_input_does_not_divide_by_zero():
    pooled = pooled_counts([])

    assert (pooled.precision, pooled.recall, pooled.f1, pooled.mean_iou) == (0.0, 0.0, 0.0, 0.0)
    assert PooledCounts().detections == 0


def test_recall_breakdowns_pool_over_clips():
    metrics = [
        {**detection_metrics("a"), "by_behaviour": {"walking": counts(3, 0, 7)}},
        {
            **detection_metrics("b"),
            "by_behaviour": {"walking": counts(1, 0, 9), "hanging": counts(0, 0, 5)},
        },
    ]

    behaviour = recall_by_behaviour(metrics)

    assert behaviour["walking"] == (4, 20)
    assert behaviour["hanging"] == (0, 5)


def test_recall_by_size_reads_the_size_bands():
    metrics = [{**detection_metrics("a"), "by_size": {"large": counts(2, 0, 2)}}]

    assert recall_by_size(metrics)["large"] == (2, 4)


# --------------------------------------------------------------------------- #
# Score bands -- the tracker-floor question
# --------------------------------------------------------------------------- #


def test_score_bands_bucket_against_the_tracker_floor(tmp_path: Path):
    """Boundaries are inclusive at the top: 0.10 is discarded, 0.15 cannot start."""
    root = tmp_path / "artifacts"
    write(
        root / "detections" / "clip-a.json",
        {
            "clip_id": "clip-a",
            "frames": [
                {"detections": [{"confidence": c} for c in (0.05, 0.10, 0.11, 0.15, 0.16, 0.9)]}
            ],
        },
    )

    bands = score_bands(root)

    assert (bands.discarded, bands.cannot_start_track, bands.usable) == (2, 2, 2)
    assert bands.total == 6
    assert bands.usable_fraction == pytest.approx(1 / 3)


def test_score_bands_on_a_run_with_no_detections(tmp_path: Path):
    assert score_bands(tmp_path / "artifacts").total == 0
    assert score_bands(tmp_path / "artifacts").usable_fraction == 0.0


# --------------------------------------------------------------------------- #
# Run metadata
# --------------------------------------------------------------------------- #


def test_latest_run_metadata_uses_mtime_not_filename(tmp_path: Path):
    """Once experiment names differ, the lexically last file is the wrong one.

    `variant-yolov9c_1.json` sorts after `phase1-colab_9.json`, so a name-based
    pick would show the wrong run's provenance beside the right run's numbers.
    """
    import os

    root = tmp_path / "artifacts"
    older = root / "metadata" / "variant-yolov9c_1.json"
    newer = root / "metadata" / "phase1-colab_9.json"
    write(older, {"experiment_name": "variant-yolov9c"})
    write(newer, {"experiment_name": "phase1-colab"})
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    assert sorted(p.name for p in (root / "metadata").glob("*.json"))[-1] == older.name
    latest = latest_run_metadata(root)
    assert latest is not None
    assert latest["experiment_name"] == "phase1-colab"


def test_latest_run_metadata_is_none_when_nothing_has_run(tmp_path: Path):
    assert latest_run_metadata(tmp_path / "artifacts") is None


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def test_recall_table_pins_a_requested_order_and_keeps_the_rest():
    rendered = format_recall_table(
        {"large": (1, 10), "small": (5, 10), "unexpected": (0, 4)},
        order=("small", "medium", "large"),
    )
    lines = [line.split()[0] for line in rendered.splitlines()]

    assert lines == ["small", "large", "unexpected"]  # medium absent, unexpected appended


def test_recall_table_handles_nothing_recorded():
    assert "nothing recorded" in format_recall_table({})


# --------------------------------------------------------------------------- #
# Mixed-model detection
#
# `detect --overwrite` rewrites metrics clip by clip, so a run that is killed
# part-way leaves some clips on the old model. Each file is individually valid;
# only comparing them reveals it. This happened.
# --------------------------------------------------------------------------- #


def test_a_mixed_metrics_directory_is_visible(tmp_path: Path):
    root = tmp_path / "artifacts"
    write(
        root / "metrics" / "clip-a.json",
        {**detection_metrics("clip-a"), "model_variant": "MDV6-yolov10-e"},
    )
    write(
        root / "metrics" / "clip-b.json",
        {**detection_metrics("clip-b"), "model_variant": "MDV6-yolov9-c"},
    )

    found = variants_in(load_detection_metrics(root))

    assert found == {"MDV6-yolov10-e", "MDV6-yolov9-c"}
    assert len(found) > 1, "more than one variant means the pooled number is meaningless"


def test_a_consistent_directory_reports_one_variant(tmp_path: Path):
    root = tmp_path / "artifacts"
    for clip in ("clip-a", "clip-b"):
        write(
            root / "metrics" / f"{clip}.json",
            {**detection_metrics(clip), "model_variant": "MDV6-yolov10-e"},
        )

    assert variants_in(load_detection_metrics(root)) == {"MDV6-yolov10-e"}


def test_records_written_before_the_field_existed_report_empty(tmp_path: Path):
    root = tmp_path / "artifacts"
    write(root / "metrics" / "clip-a.json", detection_metrics("clip-a"))

    assert variants_in(load_detection_metrics(root)) == {""}


# --------------------------------------------------------------------------- #
# pooled_track_metrics
# --------------------------------------------------------------------------- #


def test_pooled_track_metrics_sum_across_clips(tmp_path: Path):
    root = tmp_path / "artifacts"
    for clip in ("clip-a", "clip-b"):
        write(root / "metrics" / TRACK_METRICS_SUBDIR / f"{clip}.json", track_metrics(clip))

    pooled = pooled_track_metrics(load_track_metrics(root))

    assert pooled.clips == 2
    assert pooled.individuals == 4
    assert pooled.predicted_tracks == 8
    assert pooled.id_switches == 4
    assert pooled.id_merges == 2
    # 2 clips x (100 + 50) covered of (100 + 100) annotated.
    assert pooled.coverage == pytest.approx(0.75)
    # 2 clips x (100 + 25) dominant of 200 annotated.
    assert pooled.identity_coverage == pytest.approx(0.625)
    assert pooled.fragmentation == pytest.approx(2.0)
    assert pooled.track_purity == pytest.approx(0.9)
    assert pooled.jitter == pytest.approx(0.02)


def test_pooled_track_coverage_weights_by_frames_not_by_clip(tmp_path: Path):
    """A 400-frame ape and a 100-frame ape must not carry equal weight."""
    root = tmp_path / "artifacts"
    write(
        root / "metrics" / TRACK_METRICS_SUBDIR / "clip-a.json",
        {
            **track_metrics("clip-a"),
            "individuals": [
                {
                    "ape_id": 0,
                    "annotated_frames": 400,
                    "covered_frames": 400,
                    "identity_coverage": 1.0,
                },
                {
                    "ape_id": 1,
                    "annotated_frames": 100,
                    "covered_frames": 0,
                    "identity_coverage": 0.0,
                },
            ],
        },
    )

    pooled = pooled_track_metrics(load_track_metrics(root))

    # 400 of 500, not the 0.5 a per-ape mean would give.
    assert pooled.coverage == pytest.approx(0.8)
    assert pooled.identity_coverage == pytest.approx(0.8)


def test_pooled_track_metrics_of_nothing_is_empty():
    pooled = pooled_track_metrics([])

    assert pooled.clips == 0
    assert pooled.coverage == 0.0
    assert pooled.identity_coverage == 0.0
    assert pooled.fragmentation == 0.0
    # No tracks measured is not the same as impure tracks.
    assert pooled.track_purity == 1.0


def test_records_predating_the_identity_fields_do_not_claim_perfect_identity(tmp_path: Path):
    """An old file must contribute no identity coverage rather than all of it.

    Falling back to `covered_frames` would credit a run with identity it was
    never measured for, and make an old baseline look better than a new one.
    """
    root = tmp_path / "artifacts"
    legacy = {
        "clip_id": "clip-a",
        "predicted_tracks": 4,
        "total_id_switches": 2,
        "individuals": [{"ape_id": 0, "annotated_frames": 100, "covered_frames": 90}],
    }
    write(root / "metrics" / TRACK_METRICS_SUBDIR / "clip-a.json", legacy)

    pooled = pooled_track_metrics(load_track_metrics(root))

    assert pooled.coverage == pytest.approx(0.9)
    assert pooled.identity_coverage == 0.0
    assert pooled.purity_count == 0
    assert pooled.track_purity == 1.0
