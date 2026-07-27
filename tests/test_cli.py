"""Tests for the ``panaf-phase1`` command-line interface.

These tests must pass in an environment installed *without* the ``inference``
extra, and must never download model weights or require a GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from panaf_ape_detection import __version__
from panaf_ape_detection.cli import app
from panaf_ape_detection.config import load_config
from panaf_ape_detection.manifest import MANIFEST_COLUMNS
from panaf_ape_detection.reporting import TRACKING_METRICS_SCHEMA

WriteConfig = Callable[..., Path]

runner = CliRunner()


def invoke(*args: str):
    return runner.invoke(app, list(args))


def unwrap(output: str) -> str:
    """Collapse rich's terminal line-wrapping so substrings match reliably."""
    return " ".join(output.split())


def registered_command_names() -> set[str]:
    """Return the command names Typer will expose, independent of help layout."""
    return {
        command.name or (command.callback.__name__ if command.callback else "")
        for command in app.registered_commands
    }


def test_only_implemented_commands_are_registered():
    """A command may only be registered once it is implemented and tested.

    Standalone frame extraction is still unimplemented, so it must stay absent
    rather than appear as a stub.
    """
    assert registered_command_names() == {
        "doctor",
        "validate-config",
        "show-paths",
        "fetch-clips",
        "detect",
        "evaluate",
        "track",
        "track-sweep",
    }


def test_help_lists_the_implemented_commands():
    result = invoke("--help")

    assert result.exit_code == 0
    for command in ("doctor", "validate-config", "show-paths"):
        assert command in result.output


def test_version_flag():
    result = invoke("--version")

    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_shows_help():
    result = invoke()

    assert result.exit_code != 0
    assert "Usage" in result.output


def test_doctor_runs_without_inference_extra():
    result = invoke("doctor")

    assert result.exit_code == 0, result.output
    for expected in (
        "Python",
        "Operating system",
        "Repository root",
        "FFmpeg",
        "CUDA available",
        "MPS",
        "Optional inference stack",
        "Expected directories",
    ):
        assert expected in result.output


def test_doctor_says_it_only_inspects_the_environment():
    result = invoke("doctor")

    assert "only inspects the environment" in unwrap(result.output)


def test_show_paths_reports_layout_and_artifacts():
    result = invoke("show-paths")

    assert result.exit_code == 0, result.output
    assert "Repository layout" in result.output
    assert "Artifact subdirectories" in result.output
    for name in ("detections", "frames", "metadata", "metrics", "videos", "visualizations"):
        assert name in result.output


def test_validate_config_accepts_the_shipped_base_config():
    result = invoke("validate-config", "--config", "configs/base.yaml")

    assert result.exit_code == 0, result.output
    assert "is valid" in result.output


def test_validate_config_accepts_the_shipped_colab_config():
    result = invoke("validate-config", "--config", "configs/colab.yaml", "--no-show")

    assert result.exit_code == 0, result.output


def test_validate_config_defaults_to_base_config():
    result = invoke("validate-config")

    assert result.exit_code == 0, result.output
    assert "configs/base.yaml" in result.output


def test_validate_config_shows_resolved_values():
    result = invoke("validate-config", "--config", "configs/base.yaml")

    # The shipped default, whatever it currently is -- asserting a specific
    # variant here made this test a tripwire for changing models rather than a
    # test that the resolved value is displayed.
    assert load_config(Path("configs/base.yaml")).model.variant in result.output
    assert "Resolved configuration" in result.output


def test_validate_config_fails_on_missing_file():
    result = invoke("validate-config", "--config", "configs/nope.yaml")

    assert result.exit_code == 1
    assert "Configuration invalid" in result.output


def test_validate_config_fails_on_unknown_key(
    write_config: WriteConfig, config_data: dict[str, Any]
):
    config_data["model"]["typo_key"] = 1

    result = invoke("validate-config", "--config", str(write_config(config_data)))

    assert result.exit_code == 1
    assert "typo_key" in result.output


def test_validate_config_warns_about_missing_manifest(
    tmp_path: Path, write_config: WriteConfig, config_data: dict[str, Any]
):
    config_data["data"]["manifest_path"] = str(tmp_path / "absent_manifest.csv")

    result = invoke("validate-config", "--config", str(write_config(config_data)))

    assert result.exit_code == 0, result.output
    assert "Manifest not present yet" in result.output


def test_validate_config_flags_unrecognised_variant(
    write_config: WriteConfig, config_data: dict[str, Any]
):
    config_data["model"]["variant"] = "MDV6-not-a-real-variant"

    result = invoke("validate-config", "--config", str(write_config(config_data)))

    assert result.exit_code == 0, result.output
    # rich hard-wraps to the terminal width, so "docs/obsidian/05 Technical/model.md" can be
    # split across lines. Collapse whitespace before matching.
    assert "docs/obsidian/05 Technical/model.md" in unwrap(result.output)


@pytest.mark.parametrize("command", ["doctor", "show-paths"])
def test_commands_are_read_only(command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Inspection commands must not create directories as a side effect."""
    monkeypatch.setenv("PANAF_REPO_ROOT", str(tmp_path))

    result = invoke(command)

    assert result.exit_code == 0, result.output
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# evaluate / track over saved detections
#
# Both read `artifacts/detections/`, so they can be exercised end to end with no
# model, no video and no GPU -- a temporary repository root is enough.
# --------------------------------------------------------------------------- #

BOX = [10.0, 10.0, 110.0, 110.0]


def _saved_run(
    tmp_path: Path,
    config_data: dict[str, Any],
    write_config: WriteConfig,
    *,
    confidence: float = 0.2,
    scores: list[float] | None = None,
    tracking: bool = False,
    stored_track_ids: bool = False,
) -> Path:
    """Lay out a miniature repository: manifest, annotations, saved detections.

    One clip, three frames, one annotated ape holding still, and one detection
    per frame at *scores*.

    *stored_track_ids* writes the ``track_id`` and ``behavior_label`` keys that a
    real run with tracking enabled produces. It defaults to false only because
    that was the original behaviour; leaving it the *only* behaviour is what let
    ``evaluate`` ship broken against the repository's own artifacts.
    """
    import json

    (tmp_path / "data" / "raw" / "panaf500" / "annotations").mkdir(parents=True)
    (tmp_path / "artifacts" / "detections").mkdir(parents=True)

    manifest = tmp_path / "data" / "manifest.csv"
    digest = "a" * 64  # the loader requires a well-formed sha256, not a real one
    manifest.write_text(
        ",".join(MANIFEST_COLUMNS)
        + "\n"
        + f"clip-a,test,clip-a.mp4,clip-a.json,chimpanzee,,unit test,{digest},{digest},\n",
        encoding="utf-8",
    )

    (tmp_path / "data" / "raw" / "panaf500" / "annotations" / "clip-a.json").write_text(
        json.dumps(
            {
                "video": "clip-a",
                "annotations": [
                    {
                        "frame_id": index + 1,
                        "detections": [
                            {
                                "bbox": BOX,
                                "ape_id": 0,
                                "species": "chimpanzee",
                                "behaviour": "walking",
                            }
                        ],
                    }
                    for index in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )

    per_frame_scores = scores if scores is not None else [0.9, 0.9, 0.9]
    (tmp_path / "artifacts" / "detections" / "clip-a.json").write_text(
        json.dumps(
            {
                "clip_id": "clip-a",
                "model": {"name": "MegaDetectorV6", "confidence_threshold": confidence},
                "tracking": {
                    "enabled": stored_track_ids,
                    "backend": "bytetrack" if stored_track_ids else None,
                    "minimum_track_length": 1,
                },
                "video": {"width": 720, "height": 404, "fps": 24.0, "frame_count": 3},
                "frames": [
                    {
                        "clip_id": "clip-a",
                        "frame_index": index,
                        "frame_width": 720,
                        "frame_height": 404,
                        "detections": [
                            {
                                "box": {
                                    "x_min": BOX[0],
                                    "y_min": BOX[1],
                                    "x_max": BOX[2],
                                    "y_max": BOX[3],
                                },
                                "confidence": score,
                                "category_id": 0,
                                "category_name": "animal",
                                **(
                                    {"track_id": 1, "behavior_label": None}
                                    if stored_track_ids
                                    else {}
                                ),
                            }
                        ],
                    }
                    for index, score in enumerate(per_frame_scores)
                ],
            }
        ),
        encoding="utf-8",
    )

    config_data["data"]["manifest_path"] = "data/manifest.csv"
    config_data["model"]["confidence_threshold"] = confidence
    if tracking:
        config_data["tracking"] = {
            "enabled": True,
            "backend": "bytetrack",
            "minimum_track_length": 1,
        }
    return write_config(config_data)


def test_evaluate_reads_saved_detections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_data: dict[str, Any],
    write_config: WriteConfig,
):
    config = _saved_run(tmp_path, config_data, write_config)
    monkeypatch.setenv("PANAF_REPO_ROOT", str(tmp_path))

    result = invoke("evaluate", "--config", str(config))

    assert result.exit_code == 0, result.output
    output = unwrap(result.output)
    assert "clip-a" in output
    # Three frames, one true positive each, nothing else.
    assert "precision 1.0" in output
    assert "recall 1.0" in output


def test_evaluate_reads_detections_saved_with_tracking_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_data: dict[str, Any],
    write_config: WriteConfig,
):
    """A tracked detections document must not crash the detection evaluator.

    Regression: `Detection` is ``extra="forbid"`` and records written with
    tracking on carry ``track_id`` and ``behavior_label``, so ``Detection(**d)``
    raised ``ValidationError`` for every clip. Every fixture until now was
    written with tracking off, so the whole suite passed while the command was
    unusable against the repository's own default artifacts.
    """
    config = _saved_run(tmp_path, config_data, write_config, stored_track_ids=True)
    monkeypatch.setenv("PANAF_REPO_ROOT", str(tmp_path))

    result = invoke("evaluate", "--config", str(config))

    assert result.exit_code == 0, result.output
    output = unwrap(result.output)
    assert "precision 1.0" in output
    assert "recall 1.0" in output


def test_evaluate_confidence_flag_re_scores_upward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_data: dict[str, Any],
    write_config: WriteConfig,
):
    """Raising the threshold must actually discard the weaker detections."""
    config = _saved_run(tmp_path, config_data, write_config, scores=[0.9, 0.3, 0.3])
    monkeypatch.setenv("PANAF_REPO_ROOT", str(tmp_path))

    result = invoke("evaluate", "--config", str(config), "--confidence", "0.5")

    assert result.exit_code == 0, result.output
    output = unwrap(result.output)
    assert "re-scored at 0.50" in output
    # One of three annotated frames still has a surviving detection.
    assert "recall 0.3333" in output


def test_evaluate_refuses_a_threshold_below_the_saved_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_data: dict[str, Any],
    write_config: WriteConfig,
):
    """Lowering it would silently measure detections that were never saved."""
    config = _saved_run(tmp_path, config_data, write_config, confidence=0.2)
    monkeypatch.setenv("PANAF_REPO_ROOT", str(tmp_path))

    result = invoke("evaluate", "--config", str(config), "--confidence", "0.05")

    assert result.exit_code == 1
    assert "below the 0.20" in unwrap(result.output)


def test_evaluate_reports_when_nothing_has_been_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_data: dict[str, Any],
    write_config: WriteConfig,
):
    config = _saved_run(tmp_path, config_data, write_config)
    (tmp_path / "artifacts" / "detections" / "clip-a.json").unlink()
    monkeypatch.setenv("PANAF_REPO_ROOT", str(tmp_path))

    result = invoke("evaluate", "--config", str(config))

    assert result.exit_code == 1
    assert "No saved detections" in unwrap(result.output)


def test_track_refuses_to_run_when_tracking_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_data: dict[str, Any],
    write_config: WriteConfig,
):
    config = _saved_run(tmp_path, config_data, write_config, tracking=False)
    monkeypatch.setenv("PANAF_REPO_ROOT", str(tmp_path))

    result = invoke("track", "--config", str(config))

    assert result.exit_code == 1
    assert "tracking.enabled is false" in unwrap(result.output)


def test_track_measures_saved_detections_and_writes_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_data: dict[str, Any],
    write_config: WriteConfig,
):
    """The one stationary ape should end up on a single track, with no switches."""
    pytest.importorskip("supervision", reason="requires the inference extra")
    import json

    config = _saved_run(tmp_path, config_data, write_config, tracking=True)
    monkeypatch.setenv("PANAF_REPO_ROOT", str(tmp_path))

    result = invoke("track", "--config", str(config))

    assert result.exit_code == 0, result.output
    # Track metrics live in their own directory: a suffix in metrics/ meant every
    # reader had to filter, and the notebook did not.
    written = tmp_path / "artifacts" / "metrics" / "tracking" / "clip-a.json"
    metrics = json.loads(written.read_text(encoding="utf-8"))
    assert metrics["clip_id"] == "clip-a"
    assert metrics["annotated_individuals"] == 1
    assert metrics["total_id_switches"] == 0
    assert metrics["minimum_track_length"] == 1
    assert metrics["schema"] == TRACKING_METRICS_SCHEMA
    # Nothing but detection metrics in metrics/ itself.
    assert list((tmp_path / "artifacts" / "metrics").glob("*.json")) == []


def test_track_can_read_and_write_outside_the_configured_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_data: dict[str, Any],
    write_config: WriteConfig,
):
    """An experiment must not overwrite the baseline it is being compared against."""
    pytest.importorskip("supervision", reason="requires the inference extra")
    import json
    import shutil

    config = _saved_run(tmp_path, config_data, write_config, tracking=True)
    monkeypatch.setenv("PANAF_REPO_ROOT", str(tmp_path))

    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(tmp_path / "artifacts" / "detections", elsewhere / "detections")
    output = tmp_path / "experiment"

    result = invoke(
        "track",
        "--config",
        str(config),
        "--detections-dir",
        str(elsewhere / "detections"),
        "--metrics-dir",
        str(output),
    )

    assert result.exit_code == 0, result.output
    assert (output / "metrics" / "tracking" / "clip-a.json").is_file()
    # The configured artifacts tree is untouched.
    assert not (tmp_path / "artifacts" / "metrics").exists()

    written = json.loads((output / "metrics" / "tracking" / "clip-a.json").read_text())
    # Every setting is recorded beside the result, so a file is self-describing.
    assert written["activation_threshold"] == config_data["model"]["confidence_threshold"]
    assert written["lost_track_buffer"] == 30


def test_track_sweep_ranks_arms_and_writes_one_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_data: dict[str, Any],
    write_config: WriteConfig,
):
    pytest.importorskip("supervision", reason="requires the inference extra")
    import json

    from panaf_ape_detection.pipeline.retrack import TRACKING_SWEEP_SCHEMA

    config = _saved_run(tmp_path, config_data, write_config, tracking=True)
    monkeypatch.setenv("PANAF_REPO_ROOT", str(tmp_path))

    grid = tmp_path / "grid.yaml"
    grid.write_text(
        "name: unit-sweep\naxes:\n  lost_track_buffer: [30, 60]\n  minimum_track_length: [1]\n",
        encoding="utf-8",
    )

    result = invoke("track-sweep", "--grid", str(grid), "--config", str(config))

    assert result.exit_code == 0, result.output
    written = tmp_path / "artifacts" / "metrics" / "tracking-sweep" / "unit-sweep.json"
    record = json.loads(written.read_text(encoding="utf-8"))

    # Its own directory and its own schema: neither detection metrics nor
    # per-clip track metrics, so it must not share a directory with either.
    assert record["schema"] == TRACKING_SWEEP_SCHEMA
    assert len(record["arms"]) == 2
    assert record["clips"] == ["clip-a"]
    assert all("pooled" in arm and "settings" in arm for arm in record["arms"])
    # Ranked best-first by the metric the command says it ranks by.
    scores = [arm["pooled"]["identity_coverage"] for arm in record["arms"]]
    assert scores == sorted(scores, reverse=True)


def test_track_sweep_rejects_an_unknown_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_data: dict[str, Any],
    write_config: WriteConfig,
):
    """A typo would otherwise sweep one arm and read as "no setting helps"."""
    config = _saved_run(tmp_path, config_data, write_config, tracking=True)
    monkeypatch.setenv("PANAF_REPO_ROOT", str(tmp_path))

    grid = tmp_path / "grid.yaml"
    grid.write_text("name: typo\naxes:\n  lost_frame_buffer: [30]\n", encoding="utf-8")

    result = invoke("track-sweep", "--grid", str(grid), "--config", str(config))

    assert result.exit_code == 1
    assert "unknown tracker setting" in unwrap(result.output)
