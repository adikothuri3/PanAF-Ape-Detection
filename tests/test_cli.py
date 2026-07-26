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
from panaf_ape_detection.manifest import MANIFEST_COLUMNS

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

    assert "MDV6-yolov9-c" in result.output
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
) -> Path:
    """Lay out a miniature repository: manifest, annotations, saved detections.

    One clip, three frames, one annotated ape holding still, and one detection
    per frame at *scores*.
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
    written = tmp_path / "artifacts" / "metrics" / "clip-a_tracking.json"
    metrics = json.loads(written.read_text(encoding="utf-8"))
    assert metrics["clip_id"] == "clip-a"
    assert metrics["annotated_individuals"] == 1
    assert metrics["total_id_switches"] == 0
    assert metrics["minimum_track_length"] == 1
