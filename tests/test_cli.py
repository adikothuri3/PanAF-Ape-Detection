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

WriteConfig = Callable[..., Path]

runner = CliRunner()


def invoke(*args: str):
    return runner.invoke(app, list(args))


def registered_command_names() -> set[str]:
    """Return the command names Typer will expose, independent of help layout."""
    return {
        command.name or (command.callback.__name__ if command.callback else "")
        for command in app.registered_commands
    }


def test_only_implemented_commands_are_registered():
    """Unimplemented pipeline stages must be absent, not stubbed."""
    assert registered_command_names() == {"doctor", "validate-config", "show-paths"}


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


def test_doctor_states_that_the_pipeline_is_not_implemented():
    result = invoke("doctor")

    assert "not implemented yet" in result.output


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
    assert "docs/model.md" in result.output


@pytest.mark.parametrize("command", ["doctor", "show-paths"])
def test_commands_are_read_only(command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Inspection commands must not create directories as a side effect."""
    monkeypatch.setenv("PANAF_REPO_ROOT", str(tmp_path))

    result = invoke(command)

    assert result.exit_code == 0, result.output
    assert list(tmp_path.iterdir()) == []
