"""Tests for run-metadata production.

No GPU, no network, no ``inference`` extra. Git-dependent assertions tolerate
running outside a checkout.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from panaf_ape_detection.config import load_config
from panaf_ape_detection.provenance import (
    build_run_metadata,
    default_metadata_path,
    dependency_versions,
    environment_summary,
    file_sha256,
    git_state,
    input_file_record,
    write_run_metadata,
)
from panaf_ape_detection.types import DeviceKind, RunMetadata


@pytest.fixture
def config(valid_config_file: Path):
    return load_config(valid_config_file, use_env_overrides=False)


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video, but bytes are bytes")
    return path


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #


def test_sha256_matches_hashlib(sample_file: Path):
    expected = hashlib.sha256(sample_file.read_bytes()).hexdigest()

    assert file_sha256(sample_file) == expected


def test_sha256_is_chunk_size_independent(sample_file: Path):
    assert file_sha256(sample_file, chunk_size=1) == file_sha256(sample_file, chunk_size=1 << 20)


def test_sha256_of_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.bin"
    empty.touch()

    assert file_sha256(empty) == hashlib.sha256(b"").hexdigest()


def test_sha256_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "absent.mp4")


def test_input_record_stores_filename_not_path(sample_file: Path):
    """Run metadata must not leak local directory structure."""
    record = input_file_record(sample_file)

    assert record.filename == "clip.mp4"
    assert str(sample_file.parent) not in record.filename
    assert record.size_bytes == sample_file.stat().st_size
    assert len(record.sha256) == 64


# --------------------------------------------------------------------------- #
# Git and dependencies
# --------------------------------------------------------------------------- #


def test_git_state_returns_pair():
    commit, dirty = git_state()

    if commit is None:
        assert dirty is None
    else:
        assert len(commit) == 40
        assert isinstance(dirty, bool)


def test_git_state_outside_a_checkout(tmp_path: Path):
    assert git_state(tmp_path) == (None, None)


def test_dependency_versions_includes_this_package():
    versions = dependency_versions()

    assert "panaf-ape-detection" in versions


def test_dependency_versions_omits_absent_packages():
    versions = dependency_versions(("panaf-ape-detection", "definitely-not-installed-xyz"))

    assert "definitely-not-installed-xyz" not in versions


def test_environment_summary_has_expected_keys():
    summary = environment_summary()

    assert {"python", "platform", "git_commit", "git_dirty"} <= set(summary)


# --------------------------------------------------------------------------- #
# Run metadata
# --------------------------------------------------------------------------- #


def test_build_run_metadata_captures_the_contract(config, sample_file: Path):
    metadata = build_run_metadata(
        config,
        device=DeviceKind.CPU,
        inputs=[sample_file],
        outputs=[Path("artifacts/videos/out.mp4")],
        elapsed_seconds=1.5,
    )

    assert metadata.experiment_name == config.project.experiment_name
    assert metadata.model_variant == config.model.variant
    assert metadata.confidence_threshold == config.model.confidence_threshold
    assert metadata.seed == config.project.seed
    assert metadata.device is DeviceKind.CPU
    assert metadata.elapsed_seconds == 1.5
    assert len(metadata.inputs) == 1
    assert metadata.inputs[0].filename == "clip.mp4"
    assert metadata.config_snapshot["model"]["variant"] == config.model.variant


def test_build_run_metadata_timestamp_is_timezone_aware(config):
    metadata = build_run_metadata(config, device=DeviceKind.CPU)

    assert metadata.started_at_utc.tzinfo is not None


def test_naive_timestamp_is_rejected(config):
    with pytest.raises(ValueError, match="timezone-aware"):
        build_run_metadata(config, device=DeviceKind.CPU, started_at=datetime(2026, 1, 1))  # noqa: DTZ001


def test_unresolved_device_is_rejected(config):
    """`auto` in metadata would misrepresent what actually ran."""
    with pytest.raises(ValueError, match="resolve_device"):
        build_run_metadata(config, device=DeviceKind.AUTO)


def test_run_metadata_round_trips_through_json(config, tmp_path: Path):
    metadata = build_run_metadata(config, device=DeviceKind.CPU)
    written = write_run_metadata(metadata, tmp_path / "nested" / "run.json")

    assert written.is_file()
    restored = RunMetadata.model_validate(json.loads(written.read_text(encoding="utf-8")))
    assert restored.experiment_name == metadata.experiment_name
    assert restored.model_variant == metadata.model_variant


def test_write_creates_parent_directories(config, tmp_path: Path):
    target = tmp_path / "a" / "b" / "c" / "run.json"

    write_run_metadata(build_run_metadata(config, device=DeviceKind.CPU), target)

    assert target.is_file()


def test_default_metadata_path_is_under_configured_artifacts(config):
    path = default_metadata_path(config, started_at=datetime(2026, 7, 24, 12, 0, tzinfo=UTC))

    assert path.parent == config.repository_paths().artifacts_dir / "metadata"
    assert path.name.startswith(config.project.experiment_name)
    assert path.name.endswith("_20260724T120000Z.json")
