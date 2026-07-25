"""Shared pytest fixtures.

The whole suite must run in an environment installed *without* the ``inference``
extra: no model weights are downloaded, no GPU is required and no network access
is performed.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

VALID_CONFIG: dict[str, Any] = {
    "project": {"name": "panaf-ape-detection", "seed": 42, "experiment_name": "unit-test"},
    "paths": {
        "raw_data_dir": "data/raw",
        "interim_data_dir": "data/interim",
        "processed_data_dir": "data/processed",
        "artifacts_dir": "artifacts",
    },
    "data": {"manifest_path": "data/sample_manifest.csv", "max_clips": 8, "frame_stride": 1},
    "model": {
        "framework": "pytorch-wildlife",
        "model_name": "MegaDetectorV6",
        "variant": "MDV6-yolov9-c",
        "confidence_threshold": 0.2,
        "device": "auto",
    },
    "tracking": {"enabled": False, "backend": "none", "minimum_track_length": 5},
    "video": {
        "output_fps": 24.0,
        "codec": "mp4v",
        "draw_confidence": True,
        "draw_track_id": True,
        "draw_behavior_label": True,
    },
    "logging": {"level": "INFO", "save_run_metadata": True},
}
"""A minimal configuration mapping that mirrors ``configs/base.yaml``."""


@pytest.fixture
def config_data() -> dict[str, Any]:
    """Return a fresh, mutable copy of :data:`VALID_CONFIG`."""
    return copy.deepcopy(VALID_CONFIG)


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[..., Path]:
    """Return a helper that writes a config mapping to a temporary YAML file."""

    def _write(data: dict[str, Any], name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def valid_config_file(write_config: Callable[..., Path], config_data: dict[str, Any]) -> Path:
    """Write :data:`VALID_CONFIG` to a temporary file and return its path."""
    return write_config(config_data)
