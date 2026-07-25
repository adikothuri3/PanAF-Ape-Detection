"""Tests for typed YAML configuration loading and validation."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from panaf_ape_detection.config import (
    Config,
    ConfigError,
    LoggingConfig,
    apply_env_overrides,
    load_config,
)
from panaf_ape_detection.paths import repository_root
from panaf_ape_detection.types import DeviceKind, TrackingBackend

WriteConfig = Callable[..., Path]

# --------------------------------------------------------------------------- #
# Valid configuration
# --------------------------------------------------------------------------- #


def test_loads_valid_config(valid_config_file: Path):
    config = load_config(valid_config_file, use_env_overrides=False)

    assert isinstance(config, Config)
    assert config.project.seed == 42
    assert config.model.variant == "MDV6-yolov9-c"
    assert config.model.device is DeviceKind.AUTO
    assert config.tracking.backend is TrackingBackend.NONE


@pytest.mark.parametrize("name", ["base.yaml", "colab.yaml"])
def test_shipped_configs_are_valid(name: str):
    """The configs committed to the repository must validate as written."""
    config = load_config(Path("configs") / name, use_env_overrides=False)

    assert config.project.name == "panaf-ape-detection"
    assert config.model.variant_is_recognised


def test_relative_paths_resolve_against_repository_root(valid_config_file: Path):
    config = load_config(valid_config_file, use_env_overrides=False)
    root = repository_root()

    assert config.paths.raw_data_dir == root / "data" / "raw"
    assert config.paths.artifacts_dir == root / "artifacts"
    assert config.data.manifest_path == root / "data" / "sample_manifest.csv"
    assert config.paths.raw_data_dir.is_absolute()


def test_absolute_paths_are_preserved(
    tmp_path: Path, write_config: WriteConfig, config_data: dict[str, Any]
):
    absolute = tmp_path / "elsewhere" / "raw"
    config_data["paths"]["raw_data_dir"] = str(absolute)

    config = load_config(write_config(config_data), use_env_overrides=False)

    assert config.paths.raw_data_dir == absolute.resolve()


def test_log_level_is_case_insensitive(write_config: WriteConfig, config_data: dict[str, Any]):
    config_data["logging"]["level"] = "debug"

    config = load_config(write_config(config_data), use_env_overrides=False)

    assert config.logging.level == "DEBUG"


def test_describe_returns_flat_summary(valid_config_file: Path):
    summary = load_config(valid_config_file, use_env_overrides=False).describe()

    assert summary["model.variant"] == "MDV6-yolov9-c"
    assert all(isinstance(value, str) for value in summary.values())


def test_config_sections_are_frozen(valid_config_file: Path):
    config = load_config(valid_config_file, use_env_overrides=False)

    with pytest.raises(ValueError, match="frozen"):
        config.model.confidence_threshold = 0.9  # type: ignore[misc]


def test_repository_paths_are_reachable_from_config(valid_config_file: Path):
    config = load_config(valid_config_file, use_env_overrides=False)

    assert config.repository_paths().root == repository_root()


# --------------------------------------------------------------------------- #
# Invalid configuration
# --------------------------------------------------------------------------- #


def test_missing_file_raises_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="configuration file not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_empty_file_raises_config_error(tmp_path: Path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="empty"):
        load_config(path)


def test_non_mapping_file_raises_config_error(tmp_path: Path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="mapping at the top level"):
        load_config(path)


def test_malformed_yaml_raises_config_error(tmp_path: Path):
    path = tmp_path / "broken.yaml"
    path.write_text("project: {name: 'unterminated\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_unknown_key_is_rejected(write_config: WriteConfig, config_data: dict[str, Any]):
    config_data["model"]["confidance_threshold"] = 0.3  # deliberate typo

    with pytest.raises(ConfigError) as excinfo:
        load_config(write_config(config_data), use_env_overrides=False)

    message = str(excinfo.value)
    assert "confidance_threshold" in message
    assert "unknown key" in message


def test_unknown_top_level_section_is_rejected(
    write_config: WriteConfig, config_data: dict[str, Any]
):
    config_data["training"] = {"epochs": 10}

    with pytest.raises(ConfigError, match="training"):
        load_config(write_config(config_data), use_env_overrides=False)


def test_missing_section_is_reported(write_config: WriteConfig, config_data: dict[str, Any]):
    del config_data["model"]

    with pytest.raises(ConfigError, match="model"):
        load_config(write_config(config_data), use_env_overrides=False)


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_confidence_threshold_range_is_enforced(
    write_config: WriteConfig, config_data: dict[str, Any], threshold: float
):
    config_data["model"]["confidence_threshold"] = threshold

    with pytest.raises(ConfigError, match="confidence_threshold"):
        load_config(write_config(config_data), use_env_overrides=False)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("data", "frame_stride", 0),
        ("data", "max_clips", 0),
        ("video", "output_fps", 0.0),
        ("tracking", "minimum_track_length", 0),
        ("project", "seed", -1),
    ],
)
def test_positive_numeric_fields_are_enforced(
    write_config: WriteConfig,
    config_data: dict[str, Any],
    section: str,
    field: str,
    value: float,
):
    config_data[section][field] = value

    with pytest.raises(ConfigError, match=field):
        load_config(write_config(config_data), use_env_overrides=False)


def test_unknown_device_is_rejected(write_config: WriteConfig, config_data: dict[str, Any]):
    config_data["model"]["device"] = "tpu"

    with pytest.raises(ConfigError, match="device"):
        load_config(write_config(config_data), use_env_overrides=False)


def test_unknown_log_level_is_rejected(write_config: WriteConfig, config_data: dict[str, Any]):
    config_data["logging"]["level"] = "chatty"

    with pytest.raises(ConfigError, match="unknown logging level"):
        load_config(write_config(config_data), use_env_overrides=False)


def test_tracking_enabled_without_backend_is_rejected(
    write_config: WriteConfig, config_data: dict[str, Any]
):
    config_data["tracking"]["enabled"] = True
    config_data["tracking"]["backend"] = "none"

    with pytest.raises(ConfigError, match="Choose a backend"):
        load_config(write_config(config_data), use_env_overrides=False)


def test_empty_experiment_name_is_rejected(write_config: WriteConfig, config_data: dict[str, Any]):
    config_data["project"]["experiment_name"] = ""

    with pytest.raises(ConfigError, match="experiment_name"):
        load_config(write_config(config_data), use_env_overrides=False)


def test_experiment_name_rejects_path_separators(
    write_config: WriteConfig, config_data: dict[str, Any]
):
    """Experiment names become directory names; separators must not slip in."""
    config_data["project"]["experiment_name"] = "../escape"

    with pytest.raises(ConfigError, match="experiment_name"):
        load_config(write_config(config_data), use_env_overrides=False)


def test_unrecognised_variant_is_accepted_but_flagged(
    write_config: WriteConfig, config_data: dict[str, Any]
):
    config_data["model"]["variant"] = "MDV6-something-new"

    config = load_config(write_config(config_data), use_env_overrides=False)

    assert config.model.variant == "MDV6-something-new"
    assert not config.model.variant_is_recognised


# --------------------------------------------------------------------------- #
# Environment overrides
# --------------------------------------------------------------------------- #


def test_env_override_applies(valid_config_file: Path):
    config = load_config(
        valid_config_file,
        environ={"PANAF_DEVICE": "cpu", "PANAF_CONFIDENCE_THRESHOLD": "0.55"},
    )

    assert config.model.device is DeviceKind.CPU
    assert config.model.confidence_threshold == pytest.approx(0.55)


def test_env_override_can_be_disabled(valid_config_file: Path):
    config = load_config(
        valid_config_file,
        use_env_overrides=False,
        environ={"PANAF_DEVICE": "cpu"},
    )

    assert config.model.device is DeviceKind.AUTO


def test_env_override_does_not_mutate_input_mapping(config_data: dict[str, Any]):
    original = copy.deepcopy(config_data)

    merged, applied = apply_env_overrides(config_data, {"PANAF_DEVICE": "cpu"})

    assert merged["model"]["device"] == "cpu"
    assert config_data == original
    assert applied == {"PANAF_DEVICE": "model.device"}


def test_env_override_coerces_scalars(config_data: dict[str, Any]):
    merged, _ = apply_env_overrides(
        config_data, {"PANAF_MAX_CLIPS": "3", "PANAF_LOG_LEVEL": "debug"}
    )

    assert merged["data"]["max_clips"] == 3
    assert merged["logging"]["level"] == "debug"


def test_env_override_ignores_unknown_variables(config_data: dict[str, Any]):
    _merged, applied = apply_env_overrides(config_data, {"PANAF_NOT_A_SETTING": "x"})

    assert applied == {}


def test_env_override_of_invalid_value_still_reports_clearly(valid_config_file: Path):
    with pytest.raises(ConfigError, match="confidence_threshold"):
        load_config(valid_config_file, environ={"PANAF_CONFIDENCE_THRESHOLD": "7"})


def test_logging_config_can_be_constructed_directly():
    assert LoggingConfig(level="warning", save_run_metadata=False).level == "WARNING"
