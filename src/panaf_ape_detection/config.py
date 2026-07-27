"""Typed YAML configuration for the Phase 1 pipeline.

Design rules enforced here:

* Every configuration section is a strict pydantic model. Unknown keys are a
  hard error rather than a silently ignored typo.
* Relative paths resolve against the repository root, never the shell's working
  directory (see :mod:`panaf_ape_detection.paths`).
* The MegaDetector variant is configuration, not a constant buried in code, so
  that a run's exact weights are recorded rather than implied.
* Secrets never live in YAML. A small, explicit set of environment variables may
  override operational settings such as the device or the artifacts directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from panaf_ape_detection.paths import RepositoryPaths, repository_root, resolve_under_root
from panaf_ape_detection.types import DeviceKind, TrackingBackend

__all__ = [
    "ENV_OVERRIDES",
    "Config",
    "ConfigError",
    "DataConfig",
    "LoggingConfig",
    "ModelConfig",
    "PathsConfig",
    "ProjectConfig",
    "TrackingConfig",
    "VideoConfig",
    "load_config",
]


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing, malformed or invalid."""


ENV_OVERRIDES: Final[dict[str, tuple[str, str]]] = {
    # environment variable -> (section, field)
    "PANAF_DEVICE": ("model", "device"),
    "PANAF_MODEL_VARIANT": ("model", "variant"),
    "PANAF_CONFIDENCE_THRESHOLD": ("model", "confidence_threshold"),
    "PANAF_ARTIFACTS_DIR": ("paths", "artifacts_dir"),
    "PANAF_MAX_CLIPS": ("data", "max_clips"),
    "PANAF_LOG_LEVEL": ("logging", "level"),
    "PANAF_EXPERIMENT_NAME": ("project", "experiment_name"),
}
"""Operational settings that may be overridden from the environment.

Deliberately narrow: these are the knobs that legitimately differ between a
laptop, a Colab GPU runtime and CI. Anything scientifically meaningful should be
changed in a versioned YAML file so that it appears in the diff.
"""

_KNOWN_MODEL_VARIANTS: Final[frozenset[str]] = frozenset(
    {
        # PyTorch-Wildlife `MegaDetectorV6` (YOLOv9 / YOLOv10 / RT-DETR weights)
        "MDV6-yolov9-c",
        "MDV6-yolov9-e",
        "MDV6-yolov10-c",
        "MDV6-yolov10-e",
        "MDV6-rtdetr-c",
        # PyTorch-Wildlife `MegaDetectorV6Apache` (Apache-licensed RT-DETR weights)
        "MDV6-apa-rtdetr-c",
        "MDV6-apa-rtdetr-e",
    }
)
"""Variant strings accepted by PyTorch-Wildlife at the time of writing.

Used for a *warning-free* soft check only: an unrecognised variant is accepted
so that the repository does not break when upstream adds new weights, but the
recognised set is documented in ``docs/obsidian/05 Technical/model.md``.
"""

_VALID_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


class _Section(BaseModel):
    """Base class for configuration sections: strict, immutable after load."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ProjectConfig(_Section):
    """Identity and determinism settings for the project.

    Attributes:
        name: Project name; used in logs and run metadata.
        seed: Random seed applied to any stochastic component.
        experiment_name: Short slug identifying this experiment's outputs.
    """

    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    experiment_name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PathsConfig(_Section):
    """Filesystem locations, resolved relative to the repository root.

    Attributes:
        raw_data_dir: Immutable, as-obtained dataset files.
        interim_data_dir: Derived intermediates such as extracted frames.
        processed_data_dir: Analysis-ready derived data.
        artifacts_dir: Generated run outputs; git-ignored.
    """

    raw_data_dir: Path
    interim_data_dir: Path
    processed_data_dir: Path
    artifacts_dir: Path

    @field_validator("*", mode="after")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        """Resolve each path against the repository root."""
        return resolve_under_root(value)


class DataConfig(_Section):
    """Clip selection and sampling settings.

    Attributes:
        manifest_path: CSV manifest describing the selected clips.
        max_clips: Upper bound on clips processed in one run. Phase 1 targets a
            deliberately small sample (roughly 5-10 clips).
        frame_stride: Process every ``frame_stride``-th decoded frame.
    """

    manifest_path: Path
    max_clips: int = Field(ge=1, le=500)
    frame_stride: int = Field(ge=1)

    @field_validator("manifest_path", mode="after")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        """Resolve the manifest path against the repository root."""
        return resolve_under_root(value)


class ModelConfig(_Section):
    """Detector selection.

    The variant is required and never defaulted inside application code: a run
    is only reproducible if the exact weights are recorded.

    Attributes:
        framework: Library that loads the detector, e.g. ``pytorch-wildlife``.
        model_name: Detector family, e.g. ``MegaDetectorV6``.
        variant: Exact weight identifier, e.g. ``MDV6-yolov9-c``.
        confidence_threshold: Minimum score for a detection to be kept.
        device: Requested compute device; ``auto`` resolves at runtime.
    """

    framework: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    device: DeviceKind

    @property
    def variant_is_recognised(self) -> bool:
        """Whether :attr:`variant` is in the documented PyTorch-Wildlife set."""
        return self.variant in _KNOWN_MODEL_VARIANTS


class TrackingConfig(_Section):
    """Multi-object tracking settings.

    ``bytetrack`` is implemented; ``sort`` is a valid value with no
    implementation behind it, and the runner raises rather than silently
    running untracked.

    The four tuning fields default to the values the pipeline used when they
    were hardcoded, so omitting them reproduces the previous behaviour exactly.
    Before they existed, three of ByteTrack's five real knobs were unreachable
    from configuration and the fourth was welded to the detector's threshold,
    which made a tracking sweep impossible without editing code.

    Attributes:
        enabled: Whether the tracking stage should run.
        backend: Tracker implementation to use.
        minimum_track_length: Tracks shorter than this are discarded, after the
            whole clip has been tracked.
        activation_threshold: Score at which a detection may participate fully
            in association. ``None`` follows ``model.confidence_threshold``,
            which is what the pipeline did unconditionally before.
        lost_track_buffer: Frames a track survives unmatched, expressed at
            30 fps; ByteTrack rescales it by the clip's actual rate.
        minimum_matching_threshold: IoU-distance threshold for association.
        minimum_consecutive_frames: Frames a new track must be seen in before it
            is reported. A cheaper, online alternative to
            ``minimum_track_length``, which can only be applied afterwards.
        score_floor: Rescale detection scores into ``[score_floor, 1]`` before
            association, to clear ByteTrack's hardcoded 0.1 discard floor. The
            original scores are restored before anything is written. ``None``
            disables it. See
            :class:`~panaf_ape_detection.tracking.bytetrack.ScoreFloor`.
        stitch_max_gap: Frames a gap may span for two track fragments to be
            joined as one animal. ``0`` disables stitching.
        stitch_max_distance: Positional tolerance for that join, in box
            diagonals.
        interpolate_max_gap: Frames of an interior gap to fill with synthesised
            boxes, each marked ``interpolated``. ``0`` disables it. The only
            setting here that can raise coverage above detection recall.
        smooth_window: Odd window of frames to average each track's boxes over.
            ``1`` disables smoothing. Affects only where boxes sit, never which
            frames or identities they have.
    """

    enabled: bool
    backend: TrackingBackend
    minimum_track_length: int = Field(ge=1)
    activation_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    lost_track_buffer: int = Field(default=30, ge=1)
    minimum_matching_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    minimum_consecutive_frames: int = Field(default=1, ge=1)
    score_floor: float | None = Field(default=None, ge=0.0, lt=1.0)
    stitch_max_gap: int = Field(default=0, ge=0)
    stitch_max_distance: float = Field(default=1.0, gt=0.0)
    interpolate_max_gap: int = Field(default=0, ge=0)
    smooth_window: int = Field(default=1, ge=1)

    def model_post_init(self, _context: object, /) -> None:
        """Reject the contradictory ``enabled: true`` + ``backend: none`` pair."""
        if self.enabled and self.backend is TrackingBackend.NONE:
            msg = (
                "tracking.enabled is true but tracking.backend is 'none'. "
                "Choose a backend (e.g. 'bytetrack') or set tracking.enabled to false."
            )
            raise ValueError(msg)

    @field_validator("smooth_window", mode="after")
    @classmethod
    def _window_must_have_a_centre(cls, value: int) -> int:
        """Reject an even smoothing window.

        An even window has no centre frame, so it would shift every box half a
        frame forward in time -- a lag that looks like a tracking error.
        """
        if value % 2 == 0:
            msg = f"tracking.smooth_window must be odd so it has a centre frame, got {value}"
            raise ValueError(msg)
        return value

    def resolved_activation_threshold(self, confidence_threshold: float) -> float:
        """Return the activation threshold to hand the tracker.

        Args:
            confidence_threshold: ``model.confidence_threshold``, used when no
                activation threshold is configured.

        Returns:
            The configured value, or the detector's threshold when unset.
        """
        if self.activation_threshold is None:
            return confidence_threshold
        return self.activation_threshold


class VideoConfig(_Section):
    """Annotated-video rendering settings.

    Attributes:
        output_fps: Frame rate of the rendered clip.
        codec: FourCC / container codec hint passed to the video writer.
        draw_confidence: Render detector scores on each box.
        draw_track_id: Render tracker identities on each box.
        draw_behavior_label: Render the dataset-provided behaviour label.
        write_annotated: Whether to encode an annotated MP4 per clip. Defaults to
            true, which is the behaviour that existed before this field. Turn it
            off for a run whose product is metrics: encoding 500 clips costs
            gigabytes and a second decode pass each, for video nobody watches.
    """

    output_fps: float = Field(gt=0.0, le=240.0)
    codec: str = Field(min_length=1, max_length=8)
    draw_confidence: bool
    draw_track_id: bool
    draw_behavior_label: bool
    write_annotated: bool = True


class LoggingConfig(_Section):
    """Logging behaviour.

    Attributes:
        level: Standard library logging level name.
        save_run_metadata: Whether a run-metadata record should be written
            alongside generated artefacts.
    """

    level: str
    save_run_metadata: bool

    @field_validator("level", mode="before")
    @classmethod
    def _normalise_level(cls, value: object) -> object:
        """Upper-case string levels so ``info`` and ``INFO`` both work."""
        return value.upper() if isinstance(value, str) else value

    @field_validator("level", mode="after")
    @classmethod
    def _check_level(cls, value: str) -> str:
        """Reject level names the standard library would not understand."""
        if value not in _VALID_LOG_LEVELS:
            valid = ", ".join(sorted(_VALID_LOG_LEVELS))
            msg = f"unknown logging level {value!r}; expected one of: {valid}"
            raise ValueError(msg)
        return value


class Config(_Section):
    """The complete, validated project configuration.

    Attributes:
        project: Identity and seed.
        paths: Filesystem layout.
        data: Clip selection and sampling.
        model: Detector selection.
        tracking: Tracking stage settings.
        video: Rendering settings.
        logging: Logging settings.
    """

    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    tracking: TrackingConfig
    video: VideoConfig
    logging: LoggingConfig

    def repository_paths(self) -> RepositoryPaths:
        """Return the layout this configuration actually selects.

        Honours the ``paths`` section, so a relocated ``artifacts_dir`` is
        reflected here rather than silently ignored in favour of the checkout
        default.
        """
        return RepositoryPaths.from_config(self.paths, repository_root())

    def describe(self) -> dict[str, str]:
        """Return a flat, human-readable summary for CLI display."""
        return {
            "project.name": self.project.name,
            "project.experiment_name": self.project.experiment_name,
            "project.seed": str(self.project.seed),
            "paths.raw_data_dir": str(self.paths.raw_data_dir),
            "paths.interim_data_dir": str(self.paths.interim_data_dir),
            "paths.processed_data_dir": str(self.paths.processed_data_dir),
            "paths.artifacts_dir": str(self.paths.artifacts_dir),
            "data.manifest_path": str(self.data.manifest_path),
            "data.max_clips": str(self.data.max_clips),
            "data.frame_stride": str(self.data.frame_stride),
            "model.framework": self.model.framework,
            "model.model_name": self.model.model_name,
            "model.variant": self.model.variant,
            "model.confidence_threshold": str(self.model.confidence_threshold),
            "model.device": self.model.device.value,
            "tracking.enabled": str(self.tracking.enabled),
            "tracking.backend": self.tracking.backend.value,
            "tracking.minimum_track_length": str(self.tracking.minimum_track_length),
            "tracking.activation_threshold": _describe_activation(self),
            "tracking.lost_track_buffer": str(self.tracking.lost_track_buffer),
            "tracking.minimum_matching_threshold": str(self.tracking.minimum_matching_threshold),
            "tracking.minimum_consecutive_frames": str(self.tracking.minimum_consecutive_frames),
            "tracking.score_floor": str(self.tracking.score_floor),
            "tracking.stitch_max_gap": str(self.tracking.stitch_max_gap),
            "tracking.interpolate_max_gap": str(self.tracking.interpolate_max_gap),
            "tracking.smooth_window": str(self.tracking.smooth_window),
            "video.output_fps": str(self.video.output_fps),
            "logging.level": self.logging.level,
        }


def _describe_activation(config: Config) -> str:
    """Render the tracker's activation threshold, and where it came from.

    Worth the annotation: the value is the detector's confidence threshold
    whenever it is unset, and a reader comparing two configs needs to see that
    it followed rather than that it was chosen.
    """
    resolved = config.tracking.resolved_activation_threshold(config.model.confidence_threshold)
    if config.tracking.activation_threshold is None:
        return f"{resolved} (from model.confidence_threshold)"
    return str(resolved)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read *path* as a YAML mapping.

    Args:
        path: Configuration file location.

    Returns:
        The parsed mapping.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or does not
            contain a top-level mapping.
    """
    if not path.is_file():
        msg = f"configuration file not found: {path}"
        raise ConfigError(msg)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"{path} is not valid YAML: {exc}"
        raise ConfigError(msg) from exc

    if raw is None:
        msg = f"{path} is empty; expected a YAML mapping with the documented sections"
        raise ConfigError(msg)
    if not isinstance(raw, dict):
        msg = f"{path} must contain a YAML mapping at the top level, got {type(raw).__name__}"
        raise ConfigError(msg)
    return raw


def _coerce_override(value: str) -> str | int | float | bool:
    """Coerce an environment-variable string into a plausible scalar."""
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_env_overrides(
    data: dict[str, Any], environ: dict[str, str] | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply the whitelisted environment overrides to a raw config mapping.

    Args:
        data: Raw configuration mapping, as parsed from YAML.
        environ: Environment to read. Defaults to :data:`os.environ`.

    Returns:
        A ``(merged_mapping, applied)`` tuple where ``applied`` maps the
        environment variable name to the dotted config key it overrode.
    """
    env = dict(os.environ if environ is None else environ)
    merged = {key: dict(value) if isinstance(value, dict) else value for key, value in data.items()}
    applied: dict[str, str] = {}

    for env_var, (section, field) in ENV_OVERRIDES.items():
        if env_var not in env:
            continue
        target = merged.get(section)
        if not isinstance(target, dict):
            # The section is absent or malformed; let normal validation report it.
            continue
        target[field] = _coerce_override(env[env_var])
        applied[env_var] = f"{section}.{field}"

    return merged, applied


def _format_validation_error(path: Path, error: ValidationError) -> str:
    """Render a pydantic error into an actionable, file-anchored message."""
    lines = [f"{path} failed validation ({error.error_count()} problem(s)):"]
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"  - {location}: {item['msg']}")
        if item["type"] == "extra_forbidden":
            lines.append(
                f"      unknown key {location!r}; remove it or check it against configs/base.yaml"
            )
    return "\n".join(lines)


def load_config(
    path: str | os.PathLike[str],
    *,
    use_env_overrides: bool = True,
    environ: dict[str, str] | None = None,
) -> Config:
    """Load, override and validate a YAML configuration file.

    Args:
        path: Path to the YAML file. Relative paths resolve against the
            repository root.
        use_env_overrides: Whether to apply :data:`ENV_OVERRIDES`.
        environ: Environment mapping to read overrides from. Defaults to the
            real process environment.

    Returns:
        A validated :class:`Config`.

    Raises:
        ConfigError: If the file cannot be read or fails validation. The message
            names the file and every offending key.
    """
    config_path = resolve_under_root(path)
    raw = _read_yaml(config_path)

    if use_env_overrides:
        raw, _applied = apply_env_overrides(raw, environ)

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(config_path, exc)) from exc
