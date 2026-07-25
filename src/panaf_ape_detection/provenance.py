"""Producers for the run-metadata record.

``types.RunMetadata`` defines *what* a reproducible run records;
``05 Technical/reproducibility.md`` explains *why* each field is there. This module is
the single place that actually produces one.

Nothing here imports the machine-learning stack, so run metadata can be built
and written in the lightweight environment and unit-tested without a GPU.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from panaf_ape_detection.config import Config
from panaf_ape_detection.types import DeviceKind, InputFileRecord, RunMetadata

__all__ = [
    "TRACKED_DEPENDENCIES",
    "build_run_metadata",
    "default_metadata_path",
    "dependency_versions",
    "environment_summary",
    "file_sha256",
    "git_state",
    "input_file_record",
    "write_run_metadata",
]

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1 << 20  # 1 MiB — video files are large; do not read them whole.

TRACKED_DEPENDENCIES: tuple[str, ...] = (
    "panaf-ape-detection",
    "pytorchwildlife",
    "torch",
    "torchvision",
    "supervision",
    "opencv-python-headless",
    "numpy",
    "pandas",
    "imageio",
)
"""Distributions whose versions are recorded with every run.

Deliberately the ones that can change a detection: the framework, the model
loader, the array library and the codecs. Recording the entire environment would
bury the four versions that actually matter.
"""


def file_sha256(path: Path | str, *, chunk_size: int = _CHUNK_SIZE) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks.

    Args:
        path: File to digest.
        chunk_size: Read size in bytes.

    Returns:
        The 64-character lowercase hex digest, matching the format
        ``data/sample_manifest.example.csv`` expects.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def input_file_record(path: Path | str) -> InputFileRecord:
    """Build a provenance record for one input file.

    Only the file *name* is stored, never the full path — run metadata should
    not leak local directory structure or more of the dataset layout than
    necessary.

    Args:
        path: The input file.

    Returns:
        A populated :class:`~panaf_ape_detection.types.InputFileRecord`.
    """
    resolved = Path(path)
    return InputFileRecord(
        filename=resolved.name,
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _run_git(*args: str, cwd: Path) -> str | None:
    """Run a git command, returning stripped stdout or ``None`` on any failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state(root: Path | None = None) -> tuple[str | None, bool | None]:
    """Return ``(commit, dirty)`` for the checkout containing *root*.

    The dirty flag is the important half: a commit SHA recorded from a modified
    working tree describes code that did not run. Results from a dirty tree are
    provisional until re-run from a clean commit.

    Args:
        root: Directory inside the checkout. Defaults to the repository root.

    Returns:
        ``(commit_sha, is_dirty)``, or ``(None, None)`` when git is unavailable
        or this is not a checkout.
    """
    from panaf_ape_detection.paths import repository_root

    base = root or repository_root()

    if _run_git("rev-parse", "--is-inside-work-tree", cwd=base) != "true":
        return (None, None)

    commit = _run_git("rev-parse", "HEAD", cwd=base)
    if commit is None:
        return (None, None)

    status = _run_git("status", "--porcelain", cwd=base)
    dirty = None if status is None else bool(status)
    return (commit, dirty)


def dependency_versions(names: tuple[str, ...] = TRACKED_DEPENDENCIES) -> dict[str, str]:
    """Return installed versions for *names*, skipping those not installed.

    Args:
        names: Distribution names, as they appear on PyPI.

    Returns:
        A ``{distribution: version}`` mapping. Absent packages are omitted
        rather than recorded as ``"unknown"`` — the lightweight environment
        genuinely has no torch, and saying so by omission is accurate.
    """
    found: dict[str, str] = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def build_run_metadata(
    config: Config,
    *,
    device: DeviceKind,
    inputs: list[Path] | None = None,
    outputs: list[Path] | None = None,
    elapsed_seconds: float | None = None,
    started_at: datetime | None = None,
) -> RunMetadata:
    """Assemble the record describing one pipeline run.

    Args:
        config: The fully resolved configuration the run used.
        device: The device actually selected, from
            :func:`panaf_ape_detection.runtime.resolve_device` — not the
            requested value, which may have been ``auto``.
        inputs: Input files to checksum. Each is hashed, so this is O(bytes).
        outputs: Paths the run produced, recorded relative to the repository
            root where possible.
        elapsed_seconds: Wall-clock duration.
        started_at: Start time. Defaults to now, in UTC. Naive datetimes are
            rejected — a local timestamp is ambiguous across machines.

    Returns:
        A populated :class:`~panaf_ape_detection.types.RunMetadata`.

    Raises:
        ValueError: If *started_at* is naive, or *device* is ``auto``.
    """
    if device is DeviceKind.AUTO:
        msg = (
            "device must be resolved before building run metadata; pass the result of "
            "runtime.resolve_device(), not the configured value"
        )
        raise ValueError(msg)

    if started_at is None:
        started_at = datetime.now(UTC)
    elif started_at.tzinfo is None:
        msg = "started_at must be timezone-aware; use datetime.now(UTC)"
        raise ValueError(msg)

    commit, dirty = git_state()
    root = config.repository_paths().root

    return RunMetadata(
        experiment_name=config.project.experiment_name,
        started_at_utc=started_at,
        git_commit=commit,
        git_dirty=dirty,
        config_snapshot=config.model_dump(mode="json"),
        python_version=platform.python_version(),
        dependency_versions=dependency_versions(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        device=device,
        model_framework=config.model.framework,
        model_name=config.model.model_name,
        model_variant=config.model.variant,
        confidence_threshold=config.model.confidence_threshold,
        seed=config.project.seed,
        inputs=[input_file_record(path) for path in inputs or []],
        output_paths=[_relative_to_root(path, root) for path in outputs or []],
        elapsed_seconds=elapsed_seconds,
    )


def _relative_to_root(path: Path, root: Path) -> Path:
    """Return *path* relative to *root* when it is underneath it."""
    try:
        return Path(path).resolve().relative_to(root)
    except ValueError:
        return Path(path)


def write_run_metadata(metadata: RunMetadata, path: Path) -> Path:
    """Write *metadata* to *path* as indented JSON, creating parent directories.

    Args:
        metadata: The record to persist.
        path: Destination, conventionally under ``artifacts/metadata/``.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("wrote run metadata to %s", path)
    return path


def default_metadata_path(config: Config, *, started_at: datetime | None = None) -> Path:
    """Return the conventional metadata path for a run.

    ``artifacts/metadata/<experiment>_<UTC timestamp>.json``. The experiment name
    is already constrained by config validation to contain no path separators.
    """
    moment = started_at or datetime.now(UTC)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    artifacts = config.repository_paths().artifacts_dir
    return artifacts / "metadata" / f"{config.project.experiment_name}_{stamp}.json"


def environment_summary() -> dict[str, str]:
    """Return a small human-readable environment snapshot for logs."""
    commit, dirty = git_state()
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "git_commit": commit or "unknown",
        "git_dirty": "unknown" if dirty is None else str(dirty).lower(),
        "cwd": os.fspath(Path.cwd()),
    }
