"""Repository-root discovery and the canonical directory layout.

Every path used by this project is resolved from the *repository root* rather
than from the process working directory. This keeps behaviour identical whether
a command is run from the repository root, from a subdirectory, from a notebook
or from a Colab checkout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "ARTIFACT_SUBDIRECTORIES",
    "REPOSITORY_ROOT_ENV_VAR",
    "PathsLike",
    "RepositoryPaths",
    "repository_root",
    "resolve_under_root",
]


@runtime_checkable
class PathsLike(Protocol):
    """Structural type for the configured path locations.

    Declared here rather than imported from :mod:`panaf_ape_detection.config` so
    that this module remains a leaf with no intra-project imports. ``Config.paths``
    satisfies it.
    """

    @property
    def raw_data_dir(self) -> Path:
        """Immutable, as-obtained dataset files."""
        ...

    @property
    def interim_data_dir(self) -> Path:
        """Derived intermediates such as extracted frames."""
        ...

    @property
    def processed_data_dir(self) -> Path:
        """Analysis-ready derived data."""
        ...

    @property
    def artifacts_dir(self) -> Path:
        """Generated run outputs."""
        ...


REPOSITORY_ROOT_ENV_VAR = "PANAF_REPO_ROOT"
"""Environment variable that, when set, overrides repository-root discovery."""

_ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml", ".git")

ARTIFACT_SUBDIRECTORIES: tuple[str, ...] = (
    "detections",
    "frames",
    "metadata",
    "metrics",
    "videos",
    "visualizations",
)
"""Subdirectories created underneath the (git-ignored) artifacts directory."""


def _looks_like_repository_root(candidate: Path) -> bool:
    """Return ``True`` when *candidate* contains a repository root marker."""
    return any((candidate / marker).exists() for marker in _ROOT_MARKERS)


@lru_cache(maxsize=1)
def _discover_root() -> Path:
    """Locate the repository root, walking upwards from this source file."""
    override = os.environ.get(REPOSITORY_ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if candidate.is_dir() and _looks_like_repository_root(candidate):
            return candidate

    # Installed as a wheel outside a checkout (for example in Colab): fall back
    # to the working directory so relative config paths still resolve somewhere
    # predictable rather than into site-packages.
    return Path.cwd().resolve()


def repository_root() -> Path:
    """Return the absolute repository root directory.

    Resolution order:

    1. The ``PANAF_REPO_ROOT`` environment variable, when set.
    2. The nearest ancestor directory of this module containing ``pyproject.toml``
       or ``.git``.
    3. The current working directory, as a last resort.

    Returns:
        The resolved repository root as an absolute :class:`~pathlib.Path`.
    """
    if REPOSITORY_ROOT_ENV_VAR in os.environ:
        # Honour late changes to the environment (tests, notebooks) without
        # requiring callers to clear the cache themselves.
        return Path(os.environ[REPOSITORY_ROOT_ENV_VAR]).expanduser().resolve()
    return _discover_root()


def resolve_under_root(value: str | os.PathLike[str], root: Path | None = None) -> Path:
    """Resolve *value* against the repository root unless it is already absolute.

    Args:
        value: A path from configuration, possibly relative and possibly using
            ``~`` for the user's home directory.
        root: Optional explicit root. Defaults to :func:`repository_root`.

    Returns:
        An absolute, normalised path. The path is *not* required to exist.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = root if root is not None else repository_root()
    return (base / path).resolve()


@dataclass(frozen=True, slots=True)
class RepositoryPaths:
    """The canonical directory layout of the repository.

    Attributes:
        root: Repository root.
        configs_dir: Versioned YAML configuration files.
        data_dir: Root of the (mostly git-ignored) data tree.
        raw_data_dir: Immutable, as-downloaded dataset files.
        interim_data_dir: Derived-but-reusable data such as extracted frames.
        processed_data_dir: Analysis-ready derived data.
        artifacts_dir: Generated run outputs; ignored by git in its entirety.
        docs_dir: Project documentation.
        experiments_dir: Research log and experiment notes.
        reports_dir: Write-ups and figures.
    """

    root: Path
    configs_dir: Path
    data_dir: Path
    raw_data_dir: Path
    interim_data_dir: Path
    processed_data_dir: Path
    artifacts_dir: Path
    docs_dir: Path
    experiments_dir: Path
    reports_dir: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> RepositoryPaths:
        """Build the default layout underneath *root*.

        This is the layout of a fresh checkout. It ignores configuration
        entirely — use :meth:`from_config` for the paths a run will actually
        use.

        Args:
            root: Repository root. Defaults to :func:`repository_root`.

        Returns:
            A populated :class:`RepositoryPaths` instance.
        """
        base = (root or repository_root()).resolve()
        data = base / "data"
        return cls(
            root=base,
            configs_dir=base / "configs",
            data_dir=data,
            raw_data_dir=data / "raw",
            interim_data_dir=data / "interim",
            processed_data_dir=data / "processed",
            artifacts_dir=base / "artifacts",
            docs_dir=base / "docs",
            experiments_dir=base / "experiments",
            reports_dir=base / "reports",
        )

    @classmethod
    def from_config(cls, paths_config: PathsLike, root: Path | None = None) -> RepositoryPaths:
        """Build the layout a run will actually use, honouring configuration.

        The four data and artifact locations come from configuration; the rest
        (``configs_dir``, ``docs_dir``, ``experiments_dir``, ``reports_dir``) are
        properties of the checkout and keep their default positions.

        ``data_dir`` is derived from ``raw_data_dir``'s parent, so a config that
        relocates the data tree wholesale stays self-consistent.

        Args:
            paths_config: Any object exposing the four configured path
                attributes — in practice ``Config.paths``. Typed structurally so
                this module stays a leaf and never imports the config layer.
            root: Repository root. Defaults to :func:`repository_root`.

        Returns:
            A populated :class:`RepositoryPaths` instance.
        """
        base = (root or repository_root()).resolve()
        raw = resolve_under_root(paths_config.raw_data_dir, base)
        return cls(
            root=base,
            configs_dir=base / "configs",
            data_dir=raw.parent,
            raw_data_dir=raw,
            interim_data_dir=resolve_under_root(paths_config.interim_data_dir, base),
            processed_data_dir=resolve_under_root(paths_config.processed_data_dir, base),
            artifacts_dir=resolve_under_root(paths_config.artifacts_dir, base),
            docs_dir=base / "docs",
            experiments_dir=base / "experiments",
            reports_dir=base / "reports",
        )

    def ensure_artifact_dirs(self) -> dict[str, Path]:
        """Create the artifacts tree and return the created subdirectories.

        Called explicitly by whichever stage is about to write output. It is
        deliberately **not** called on import or by read-only commands such as
        ``doctor`` and ``show-paths``, which must leave the filesystem untouched.

        Returns:
            The ``{name: path}`` mapping of subdirectories, all now existing.
        """
        subdirectories = self.artifact_subdirectories()
        for path in subdirectories.values():
            path.mkdir(parents=True, exist_ok=True)
        return subdirectories

    def as_mapping(self) -> dict[str, Path]:
        """Return the layout as an ordered ``{name: path}`` mapping."""
        return {
            "root": self.root,
            "configs_dir": self.configs_dir,
            "data_dir": self.data_dir,
            "raw_data_dir": self.raw_data_dir,
            "interim_data_dir": self.interim_data_dir,
            "processed_data_dir": self.processed_data_dir,
            "artifacts_dir": self.artifacts_dir,
            "docs_dir": self.docs_dir,
            "experiments_dir": self.experiments_dir,
            "reports_dir": self.reports_dir,
        }

    def artifact_subdirectories(self) -> dict[str, Path]:
        """Return the expected ``artifacts/`` subdirectories.

        These are *not* created as a side effect of importing or of running
        read-only commands; creation is the responsibility of whichever
        pipeline stage writes into them.
        """
        return {name: self.artifacts_dir / name for name in ARTIFACT_SUBDIRECTORIES}

    def missing_directories(self) -> list[Path]:
        """Return the layout directories that do not currently exist.

        ``artifacts_dir`` is excluded because it is created on demand by
        pipeline runs and is intentionally absent in a fresh checkout.
        """
        return [
            path
            for name, path in self.as_mapping().items()
            if name != "artifacts_dir" and not path.is_dir()
        ]
