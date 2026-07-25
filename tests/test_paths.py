"""Tests for repository-root discovery and the canonical directory layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from panaf_ape_detection.paths import (
    ARTIFACT_SUBDIRECTORIES,
    REPOSITORY_ROOT_ENV_VAR,
    RepositoryPaths,
    repository_root,
    resolve_under_root,
)


def test_repository_root_contains_pyproject():
    root = repository_root()

    assert root.is_absolute()
    assert (root / "pyproject.toml").is_file()


def test_repository_root_is_independent_of_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    expected = repository_root()
    monkeypatch.chdir(tmp_path)

    assert repository_root() == expected


def test_repository_root_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(REPOSITORY_ROOT_ENV_VAR, str(tmp_path))

    assert repository_root() == tmp_path.resolve()


def test_resolve_under_root_makes_relative_paths_absolute():
    resolved = resolve_under_root("data/raw")

    assert resolved == repository_root() / "data" / "raw"
    assert resolved.is_absolute()


def test_resolve_under_root_preserves_absolute_paths(tmp_path: Path):
    assert resolve_under_root(tmp_path / "x") == (tmp_path / "x").resolve()


def test_resolve_under_root_expands_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_under_root("~/clips") == (tmp_path / "clips").resolve()


def test_resolve_under_root_accepts_explicit_root(tmp_path: Path):
    assert resolve_under_root("data/raw", root=tmp_path) == (tmp_path / "data" / "raw").resolve()


def test_repository_paths_layout(tmp_path: Path):
    paths = RepositoryPaths.from_root(tmp_path)

    assert paths.root == tmp_path.resolve()
    assert paths.raw_data_dir == tmp_path.resolve() / "data" / "raw"
    assert paths.interim_data_dir == tmp_path.resolve() / "data" / "interim"
    assert paths.processed_data_dir == tmp_path.resolve() / "data" / "processed"
    assert paths.artifacts_dir == tmp_path.resolve() / "artifacts"
    assert paths.configs_dir == tmp_path.resolve() / "configs"


def test_repository_paths_are_all_absolute():
    for path in RepositoryPaths.from_root().as_mapping().values():
        assert path.is_absolute()


def test_artifact_subdirectories_match_documented_set(tmp_path: Path):
    subdirectories = RepositoryPaths.from_root(tmp_path).artifact_subdirectories()

    assert set(subdirectories) == set(ARTIFACT_SUBDIRECTORIES)
    assert subdirectories["videos"] == tmp_path.resolve() / "artifacts" / "videos"


def test_missing_directories_reports_absent_paths(tmp_path: Path):
    paths = RepositoryPaths.from_root(tmp_path)

    missing = paths.missing_directories()

    # Nothing exists under a fresh tmp_path except the root itself.
    assert paths.root not in missing
    assert paths.raw_data_dir in missing
    # artifacts/ is created on demand and must never be reported as missing.
    assert paths.artifacts_dir not in missing


def test_real_checkout_has_the_expected_directories():
    """The committed repository ships every layout directory except artifacts/."""
    assert RepositoryPaths.from_root().missing_directories() == []


# --------------------------------------------------------------------------- #
# from_config -- configured paths must actually be honoured.
#
# Regression guard: `PathsConfig` was previously validated and then discarded,
# so `paths.artifacts_dir` and `PANAF_ARTIFACTS_DIR` had no effect on anything.
# --------------------------------------------------------------------------- #


class _StubPaths:
    """Minimal stand-in satisfying the `PathsLike` protocol."""

    def __init__(self, base: Path) -> None:
        self.raw_data_dir = base / "elsewhere" / "raw"
        self.interim_data_dir = base / "elsewhere" / "interim"
        self.processed_data_dir = base / "elsewhere" / "processed"
        self.artifacts_dir = base / "somewhere-else" / "artifacts"


def test_from_config_uses_configured_locations(tmp_path: Path):
    stub = _StubPaths(tmp_path)

    paths = RepositoryPaths.from_config(stub, tmp_path)

    assert paths.raw_data_dir == stub.raw_data_dir
    assert paths.interim_data_dir == stub.interim_data_dir
    assert paths.processed_data_dir == stub.processed_data_dir
    assert paths.artifacts_dir == stub.artifacts_dir


def test_from_config_differs_from_the_checkout_default(tmp_path: Path):
    configured = RepositoryPaths.from_config(_StubPaths(tmp_path), tmp_path)
    default = RepositoryPaths.from_root(tmp_path)

    assert configured.artifacts_dir != default.artifacts_dir


def test_from_config_derives_data_dir_from_raw(tmp_path: Path):
    paths = RepositoryPaths.from_config(_StubPaths(tmp_path), tmp_path)

    assert paths.data_dir == (tmp_path / "elsewhere").resolve()


def test_from_config_keeps_checkout_directories_at_defaults(tmp_path: Path):
    paths = RepositoryPaths.from_config(_StubPaths(tmp_path), tmp_path)

    assert paths.configs_dir == tmp_path.resolve() / "configs"
    assert paths.docs_dir == tmp_path.resolve() / "docs"
    assert paths.reports_dir == tmp_path.resolve() / "reports"


def test_artifact_subdirectories_follow_the_configured_root(tmp_path: Path):
    paths = RepositoryPaths.from_config(_StubPaths(tmp_path), tmp_path)

    assert paths.artifact_subdirectories()["metadata"] == (
        tmp_path / "somewhere-else" / "artifacts" / "metadata"
    )


# --------------------------------------------------------------------------- #
# ensure_artifact_dirs
# --------------------------------------------------------------------------- #


def test_ensure_artifact_dirs_creates_every_subdirectory(tmp_path: Path):
    paths = RepositoryPaths.from_root(tmp_path)

    created = paths.ensure_artifact_dirs()

    assert set(created) == set(ARTIFACT_SUBDIRECTORIES)
    for path in created.values():
        assert path.is_dir()


def test_ensure_artifact_dirs_is_idempotent(tmp_path: Path):
    paths = RepositoryPaths.from_root(tmp_path)

    paths.ensure_artifact_dirs()
    paths.ensure_artifact_dirs()  # must not raise

    assert (paths.artifacts_dir / "videos").is_dir()


def test_ensure_artifact_dirs_respects_configured_location(tmp_path: Path):
    paths = RepositoryPaths.from_config(_StubPaths(tmp_path), tmp_path)

    paths.ensure_artifact_dirs()

    assert (tmp_path / "somewhere-else" / "artifacts" / "frames").is_dir()
    assert not (tmp_path / "artifacts").exists()
