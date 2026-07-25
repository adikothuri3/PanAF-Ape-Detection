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
