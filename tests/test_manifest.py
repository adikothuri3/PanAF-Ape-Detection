"""Tests for the clip-manifest schema.

Schema only — loading and checksum verification against real files are Phase 1b.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from panaf_ape_detection.manifest import MANIFEST_COLUMNS, ManifestRow
from panaf_ape_detection.paths import repository_root

SHA_A = "a" * 64
SHA_B = "b" * 64


def row(**overrides: Any) -> ManifestRow:
    values: dict[str, Any] = {
        "clip_id": "clip-001",
        "split": "train",
        "video_filename": "clip-001.mp4",
        "annotation_filename": "clip-001.json",
        "species": "chimpanzee",
        "site": "site-a",
        "selected_reason": "night-time infrared, tests low-light failure",
        "video_sha256": SHA_A,
        "annotation_sha256": SHA_B,
        "notes": "",
    }
    values.update(overrides)
    return ManifestRow(**values)


# --------------------------------------------------------------------------- #
# The template and the package must agree -- this is the whole point of
# MANIFEST_COLUMNS existing.
# --------------------------------------------------------------------------- #


def test_example_csv_header_matches_the_canonical_columns():
    path = repository_root() / "data" / "sample_manifest.example.csv"

    with path.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))

    assert tuple(header) == MANIFEST_COLUMNS


def test_data_readme_documents_every_column():
    readme = (repository_root() / "data" / "README.md").read_text(encoding="utf-8")

    for column in MANIFEST_COLUMNS:
        assert column in readme, f"{column} is not documented in data/README.md"


def test_every_model_field_is_a_manifest_column():
    assert set(ManifestRow.model_fields) == set(MANIFEST_COLUMNS)


# --------------------------------------------------------------------------- #
# Row validation
# --------------------------------------------------------------------------- #


def test_valid_row():
    parsed = row()

    assert parsed.clip_id == "clip-001"
    assert parsed.video_sha256 == SHA_A


def test_rows_are_frozen():
    with pytest.raises(ValidationError, match="frozen"):
        row().clip_id = "other"  # type: ignore[misc]


def test_unknown_column_is_rejected():
    with pytest.raises(ValidationError, match=r"extra_column|Extra inputs"):
        ManifestRow(**{**row().model_dump(), "extra_column": "x"})


@pytest.mark.parametrize("field", ["clip_id", "split", "video_filename"])
def test_required_text_fields_reject_empty(field: str):
    with pytest.raises(ValidationError, match=field):
        row(**{field: ""})


def test_selected_reason_is_required():
    """The column that makes the sample purposive rather than arbitrary."""
    with pytest.raises(ValidationError, match="selected_reason"):
        row(selected_reason="")


@pytest.mark.parametrize("digest", ["", "not-a-digest", "A" * 64, "a" * 63])
def test_bad_video_checksum_is_rejected(digest: str):
    with pytest.raises(ValidationError, match="video_sha256"):
        row(video_sha256=digest)


def test_annotation_checksum_may_be_absent():
    assert row(annotation_filename=None, annotation_sha256=None).annotation_sha256 is None


@pytest.mark.parametrize("value", ["../raw/videos/clip.mp4", "videos/clip.mp4", "sub\\clip.mp4"])
def test_filenames_must_not_be_paths(value: str):
    """A path in the manifest ties the record to one machine's layout."""
    with pytest.raises(ValidationError, match="must be a filename, not a path"):
        row(video_filename=value)


def test_annotation_filename_must_not_be_a_path():
    with pytest.raises(ValidationError, match="must be a filename, not a path"):
        row(annotation_filename="annotations/clip.json")


def test_a_template_row_can_be_written_and_reread(tmp_path: Path):
    """Round-trip through CSV in the canonical column order."""
    target = tmp_path / "manifest.csv"
    original = row()

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                key: (value if value is not None else "")
                for key, value in original.model_dump().items()
            }
        )

    with target.open(encoding="utf-8") as handle:
        record = next(iter(csv.DictReader(handle)))

    restored = ManifestRow(
        **{
            key: (value or None)
            if key not in {"clip_id", "split", "video_filename", "selected_reason", "video_sha256"}
            else value
            for key, value in record.items()
        }
    )

    assert restored.clip_id == original.clip_id
    assert restored.video_sha256 == original.video_sha256
