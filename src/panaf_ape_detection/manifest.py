"""Schema for the clip-selection manifest.

The manifest is how clip selection stops being a decision buried in a shell
history: which clips, **why** each was chosen, and the checksums that prove the
files have not changed underneath a result.

This module defines the **schema only** — the column list and a typed row.
Loading, checksum verification against files on disk, and clip-count limits are
Phase 1b and belong in a ``data`` package alongside frame extraction.

The column order here is the single source of truth. ``data/README.md`` documents
it for humans, ``data/sample_manifest.example.csv`` is the copyable template, and
``scripts/verify_repository.py`` checks the template against
:data:`MANIFEST_COLUMNS` — so the three cannot drift apart.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["MANIFEST_COLUMNS", "ManifestRow"]

MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "clip_id",
    "split",
    "video_filename",
    "annotation_filename",
    "species",
    "site",
    "selected_reason",
    "video_sha256",
    "annotation_sha256",
    "notes",
)
"""Canonical manifest column order.

Changing this is a breaking change to every existing manifest. Update
``data/sample_manifest.example.csv`` and ``data/README.md`` in the same commit.
"""

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ManifestRow(BaseModel):
    """One selected clip.

    Attributes:
        clip_id: Stable identifier, used in output filenames and log entries.
        split: Dataset split, as the deposit defines it.
        video_filename: Filename only, relative to the videos directory. Not an
            absolute path — the manifest must not encode one machine's layout.
        annotation_filename: Corresponding annotation file, when one exists.
        species: Species **as recorded by the dataset**, never as guessed by the
            detector. MegaDetector cannot supply this.
        site: Field site as recorded by the dataset.
        selected_reason: Why this clip was chosen. The column that makes the
            sample purposive rather than arbitrary — "night-time IR, tests
            low-light failure" is useful; "" is not.
        video_sha256: SHA-256 of the video file.
        annotation_sha256: SHA-256 of the annotation file, when present.
        notes: Anything else worth knowing — unusual framing, corrupt frames,
            ambiguity in the labels.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    clip_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    video_filename: str = Field(min_length=1)
    annotation_filename: str | None = None
    species: str | None = None
    site: str | None = None
    selected_reason: str = Field(min_length=1)
    video_sha256: str = Field(pattern=_SHA256_PATTERN)
    annotation_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    notes: str | None = None

    def model_post_init(self, _context: object, /) -> None:
        """Reject filenames that are really paths.

        A manifest carrying ``../raw/videos/clip.mp4`` ties the record to one
        machine's directory layout and defeats the point of resolving paths from
        configuration.
        """
        for field_name in ("video_filename", "annotation_filename"):
            value = getattr(self, field_name)
            if value and ("/" in value or "\\" in value):
                msg = (
                    f"{field_name} must be a filename, not a path (got {value!r}). "
                    "The directory comes from paths.raw_data_dir in configuration."
                )
                raise ValueError(msg)
