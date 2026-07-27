"""Read and aggregate what a run wrote to ``artifacts/``.

**This module exists because the notebook did this inline and got it wrong.**
``artifacts/metrics/`` came to hold two incompatible JSON schemas -- detection
metrics and track metrics -- distinguished only by a filename suffix. A cell that
did ``glob("*.json")`` and read ``m["overall"]`` crashed with ``KeyError`` the
moment tracking was enabled. Nothing caught it, because notebook code is not
linted, type-checked or tested.

Everything the notebook needs to read from disk now lives here, where all three
apply. The rule the module enforces:

* ``metrics/<clip>.json``          -- detection metrics, and nothing else
* ``metrics/tracking/<clip>.json`` -- track metrics

:func:`load_detection_metrics` also skips the **legacy** ``<clip>_tracking.json``
layout, so artifact trees produced before the split still load correctly.

Pure standard library: no pydantic, no torch, no pandas. It must import after a
bare ``uv sync``, and it must work against a Drive copy of ``artifacts/`` with no
repository around it.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DETECTION_METRICS_SCHEMA",
    "TRACKING_METRICS_SCHEMA",
    "TRACK_METRICS_SUBDIR",
    "PooledCounts",
    "PooledTrackCounts",
    "ScoreBands",
    "detection_fields",
    "format_recall_table",
    "latest_run_metadata",
    "load_detection_metrics",
    "load_track_metrics",
    "pooled_counts",
    "pooled_track_metrics",
    "recall_by_behaviour",
    "recall_by_size",
    "score_bands",
    "variants_in",
]

TRACK_METRICS_SUBDIR = "tracking"
"""Where track metrics live, relative to ``artifacts/metrics/``."""

DETECTION_METRICS_SCHEMA = "panaf.detection-metrics/v1"
TRACKING_METRICS_SCHEMA = "panaf.tracking-metrics/v1"
"""Written into every metrics file so one can be identified in isolation."""

_LEGACY_TRACK_SUFFIX = "_tracking.json"

# Recorded per detection, so a band can be read against a tracker's floor.
_LOW_BAND = 0.10
_MID_BAND = 0.15

# Everything a plain `Detection` accepts, and nothing a tracked one adds.
_DETECTION_FIELDS = ("box", "confidence", "category_id", "category_name")


@dataclass(frozen=True, slots=True)
class PooledCounts:
    """Detection counts summed over clips, with the rates derived from them.

    Pooled over *detections*, not averaged over clips: a 600-box clip and a
    2-box clip must not carry equal weight.

    Attributes:
        true_positives: Predictions that matched an annotation.
        false_positives: Predictions that matched nothing.
        false_negatives: Annotations no prediction matched.
        matched_iou_sum: Sum of IoU over matched pairs, for a weighted mean.
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_iou_sum: float = 0.0

    @property
    def detections(self) -> int:
        """Predictions kept, i.e. true plus false positives."""
        return self.true_positives + self.false_positives

    @property
    def annotated(self) -> int:
        """Ground-truth boxes, i.e. true positives plus misses."""
        return self.true_positives + self.false_negatives

    @property
    def precision(self) -> float:
        """Fraction of predictions that were right."""
        return self.true_positives / self.detections if self.detections else 0.0

    @property
    def recall(self) -> float:
        """Fraction of annotated apes that were found."""
        return self.true_positives / self.annotated if self.annotated else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def mean_iou(self) -> float:
        """Mean IoU over matched pairs.

        Weighted by matched pairs, never averaged over clips -- a clip with no
        matches has a per-clip mean of 0.0 by construction, and averaging those
        in reports bad localisation where there was really none at all.
        """
        return self.matched_iou_sum / self.true_positives if self.true_positives else 0.0


@dataclass(frozen=True, slots=True)
class PooledTrackCounts:
    """Track quality summed over clips, with the rates derived from them.

    Coverage is pooled over *annotated frames*, not averaged over clips, for the
    same reason detection counts are: a 4-ape clip and a 1-ape clip must not
    carry equal weight. The per-clip files report per-clip means; this reports
    the dataset.

    Attributes:
        clips: Clips contributing.
        individuals: Annotated apes over those clips.
        predicted_tracks: Distinct tracks produced.
        id_switches: Times a followed ape's track id changed.
        id_merges: Tracks matched to more than one ape.
        annotated_frames: Ape-frames that could have been covered.
        covered_frames: Ape-frames some track covered.
        dominant_track_frames: Ape-frames the ape's single best track covered.
        mostly_tracked: Apes covered in at least 80% of their frames.
        mostly_lost: Apes covered in at most 20%.
        purity_sum: Track purities summed, for the mean.
        purity_count: Tracks contributing a purity.
        jitter_sum: Per-clip jitter summed, for the mean.
        jitter_count: Clips contributing a jitter.
    """

    clips: int = 0
    individuals: int = 0
    predicted_tracks: int = 0
    id_switches: int = 0
    id_merges: int = 0
    annotated_frames: int = 0
    covered_frames: int = 0
    dominant_track_frames: int = 0
    mostly_tracked: int = 0
    mostly_lost: int = 0
    purity_sum: float = 0.0
    purity_count: int = 0
    jitter_sum: float = 0.0
    jitter_count: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of annotated ape-frames any track covered."""
        return self.covered_frames / self.annotated_frames if self.annotated_frames else 0.0

    @property
    def identity_coverage(self) -> float:
        """Fraction of annotated ape-frames each ape's *best single* track held.

        The headline number for comparing tracker settings: it falls both when
        an ape is lost and when it is split across tracks.
        """
        if not self.annotated_frames:
            return 0.0
        return self.dominant_track_frames / self.annotated_frames

    @property
    def fragmentation(self) -> float:
        """Predicted tracks per annotated individual. ``1.0`` is ideal."""
        return self.predicted_tracks / self.individuals if self.individuals else 0.0

    @property
    def track_purity(self) -> float:
        """Mean share of a track's matched frames belonging to one ape."""
        return self.purity_sum / self.purity_count if self.purity_count else 1.0

    @property
    def jitter(self) -> float:
        """Mean normalised box acceleration over clips."""
        return self.jitter_sum / self.jitter_count if self.jitter_count else 0.0


@dataclass(frozen=True, slots=True)
class ScoreBands:
    """Detection scores bucketed against a tracker's usable range.

    ByteTrack discards detections at or below 0.1 outright and needs
    ``track_activation_threshold + 0.1`` to start a track, so recall recovered
    below roughly 0.15 never reaches it. These bands make that visible.

    Attributes:
        discarded: Scores at or below 0.10 -- a tracker cannot use these at all.
        cannot_start_track: Scores in (0.10, 0.15].
        usable: Scores above 0.15.
    """

    discarded: int = 0
    cannot_start_track: int = 0
    usable: int = 0

    @property
    def total(self) -> int:
        """All detections counted."""
        return self.discarded + self.cannot_start_track + self.usable

    @property
    def usable_fraction(self) -> float:
        """Fraction of detections a tracker could start a track from."""
        return self.usable / self.total if self.total else 0.0


def _metrics_dir(artifacts_dir: Path | str) -> Path:
    return Path(artifacts_dir) / "metrics"


def detection_fields(record: Mapping[str, object]) -> dict[str, object]:
    """Return only the fields a plain ``Detection`` accepts, from a saved record.

    Records in ``artifacts/detections/`` carry ``track_id`` and
    ``behavior_label`` whenever tracking ran. ``Detection`` is
    ``extra="forbid"``, so splatting such a record into it raises
    ``ValidationError`` -- which is exactly how ``panaf-phase1 evaluate`` came to
    crash on the repository's own default artifacts while its unit tests, whose
    fixtures were written with tracking off, kept passing.

    Reading a saved record therefore goes through here rather than through
    ``Detection(**record)`` at each call site.

    Args:
        record: One entry from a frame's ``detections`` list.

    Returns:
        A new dict holding only ``box``, ``confidence``, ``category_id`` and
        ``category_name``. Track fields are dropped, not merely ignored.

    Raises:
        KeyError: If the record is missing a field every detection must have.
    """
    return {field: record[field] for field in _DETECTION_FIELDS}


def load_detection_metrics(artifacts_dir: Path | str) -> list[dict[str, object]]:
    """Load every clip's detection metrics from ``artifacts/metrics/``.

    Track metrics are **excluded** two ways: the ``tracking/`` subdirectory is
    not descended into, and legacy ``<clip>_tracking.json`` files left by an
    older layout are skipped by name. A file that survives both but still lacks
    the detection schema is an error rather than something to paper over.

    Args:
        artifacts_dir: A run's artifacts directory.

    Returns:
        One dict per clip, sorted by ``clip_id``.

    Raises:
        FileNotFoundError: If the metrics directory does not exist.
        ValueError: If a file in ``metrics/`` is not detection metrics.
    """
    directory = _metrics_dir(artifacts_dir)
    if not directory.is_dir():
        msg = (
            f"no metrics directory at {directory}. Run `panaf-phase1 detect` first, "
            "or point this at the artifacts directory of a completed run."
        )
        raise FileNotFoundError(msg)

    loaded: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.json")):  # glob does not descend into tracking/
        if path.name.endswith(_LEGACY_TRACK_SUFFIX):
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if "overall" not in document:
            msg = (
                f"{path} is in metrics/ but is not detection metrics (no 'overall' key). "
                f"Detection metrics belong in metrics/, track metrics in "
                f"metrics/{TRACK_METRICS_SUBDIR}/."
            )
            raise ValueError(msg)
        loaded.append(document)
    return sorted(loaded, key=lambda d: str(d.get("clip_id", "")))


def load_track_metrics(artifacts_dir: Path | str) -> list[dict[str, object]]:
    """Load every clip's track metrics from ``artifacts/metrics/tracking/``.

    Falls back to the legacy ``metrics/<clip>_tracking.json`` layout, so an
    artifacts tree written before the split still reads.

    Args:
        artifacts_dir: A run's artifacts directory.

    Returns:
        One dict per clip, sorted by ``clip_id``. Empty when tracking was off.
    """
    directory = _metrics_dir(artifacts_dir)
    paths = sorted((directory / TRACK_METRICS_SUBDIR).glob("*.json"))
    if not paths:
        paths = sorted(directory.glob(f"*{_LEGACY_TRACK_SUFFIX}"))

    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return sorted(loaded, key=lambda d: str(d.get("clip_id", "")))


def _as_int(value: object) -> int:
    """Coerce a JSON value to int, treating absent and null as zero."""
    return int(value) if isinstance(value, (int, float)) else 0


def _as_float(value: object) -> float:
    """Coerce a JSON value to float, treating absent and null as zero."""
    return float(value) if isinstance(value, (int, float)) else 0.0


def _as_records(value: object) -> list[Mapping[str, object]]:
    """Return *value* as a list of mappings, or empty if it is neither."""
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def pooled_track_metrics(metrics: Iterable[Mapping[str, object]]) -> PooledTrackCounts:
    """Sum track metrics over clips into one comparable summary.

    Takes what :func:`load_track_metrics` returns. Tolerates records written
    before the identity fields existed: an older file simply contributes nothing
    to purity and jitter rather than being read as purity zero.

    Args:
        metrics: Per-clip track metric documents.

    Returns:
        The pooled counts.
    """
    clips = individuals = predicted_tracks = id_switches = id_merges = 0
    annotated_frames = covered_frames = dominant_track_frames = 0
    mostly_tracked = mostly_lost = 0
    purity_sum = jitter_sum = 0.0
    purity_count = jitter_count = 0

    for document in metrics:
        clips += 1
        predicted_tracks += _as_int(document.get("predicted_tracks"))
        id_switches += _as_int(document.get("total_id_switches"))
        id_merges += _as_int(document.get("id_merges"))
        mostly_tracked += _as_int(document.get("mostly_tracked"))
        mostly_lost += _as_int(document.get("mostly_lost"))

        if "mean_track_purity" in document:
            purity_sum += _as_float(document["mean_track_purity"])
            purity_count += 1
        if "mean_jitter" in document:
            jitter_sum += _as_float(document["mean_jitter"])
            jitter_count += 1

        for individual in _as_records(document.get("individuals")):
            individuals += 1
            annotated = _as_int(individual.get("annotated_frames"))
            annotated_frames += annotated
            covered_frames += _as_int(individual.get("covered_frames"))
            # Older records carry no identity_coverage. Falling back to
            # `covered` would credit them with perfect identity they were never
            # measured for, so they contribute nothing instead.
            share = individual.get("identity_coverage")
            if share is not None:
                dominant_track_frames += round(_as_float(share) * annotated)

    return PooledTrackCounts(
        clips=clips,
        individuals=individuals,
        predicted_tracks=predicted_tracks,
        id_switches=id_switches,
        id_merges=id_merges,
        annotated_frames=annotated_frames,
        covered_frames=covered_frames,
        dominant_track_frames=dominant_track_frames,
        mostly_tracked=mostly_tracked,
        mostly_lost=mostly_lost,
        purity_sum=purity_sum,
        purity_count=purity_count,
        jitter_sum=jitter_sum,
        jitter_count=jitter_count,
    )


def variants_in(metrics: Iterable[Mapping[str, object]]) -> set[str]:
    """Return the model variants that produced *metrics*.

    More than one means the directory is **mixed** -- almost always a run that
    stopped part-way, leaving some clips measured with the old model and some
    with the new. Pooling that mixture produces a number describing no model at
    all, and nothing else makes it visible: the files are individually valid.

    An empty string appears for records written before the variant was recorded.

    Args:
        metrics: Documents from :func:`load_detection_metrics`.

    Returns:
        The distinct variant strings found.
    """
    return {str(document.get("model_variant", "")) for document in metrics}


def pooled_counts(metrics: Iterable[Mapping[str, object]]) -> PooledCounts:
    """Sum detection counts across clips.

    Args:
        metrics: Documents from :func:`load_detection_metrics`.

    Returns:
        The pooled counts, with precision, recall, F1 and mean IoU derived.
    """
    true_positives = false_positives = false_negatives = 0
    iou_sum = 0.0
    for document in metrics:
        overall = document["overall"]
        assert isinstance(overall, Mapping)
        clip_tp = int(overall["true_positives"])
        true_positives += clip_tp
        false_positives += int(overall["false_positives"])
        false_negatives += int(overall["false_negatives"])
        clip_iou = document.get("mean_iou", 0.0)
        assert isinstance(clip_iou, (int, float))
        iou_sum += float(clip_iou) * clip_tp
    return PooledCounts(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        matched_iou_sum=iou_sum,
    )


def _recall_breakdown(
    metrics: Iterable[Mapping[str, object]], key: str
) -> dict[str, tuple[int, int]]:
    found: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for document in metrics:
        breakdown = document.get(key, {})
        assert isinstance(breakdown, Mapping)
        for label, counts in breakdown.items():
            assert isinstance(counts, Mapping)
            true_positives = int(counts["true_positives"])
            found[label] += true_positives
            total[label] += true_positives + int(counts["false_negatives"])
    return {label: (found[label], total[label]) for label in total}


def recall_by_behaviour(metrics: Iterable[Mapping[str, object]]) -> dict[str, tuple[int, int]]:
    """Return ``{behaviour: (found, annotated)}`` pooled over clips.

    The behaviour labels come from the **dataset**, never from the model --
    MegaDetector emits only ``animal``. This is a breakdown of detection recall
    conditioned on what the animal was doing, not a behaviour prediction.
    """
    return _recall_breakdown(metrics, "by_behaviour")


def recall_by_size(metrics: Iterable[Mapping[str, object]]) -> dict[str, tuple[int, int]]:
    """Return ``{size band: (found, annotated)}`` pooled over clips.

    Read this alongside the per-clip numbers: pooled size bands are a
    composition effect when the bands are unevenly spread across clips of very
    different difficulty.
    """
    return _recall_breakdown(metrics, "by_size")


def score_bands(artifacts_dir: Path | str) -> ScoreBands:
    """Bucket every saved detection's confidence into tracker-usable bands.

    Args:
        artifacts_dir: A run's artifacts directory; reads ``detections/``.

    Returns:
        The bands. All-zero when nothing has been saved.
    """
    directory = Path(artifacts_dir) / "detections"
    bands = [0, 0, 0]
    for path in sorted(directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for frame in document.get("frames", []):
            for detection in frame.get("detections", []):
                score = float(detection["confidence"])
                index = 0 if score <= _LOW_BAND else 1 if score <= _MID_BAND else 2
                bands[index] += 1
    return ScoreBands(discarded=bands[0], cannot_start_track=bands[1], usable=bands[2])


def latest_run_metadata(artifacts_dir: Path | str) -> dict[str, object] | None:
    """Return the most recently written run-metadata record, or ``None``.

    Selected by **modification time**, not by filename. Metadata files are named
    ``<experiment>_<timestamp>.json``, so once several experiments share a
    directory the lexically last file belongs to whichever experiment name sorts
    highest -- not to the run that just finished.

    Args:
        artifacts_dir: A run's artifacts directory; reads ``metadata/``.

    Returns:
        The parsed record, or ``None`` when no run has written one.
    """
    paths = list((Path(artifacts_dir) / "metadata").glob("*.json"))
    if not paths:
        return None
    newest = max(paths, key=lambda path: path.stat().st_mtime)
    document: dict[str, object] = json.loads(newest.read_text(encoding="utf-8"))
    return document


def format_recall_table(
    breakdown: Mapping[str, tuple[int, int]], *, order: Sequence[str] = ()
) -> str:
    """Render a ``{label: (found, total)}`` breakdown as aligned text.

    Args:
        breakdown: From :func:`recall_by_behaviour` or :func:`recall_by_size`.
        order: Labels to show in this order. Anything omitted is appended,
            worst recall first, so a caller can pin a natural ordering (small,
            medium, large) without losing labels it did not anticipate.

    Returns:
        One line per label, or a short message when there is nothing to show.
    """
    if not breakdown:
        return "  (nothing recorded)"

    def recall(label: str) -> float:
        found, total = breakdown[label]
        return found / total if total else 0.0

    pinned = [label for label in order if label in breakdown]
    rest = sorted((label for label in breakdown if label not in pinned), key=recall)
    width = max(len(label) for label in breakdown)
    lines = []
    for label in [*pinned, *rest]:
        found, total = breakdown[label]
        lines.append(f"  {label:<{width}}  {found:>6}/{total:<6}  {recall(label):.3f}")
    return "\n".join(lines)
