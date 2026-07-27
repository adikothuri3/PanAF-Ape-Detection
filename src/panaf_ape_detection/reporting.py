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
    "ScoreBands",
    "format_recall_table",
    "latest_run_metadata",
    "load_detection_metrics",
    "load_track_metrics",
    "pooled_counts",
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
