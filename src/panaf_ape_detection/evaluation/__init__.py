"""Measure detections against PanAf500 ground truth.

Pure arithmetic over the shared schemas -- no machine-learning imports, so the
metrics can be recomputed from saved detection records without a GPU or the
``inference`` extra.
"""

from __future__ import annotations

from panaf_ape_detection.evaluation.detection import (
    ClipEvaluation,
    FrameMatch,
    MatchCounts,
    SizeBand,
    evaluate_clip,
    intersection_over_union,
    match_frame,
    size_band,
)

__all__ = [
    "ClipEvaluation",
    "FrameMatch",
    "MatchCounts",
    "SizeBand",
    "evaluate_clip",
    "intersection_over_union",
    "match_frame",
    "size_band",
]
