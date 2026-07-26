"""Confidence filtering.

Kept separate from the detector so that raw scores are recorded once and can be
re-thresholded later without re-running inference -- which is what makes a
threshold sweep cheap rather than a fresh GPU-hour.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from panaf_ape_detection.types import Detection

__all__ = ["ANIMAL_CATEGORY", "filter_by_confidence", "keep_animals"]

ANIMAL_CATEGORY = "animal"
"""MegaDetector's class for any animal.

Not a species. A chimpanzee, a gorilla and a duiker are all ``animal``; the
detector cannot distinguish them, and neither may anything downstream.
"""


def filter_by_confidence(detections: Iterable[Detection], threshold: float) -> list[Detection]:
    """Keep detections scoring at or above *threshold*.

    Args:
        detections: Raw detections.
        threshold: Minimum score, in ``[0, 1]``.

    Returns:
        The surviving detections, order preserved.

    Raises:
        ValueError: If *threshold* is outside ``[0, 1]``.
    """
    if not 0.0 <= threshold <= 1.0:
        msg = f"threshold must be in [0, 1], got {threshold}"
        raise ValueError(msg)
    return [d for d in detections if d.confidence >= threshold]


def keep_animals(detections: Sequence[Detection]) -> list[Detection]:
    """Keep only the ``animal`` class.

    PanAf footage is camera-trap video of apes, so ``person`` and ``vehicle``
    detections are either wrong or irrelevant to the question being asked. They
    are dropped here rather than in the detector so the raw record keeps them --
    a spike in ``person`` detections would itself be a finding.
    """
    return [d for d in detections if d.category_name == ANIMAL_CATEGORY]
