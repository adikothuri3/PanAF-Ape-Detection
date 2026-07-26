"""Detector adapters.

The protocol in :mod:`panaf_ape_detection.inference.base` is the seam: frames in,
:class:`~panaf_ape_detection.types.Detection` out. Everything downstream depends
on that shape, never on PyTorch-Wildlife, so the detector can be swapped or
stubbed without touching evaluation, visualization or the runner.
"""

from __future__ import annotations

from panaf_ape_detection.inference.base import Detector
from panaf_ape_detection.inference.filtering import filter_by_confidence

__all__ = ["Detector", "filter_by_confidence"]
