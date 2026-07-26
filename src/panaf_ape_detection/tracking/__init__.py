"""Multi-object tracking: give each ape a stable identity across frames.

Phase 1 step 4. The backend was deliberately left undecided until the detection
baseline was measured; ByteTrack was chosen because it is already available
(``supervision`` ships it as a PyTorch-Wildlife dependency, so no new package)
and because its second-pass association of low-confidence boxes targets exactly
the flicker that a recall-0.41 detector produces.

Every ``supervision`` import is lazy, so this package imports without the
``inference`` extra installed.
"""

from __future__ import annotations

from panaf_ape_detection.tracking.base import Tracker

__all__ = ["Tracker"]
