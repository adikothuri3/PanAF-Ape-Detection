"""The tracker seam.

Structural, like the detector protocol: anything with a matching ``update`` and
``reset`` is a tracker. Evaluation and tests use a stub, so track-quality metrics
can be exercised without ``supervision`` installed and without running a detector.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from panaf_ape_detection.types import Detection, TrackedDetection

__all__ = ["Tracker"]


@runtime_checkable
class Tracker(Protocol):
    """Assigns stable identities to detections across frames of one clip."""

    @property
    def name(self) -> str:
        """Backend name, e.g. ``bytetrack``. Recorded in run metadata."""
        ...

    def update(self, detections: Sequence[Detection]) -> list[TrackedDetection]:
        """Associate one frame's detections with existing tracks.

        Called once per frame, **in order**. A tracker is stateful across a clip,
        which is why :meth:`reset` exists.

        Args:
            detections: This frame's detections, already confidence-filtered.

        Returns:
            The subset carrying track identities. A tracker may return fewer
            detections than it was given -- ByteTrack withholds a track until it
            has been seen enough times -- so the count is not guaranteed to match.
        """
        ...

    def reset(self) -> None:
        """Forget all state.

        **Must be called between clips.** Track ids are only meaningful within one
        video; carrying state across clips would invent continuity that does not
        exist.
        """
        ...
