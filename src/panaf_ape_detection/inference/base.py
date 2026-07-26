"""The detector seam.

Structural, not inherited: anything with a matching ``detect`` is a detector.
Tests use a stub that never downloads weights, and evaluation never learns which
implementation produced its input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from panaf_ape_detection.types import Detection, DeviceKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = ["Detector"]


@runtime_checkable
class Detector(Protocol):
    """Something that finds objects in a frame."""

    @property
    def name(self) -> str:
        """Detector family, e.g. ``MegaDetectorV6``."""
        ...

    @property
    def variant(self) -> str:
        """Exact weight identifier, e.g. ``MDV6-yolov9-c``.

        Recorded with every run: a detection count without the variant that
        produced it cannot be compared with anything.
        """
        ...

    @property
    def device(self) -> DeviceKind:
        """The device the weights were **verified** to be on.

        Not the device that was requested. PyTorch-Wildlife accepts a ``device=``
        argument and silently ignores it, so implementations must check the
        tensors and report what they find.
        """
        ...

    def detect(self, frame: np.ndarray[Any, Any]) -> list[Detection]:
        """Detect objects in one BGR frame.

        Args:
            frame: Height-by-width-by-3 BGR array, as OpenCV decodes.

        Returns:
            Unfiltered detections; confidence thresholding happens downstream so
            the raw scores stay available for threshold sweeps.
        """
        ...
