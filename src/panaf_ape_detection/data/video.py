"""Decode clips into frames.

OpenCV is imported lazily inside each function, so importing this module stays
cheap and works without the ``inference`` extra installed.

Frames are yielded as BGR ``numpy`` arrays -- OpenCV's native order, and what
PyTorch-Wildlife's detectors expect -- paired with their **zero-based** index in
the full decoded sequence. The index is the position in the video, *not* a count
of yielded frames, so it still lines up with ground truth when ``frame_stride``
skips frames.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = ["VideoProperties", "iter_frames", "read_video_properties"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VideoProperties:
    """What a clip reports about itself.

    Attributes:
        path: The decoded file.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second as reported by the container.
        frame_count: Frame count as reported by the container. Containers
            sometimes lie about this; treat it as a hint, not a guarantee.
    """

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        """Approximate duration from the reported frame count and rate."""
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def read_video_properties(path: Path | str) -> VideoProperties:
    """Read a clip's dimensions and frame rate without decoding it.

    Args:
        path: Video file.

    Returns:
        The clip's :class:`VideoProperties`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If OpenCV cannot open the file.
    """
    import cv2

    video_path = Path(path)
    if not video_path.is_file():
        msg = f"video not found: {video_path}"
        raise FileNotFoundError(msg)

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            msg = f"OpenCV could not open {video_path}. Is FFmpeg available?"
            raise RuntimeError(msg)
        return VideoProperties(
            path=video_path,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def iter_frames(
    path: Path | str, *, frame_stride: int = 1, limit: int | None = None
) -> Iterator[tuple[int, np.ndarray[Any, Any]]]:
    """Yield ``(frame_index, frame)`` pairs from a clip.

    Args:
        path: Video file.
        frame_stride: Yield every ``frame_stride``-th frame. Every frame is
            still *decoded* -- seeking is unreliable on many codecs, and a wrong
            frame index is worse than a slow one.
        limit: Stop after yielding this many frames. Useful for smoke runs.

    Yields:
        ``(frame_index, frame)`` where ``frame_index`` is the **zero-based
        position in the full video** and ``frame`` is a BGR array.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If OpenCV cannot open the file.
        ValueError: If *frame_stride* is not positive.
    """
    import cv2

    if frame_stride < 1:
        msg = f"frame_stride must be >= 1, got {frame_stride}"
        raise ValueError(msg)

    video_path = Path(path)
    if not video_path.is_file():
        msg = f"video not found: {video_path}"
        raise FileNotFoundError(msg)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        msg = f"OpenCV could not open {video_path}. Is FFmpeg available?"
        raise RuntimeError(msg)

    yielded = 0
    index = 0
    try:
        while True:
            got, frame = capture.read()
            if not got:
                break
            if index % frame_stride == 0:
                yield index, frame
                yielded += 1
                if limit is not None and yielded >= limit:
                    break
            index += 1
    finally:
        capture.release()
        logger.debug("decoded %d frames from %s, yielded %d", index, video_path.name, yielded)
