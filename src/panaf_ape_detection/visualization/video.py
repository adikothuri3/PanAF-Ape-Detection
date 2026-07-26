"""Stitch annotated frames back into video.

OpenCV writes the MP4 and imageio writes the GIF; both are imported lazily.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = ["VideoWriter", "write_gif"]

logger = logging.getLogger(__name__)


class VideoWriter:
    """Append frames to an MP4, as a context manager.

    Frames are written as they are produced rather than accumulated, so memory
    stays flat regardless of clip length.

    Example:
        >>> with VideoWriter(path, width=720, height=404, fps=24.0) as writer:
        ...     writer.write(frame)
    """

    def __init__(
        self,
        path: Path | str,
        *,
        width: int,
        height: int,
        fps: float,
        codec: str = "mp4v",
    ) -> None:
        """Open a video file for writing.

        Args:
            path: Destination. Parent directories are created.
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Output frame rate.
            codec: FourCC code. ``mp4v`` is the most portable across the OpenCV
                builds this project installs.

        Raises:
            ValueError: If dimensions or frame rate are not positive.
            RuntimeError: If OpenCV cannot open the writer.
        """
        import cv2

        if width <= 0 or height <= 0:
            msg = f"frame dimensions must be positive, got {width}x{height}"
            raise ValueError(msg)
        if fps <= 0:
            msg = f"fps must be positive, got {fps}"
            raise ValueError(msg)

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.frames_written = 0

        # `VideoWriter.fourcc` rather than the older module-level
        # `cv2.VideoWriter_fourcc`: same result, but it is the API the OpenCV
        # type stubs actually declare.
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter.fourcc(*codec), fps, (width, height)
        )
        if not self._writer.isOpened():
            msg = f"OpenCV could not open a {codec!r} writer for {self.path}"
            raise RuntimeError(msg)

    def write(self, frame: np.ndarray[Any, Any]) -> None:
        """Append one BGR frame.

        Raises:
            ValueError: If the frame's size does not match the writer's. OpenCV
                silently drops mismatched frames, which produces a truncated
                video with no error at all.
        """
        height, width = frame.shape[:2]
        if (width, height) != (self.width, self.height):
            msg = (
                f"frame is {width}x{height} but the writer expects "
                f"{self.width}x{self.height}; OpenCV would drop it silently"
            )
            raise ValueError(msg)
        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        """Finalise the file."""
        self._writer.release()
        logger.info("wrote %d frames to %s", self.frames_written, self.path)

    def __enter__(self) -> VideoWriter:
        """Enter the context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Always finalise, even if the caller raised."""
        self.close()


def write_gif(
    path: Path | str,
    frames: Sequence[np.ndarray[Any, Any]],
    *,
    fps: float = 8.0,
    max_width: int = 480,
) -> Path:
    """Write frames to an animated GIF.

    GIF is one of the accepted Phase 1 deliverable formats and embeds directly
    in a write-up. Frames are downscaled and the rate reduced, because a
    full-resolution GIF of 360 frames is enormous for no benefit.

    Args:
        path: Destination ``.gif``.
        frames: BGR frames, as produced by the overlay.
        fps: Playback rate.
        max_width: Downscale so width does not exceed this.

    Returns:
        The path written.

    Raises:
        ValueError: If *frames* is empty.
    """
    import cv2
    import imageio.v2 as imageio

    if not frames:
        msg = "cannot write a GIF with no frames"
        raise ValueError(msg)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    prepared: list[Any] = []
    for frame in frames:
        height, width = frame.shape[:2]
        resized = frame
        if width > max_width:
            scale = max_width / width
            resized = cv2.resize(frame, (max_width, int(height * scale)))
        prepared.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    imageio.mimsave(str(target), prepared, duration=1.0 / fps, loop=0)
    logger.info("wrote %d frames to %s", len(prepared), target)
    return target
