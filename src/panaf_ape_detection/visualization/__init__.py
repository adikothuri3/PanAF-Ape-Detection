"""Drawing detections and stitching annotated video.

Presentation only. Nothing here may alter a recorded detection -- a bug in the
overlay must never be able to change a number in the metrics.
"""

from __future__ import annotations

from panaf_ape_detection.visualization.overlays import draw_frame
from panaf_ape_detection.visualization.video import VideoWriter, write_gif

__all__ = ["VideoWriter", "draw_frame", "write_gif"]
