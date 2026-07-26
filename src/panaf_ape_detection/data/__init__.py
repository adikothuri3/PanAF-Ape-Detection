"""Dataset loading: the clip manifest, PanAf500 annotations, and video decoding.

Nothing here imports the machine-learning stack. Video decoding needs OpenCV, so
that import is lazy and lives inside the function that uses it.
"""

from __future__ import annotations

from panaf_ape_detection.data.annotations import (
    GroundTruthDetection,
    GroundTruthFrame,
    load_ground_truth,
    normalise_behaviour,
)
from panaf_ape_detection.data.video import VideoProperties, iter_frames, read_video_properties

__all__ = [
    "GroundTruthDetection",
    "GroundTruthFrame",
    "VideoProperties",
    "iter_frames",
    "load_ground_truth",
    "normalise_behaviour",
    "read_video_properties",
]
