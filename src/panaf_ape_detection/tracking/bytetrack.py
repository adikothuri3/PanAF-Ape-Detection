"""ByteTrack, via ``supervision``.

Chosen over SORT for two reasons that only became clear once detection was
measured. It needs no new dependency — ``supervision`` arrives with
PyTorch-Wildlife — and its defining feature is a second association pass over
*low-confidence* boxes, which is aimed precisely at the detector flicker a
recall-0.41 baseline produces.

**One trap is handled here explicitly.** ``sv.ByteTrack`` defaults to
``track_activation_threshold=0.25``, which is *above* this project's default
confidence threshold of 0.20. Left alone, detections scoring 0.20-0.25 would
survive filtering and then be silently ignored by the tracker — a second,
invisible threshold. The adapter drives it from the configured confidence
threshold instead.

**Two more thresholds are hardcoded upstream and cannot be configured away.**
Setting ``track_activation_threshold`` low does *not* make the tracker consider
everything the detector kept, which is what this module originally assumed:

* ``inds_low = scores > 0.1`` — detections scoring **0.1 or less are discarded
  outright**, whatever the activation threshold says.
* ``self.det_thresh = self.track_activation_threshold + 0.1`` — a new track is
  only started from a detection scoring above *activation + 0.1*.

So there is a hard floor: **ByteTrack cannot start a track from anything at or
below 0.1**, and the effective floor is ``activation + 0.1``. Measured
consequence over the 10-clip sample: dropping the detector to confidence 0.05
raises detector recall 0.386 → 0.563, but leaves tracking coverage unchanged
(0.353) and *increases* ID switches, because 87% of the newly recovered
detections in the hardest clip score at or below 0.1 and are thrown away here.
See ``reports/phase1_findings_2026-07-26.md`` §7.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from panaf_ape_detection.tracking.convert import from_supervision, to_supervision
from panaf_ape_detection.types import Detection, TrackedDetection

__all__ = ["DEFAULT_LOST_TRACK_BUFFER", "ByteTrackTracker", "drop_short_tracks"]

logger = logging.getLogger(__name__)

DEFAULT_LOST_TRACK_BUFFER = 30
"""Frames a track survives unmatched before it is dropped.

At 24 fps this is 1.25 s. Generous on purpose: with recall around 0.4 an ape is
routinely undetected for a stretch, and a short buffer would split one animal
into many tracks purely because the detector blinked.
"""


class ByteTrackTracker:
    """Wraps ``supervision.ByteTrack`` behind the project's tracker protocol."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.2,
        frame_rate: float = 24.0,
        lost_track_buffer: int = DEFAULT_LOST_TRACK_BUFFER,
        minimum_matching_threshold: float = 0.8,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None:
        """Create a tracker for one clip.

        Args:
            confidence_threshold: The threshold detections were already filtered
                at. Passed through as ``track_activation_threshold`` so the
                tracker does not apply a second, higher one of its own.
            frame_rate: The clip's actual frame rate. ByteTrack's default is 30;
                PanAf clips are 24, and the rate scales its motion model.
            lost_track_buffer: Frames a track survives unmatched.
            minimum_matching_threshold: IoU-distance threshold for association.
            frame_width: Clamp output boxes to this width.
            frame_height: Clamp output boxes to this height.
        """
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._confidence_threshold = confidence_threshold
        self._frame_rate = frame_rate
        self._lost_track_buffer = lost_track_buffer
        self._minimum_matching_threshold = minimum_matching_threshold

        self._tracker = self._build()
        logger.info(
            "ByteTrack: activation=%.2f, buffer=%d frames, match=%.2f, fps=%.1f",
            confidence_threshold,
            lost_track_buffer,
            minimum_matching_threshold,
            frame_rate,
        )

    def _build(self) -> Any:
        """Construct a fresh ByteTrack with this tracker's settings.

        One construction point, used by both ``__init__`` and :meth:`reset`, so
        the two cannot drift apart and leave a reset tracker configured
        differently from a new one.
        """
        import supervision as sv

        return sv.ByteTrack(
            track_activation_threshold=self._confidence_threshold,
            lost_track_buffer=self._lost_track_buffer,
            minimum_matching_threshold=self._minimum_matching_threshold,
            frame_rate=round(self._frame_rate) or 1,
        )

    @property
    def name(self) -> str:
        """Backend name, recorded in run metadata."""
        return "bytetrack"

    def update(self, detections: Sequence[Detection]) -> list[TrackedDetection]:
        """Associate one frame's detections with existing tracks.

        Frames with no detections **must** still be passed in, otherwise the
        tracker never learns a track went missing and its lost-track buffer
        cannot expire.
        """
        result = self._tracker.update_with_detections(to_supervision(detections))
        return from_supervision(
            result, frame_width=self._frame_width, frame_height=self._frame_height
        )

    def reset(self) -> None:
        """Forget all state. Must be called between clips."""
        self._tracker = self._build()


def drop_short_tracks(
    per_frame: Mapping[int, Sequence[TrackedDetection]], minimum_length: int
) -> dict[int, list[TrackedDetection]]:
    """Remove tracks appearing in fewer than *minimum_length* frames.

    Applied after the whole clip has been tracked, because track length is only
    knowable once. A track seen in one or two frames is usually a flicker rather
    than an animal, and counting it would flatter the fragmentation numbers.

    Args:
        per_frame: ``{frame_index: tracked detections}`` for one clip.
        minimum_length: Minimum frames a track must appear in. ``1`` keeps
            everything.

    Returns:
        The same mapping with short tracks removed. Frames left empty are
        **kept** as empty lists -- dropping them would lose the fact that the
        frame was processed.

    Raises:
        ValueError: If *minimum_length* is below 1.
    """
    if minimum_length < 1:
        msg = f"minimum_length must be >= 1, got {minimum_length}"
        raise ValueError(msg)

    counts: dict[int, int] = {}
    for detections in per_frame.values():
        for detection in detections:
            counts[detection.track_id] = counts.get(detection.track_id, 0) + 1

    keep = {track_id for track_id, count in counts.items() if count >= minimum_length}
    dropped = len(counts) - len(keep)
    if dropped:
        logger.info("dropped %d track(s) shorter than %d frames", dropped, minimum_length)

    return {
        index: [d for d in detections if d.track_id in keep]
        for index, detections in per_frame.items()
    }
