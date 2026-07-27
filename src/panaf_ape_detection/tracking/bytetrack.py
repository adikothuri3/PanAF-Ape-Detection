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
from dataclasses import dataclass
from typing import Any

from panaf_ape_detection.tracking.convert import from_supervision, to_supervision
from panaf_ape_detection.types import Detection, TrackedDetection

__all__ = [
    "DEFAULT_LOST_TRACK_BUFFER",
    "UPSTREAM_SCORE_FLOOR",
    "ByteTrackTracker",
    "ScoreFloor",
    "drop_short_tracks",
]

logger = logging.getLogger(__name__)

UPSTREAM_SCORE_FLOOR = 0.1
"""Score at or below which supervision discards a detection outright.

A literal in ``byte_tracker/core.py`` (``inds_low = scores > 0.1``), not a
parameter, so no configuration can move it. Named here so code that has to work
around it can refer to the constraint rather than repeat the number.
"""

_UPSTREAM_NEW_TRACK_MARGIN = 0.1
"""How far above the activation threshold a detection must score to start a track.

``self.det_thresh = self.track_activation_threshold + 0.1``, also a literal.
"""


@dataclass(frozen=True, slots=True)
class ScoreFloor:
    """Lifts detection scores clear of ByteTrack's hardcoded discard floor.

    **Why this exists.** ``supervision`` drops every detection scoring at or
    below :data:`UPSTREAM_SCORE_FLOOR` before either association pass, and the
    number is a literal rather than a parameter. Over this project's 10-clip
    sample at confidence 0.05 that is **1,185 of 6,384 detections, 18.6%** --
    thrown away by the low-score pass that is the entire reason ByteTrack was
    chosen over SORT.

    The fix is a monotone affine map applied on the way in and undone on the way
    out::

        forward(s) = floor + s * (1 - floor)

    It fixes ``1.0``, maps ``0.0`` to *floor*, and preserves order, so it changes
    *which boxes ByteTrack will look at* without changing their ranking. The
    activation threshold is mapped through it too, so "activation 0.30" keeps
    meaning a 0.30 detector score.

    **Nothing downstream ever sees a transformed score.** :meth:`inverse` is
    applied to every tracked detection before it is returned, so artifacts,
    metrics and video all carry the detector's own numbers. Rescaling scores and
    then reporting them would be inventing results.

    One honest cost: because the upstream new-track margin is a fixed ``+0.1``
    in transformed space, starting a track needs ``activation + 0.1 / (1 - floor)``
    in original terms -- very slightly stricter than without the map.

    Attributes:
        floor: Where an original score of ``0.0`` lands. Must be above
            :data:`UPSTREAM_SCORE_FLOOR` to achieve anything, and below 1.0.
    """

    floor: float

    def __post_init__(self) -> None:
        """Reject a floor that cannot work.

        Raises:
            ValueError: If *floor* is outside ``[0, 1)``.
        """
        if not 0.0 <= self.floor < 1.0:
            msg = f"score floor must be in [0, 1), got {self.floor}"
            raise ValueError(msg)

    def forward(self, score: float) -> float:
        """Map a detector score into the range ByteTrack will consider."""
        return self.floor + score * (1.0 - self.floor)

    def inverse(self, score: float) -> float:
        """Recover the original detector score, clamped to ``[0, 1]``."""
        original = (score - self.floor) / (1.0 - self.floor)
        return min(1.0, max(0.0, original))


DEFAULT_LOST_TRACK_BUFFER = 30
"""Frames a track survives unmatched before it is dropped, **at 30 fps**.

Not a frame count: supervision rescales it by the clip's rate,
``max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)``. At PanAf's 24 fps
a buffer of 30 is therefore **24 frames, 1.0 s** -- not the 1.25 s this docstring
claimed until it was checked against the installed source.

Generous on purpose: an ape is routinely undetected for a stretch, and a short
buffer would split one animal into many tracks purely because the detector
blinked.
"""


class ByteTrackTracker:
    """Wraps ``supervision.ByteTrack`` behind the project's tracker protocol."""

    def __init__(
        self,
        *,
        activation_threshold: float = 0.2,
        frame_rate: float = 24.0,
        lost_track_buffer: int = DEFAULT_LOST_TRACK_BUFFER,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
        score_floor: float | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None:
        """Create a tracker for one clip.

        Args:
            activation_threshold: Score at or above which a detection takes part
                fully in association. Named for what ByteTrack does with it
                rather than for where the value used to come from: it was the
                detector's confidence threshold only because nothing could set
                it separately.
            frame_rate: The clip's actual frame rate. ByteTrack's default is 30;
                PanAf clips are 24, and the rate scales its motion model.
            lost_track_buffer: Frames a track survives unmatched, at 30 fps.
            minimum_matching_threshold: IoU-distance threshold for association.
            minimum_consecutive_frames: Frames a new track must be seen in before
                it is reported.
            score_floor: Lift scores clear of ByteTrack's hardcoded 0.1 discard
                floor before association; see :class:`ScoreFloor`. ``None``
                leaves scores untouched, which is the previous behaviour.
            frame_width: Clamp output boxes to this width.
            frame_height: Clamp output boxes to this height.
        """
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._activation_threshold = activation_threshold
        self._frame_rate = frame_rate
        self._lost_track_buffer = lost_track_buffer
        self._minimum_matching_threshold = minimum_matching_threshold
        self._minimum_consecutive_frames = minimum_consecutive_frames
        self._score_floor = ScoreFloor(score_floor) if score_floor else None

        self._tracker = self._build()
        logger.info(
            "ByteTrack: activation=%.2f (new track needs %.2f), buffer=%d@30fps "
            "(%d frames at %.1f fps), match=%.2f, min consecutive=%d, score floor=%s",
            activation_threshold,
            activation_threshold + _UPSTREAM_NEW_TRACK_MARGIN,
            lost_track_buffer,
            int(frame_rate / 30.0 * lost_track_buffer),
            frame_rate,
            minimum_matching_threshold,
            minimum_consecutive_frames,
            "off" if self._score_floor is None else f"{self._score_floor.floor:.2f}",
        )

    def _build(self) -> Any:
        """Construct a fresh ByteTrack with this tracker's settings.

        One construction point, used by both ``__init__`` and :meth:`reset`, so
        the two cannot drift apart and leave a reset tracker configured
        differently from a new one.
        """
        import supervision as sv

        # The activation threshold goes through the same map as the scores, so
        # it keeps meaning a detector score rather than a transformed one.
        activation = self._activation_threshold
        if self._score_floor is not None:
            activation = self._score_floor.forward(activation)

        return sv.ByteTrack(
            track_activation_threshold=activation,
            lost_track_buffer=self._lost_track_buffer,
            minimum_matching_threshold=self._minimum_matching_threshold,
            minimum_consecutive_frames=self._minimum_consecutive_frames,
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

        When a score floor is configured the transform is applied here and undone
        before returning, so it is contained entirely within this call and no
        caller ever handles a rescaled score.
        """
        if self._score_floor is None:
            result = self._tracker.update_with_detections(to_supervision(detections))
            return from_supervision(
                result, frame_width=self._frame_width, frame_height=self._frame_height
            )

        floor = self._score_floor
        lifted = [
            detection.model_copy(update={"confidence": floor.forward(detection.confidence)})
            for detection in detections
        ]
        result = self._tracker.update_with_detections(to_supervision(lifted))
        return [
            tracked.model_copy(update={"confidence": floor.inverse(tracked.confidence)})
            for tracked in from_supervision(
                result, frame_width=self._frame_width, frame_height=self._frame_height
            )
        ]

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
