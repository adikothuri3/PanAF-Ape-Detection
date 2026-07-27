"""Post-processing over finished tracks: stitch, interpolate, smooth.

Three pure functions over ``{frame_index: tracked detections}``, each doing one
thing and each independently switchable, so a sweep can attribute any gain to
the step that produced it. They run in a fixed order, documented at
:func:`refine`, and none of them needs the video, the detector or a GPU.

Why each exists, measured rather than assumed:

* **Stitching** is the fragmentation fix. The detector loses an ape and finds it
  again; ByteTrack cannot always reconnect the two, so one animal becomes two
  tracks. Nothing downstream can undo that, because the identity was lost at the
  moment the second track was created.
* **Interpolation** is the *only* thing here that can raise coverage. Tracking
  coverage is otherwise pinned to detection recall exactly -- measured at 0.7149
  against recall 0.7149 on the 10-clip sample -- because a track can only cover a
  frame the detector produced a box for. Filling a gap inside a track is the one
  way to cover a frame the detector missed.
* **Smoothing** changes no identity and no coverage. It exists because the
  annotated video is a deliverable and a box that jitters frame to frame reads as
  a broken tracker even when the association is perfect.

**Two honesty constraints run through the module.**

Interpolated boxes are *synthesised*, not detected, so every one is marked
:attr:`~panaf_ape_detection.types.TrackedDetection.interpolated`. They are
predictions and are scored as predictions -- an interpolated box that lands on
nothing is a false positive like any other, which is why interpolation has to be
measured against detection precision and not only against coverage.

Stitching can *lower* ID switches and fragmentation by being wrong: join two
different apes into one track and both metrics improve. Two rules guard it --
tracks that overlap in time are never merged, since simultaneous visibility
proves they are different animals, and
:attr:`~panaf_ape_detection.evaluation.tracking.ClipTrackEvaluation.mean_track_purity`
measures whether the rule was enough.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from itertools import pairwise

from panaf_ape_detection.types import BoundingBox, TrackedDetection

__all__ = [
    "interpolate_gaps",
    "refine",
    "smooth_tracks",
    "stitch_tracks",
]

logger = logging.getLogger(__name__)

_Tracks = dict[int, dict[int, TrackedDetection]]
"""``{track_id: {frame_index: detection}}`` -- the per-track view these need."""


def _to_tracks(per_frame: Mapping[int, Sequence[TrackedDetection]]) -> _Tracks:
    """Regroup per-frame detections by track id."""
    tracks: _Tracks = {}
    for frame_index, detections in per_frame.items():
        for detection in detections:
            tracks.setdefault(detection.track_id, {})[frame_index] = detection
    return tracks


def _to_frames(tracks: _Tracks, frame_indices: Sequence[int]) -> dict[int, list[TrackedDetection]]:
    """Regroup tracks back into per-frame lists.

    *frame_indices* is passed in rather than derived, so frames that end up empty
    are still present -- dropping them would lose the fact that they were
    processed, and coverage would silently be computed over fewer frames.
    """
    rebuilt: dict[int, list[TrackedDetection]] = {index: [] for index in frame_indices}
    for track in tracks.values():
        for frame_index, detection in track.items():
            rebuilt.setdefault(frame_index, []).append(detection)
    return {index: rebuilt[index] for index in sorted(rebuilt)}


def _centre(box: BoundingBox) -> tuple[float, float]:
    return ((box.x_min + box.x_max) / 2.0, (box.y_min + box.y_max) / 2.0)


def _diagonal(box: BoundingBox) -> float:
    return float((box.width**2 + box.height**2) ** 0.5)


def stitch_tracks(
    per_frame: Mapping[int, Sequence[TrackedDetection]],
    *,
    max_gap: int,
    max_distance: float = 1.0,
    min_size_ratio: float = 0.5,
) -> dict[int, list[TrackedDetection]]:
    """Join track fragments that are plausibly the same animal.

    A fragment *B* may be appended to a chain ending in *A* when all of the
    following hold:

    1. **They do not overlap in time.** ``A`` must end strictly before ``B``
       begins. Two tracks visible in the same frame are two animals, and no
       distance test may override that.
    2. The gap between them is at most *max_gap* frames.
    3. ``A``'s motion, extrapolated linearly to ``B``'s first frame, lands within
       *max_distance* box-diagonals of where ``B`` starts.
    4. Their box sizes are compatible, within *min_size_ratio*.

    Linking is greedy and one-to-one, in order of first appearance: each fragment
    takes at most one predecessor, and each chain at most one successor, so a
    single ape cannot absorb a crowd.

    Args:
        per_frame: ``{frame_index: tracked detections}`` for one clip.
        max_gap: Largest gap to bridge, in frames. ``0`` disables stitching.
        max_distance: Positional tolerance, in box diagonals.
        min_size_ratio: Smallest allowed ratio between the two boxes' areas.

    Returns:
        The same mapping with merged tracks renumbered to their chain's first id.

    Raises:
        ValueError: If *max_gap* is negative.
    """
    if max_gap < 0:
        msg = f"max_gap must be >= 0, got {max_gap}"
        raise ValueError(msg)
    if max_gap == 0:
        return {index: list(detections) for index, detections in per_frame.items()}

    frame_indices = sorted(per_frame)
    tracks = _to_tracks(per_frame)

    # Chains, keyed by the id they will be renumbered to, in order of first frame.
    order = sorted(tracks, key=lambda track_id: min(tracks[track_id]))
    chains: dict[int, dict[int, TrackedDetection]] = {}
    merges = 0

    for track_id in order:
        fragment = tracks[track_id]
        start = min(fragment)
        best_chain: int | None = None
        best_cost = float("inf")

        for chain_id, chain in chains.items():
            end = max(chain)
            gap = start - end
            # Rule 1 and 2: strictly after, and close enough in time.
            if gap <= 0 or gap > max_gap:
                continue

            tail = chain[end]
            head = fragment[start]
            # Rule 4: comparable size.
            larger = max(tail.box.area, head.box.area)
            if larger <= 0 or min(tail.box.area, head.box.area) / larger < min_size_ratio:
                continue

            # Rule 3: where constant velocity would have carried the tail.
            previous = chain.get(end - 1)
            tail_x, tail_y = _centre(tail.box)
            if previous is None:
                predicted = (tail_x, tail_y)
            else:
                previous_x, previous_y = _centre(previous.box)
                predicted = (
                    tail_x + (tail_x - previous_x) * gap,
                    tail_y + (tail_y - previous_y) * gap,
                )
            head_x, head_y = _centre(head.box)
            scale = (_diagonal(tail.box) + _diagonal(head.box)) / 2.0
            if scale <= 0:
                continue
            distance = ((predicted[0] - head_x) ** 2 + (predicted[1] - head_y) ** 2) ** 0.5 / scale

            if distance <= max_distance and distance < best_cost:
                best_chain, best_cost = chain_id, distance

        if best_chain is None:
            chains[track_id] = dict(fragment)
        else:
            chains[best_chain].update(
                {
                    index: d.model_copy(update={"track_id": best_chain})
                    for index, d in fragment.items()
                }
            )
            merges += 1

    if merges:
        logger.info("stitched %d fragment(s) into earlier tracks", merges)
    return _to_frames(chains, frame_indices)


def interpolate_gaps(
    per_frame: Mapping[int, Sequence[TrackedDetection]], *, max_gap: int
) -> dict[int, list[TrackedDetection]]:
    """Fill frames inside a track where the detector produced nothing.

    Only *interior* gaps are filled: a track is never extended before its first
    frame or past its last, because there is no evidence about where the animal
    was outside the span it was actually seen.

    Every synthesised box is marked ``interpolated=True`` and takes the lower of
    the two surrounding confidences, so nothing downstream can mistake it for a
    detection.

    Args:
        per_frame: ``{frame_index: tracked detections}`` for one clip.
        max_gap: Largest run of missing frames to fill. ``0`` disables it.

    Returns:
        The same mapping with interior gaps filled.

    Raises:
        ValueError: If *max_gap* is negative.
    """
    if max_gap < 0:
        msg = f"max_gap must be >= 0, got {max_gap}"
        raise ValueError(msg)
    if max_gap == 0:
        return {index: list(detections) for index, detections in per_frame.items()}

    frame_indices = sorted(per_frame)
    tracks = _to_tracks(per_frame)
    added = 0

    for track in tracks.values():
        present = sorted(track)
        for start, end in pairwise(present):
            missing = end - start - 1
            if missing <= 0 or missing > max_gap:
                continue
            first, last = track[start], track[end]
            for step in range(1, missing + 1):
                weight = step / (missing + 1)
                track[start + step] = first.model_copy(
                    update={
                        "box": _blend(first.box, last.box, weight),
                        "confidence": min(first.confidence, last.confidence),
                        "interpolated": True,
                    }
                )
                added += 1

    if added:
        logger.info("interpolated %d box(es) across interior gaps", added)
    return _to_frames(tracks, frame_indices)


def _blend(first: BoundingBox, last: BoundingBox, weight: float) -> BoundingBox:
    """Linearly interpolate between two boxes."""

    def mix(a: float, b: float) -> float:
        return a + (b - a) * weight

    return BoundingBox(
        x_min=mix(first.x_min, last.x_min),
        y_min=mix(first.y_min, last.y_min),
        x_max=mix(first.x_max, last.x_max),
        y_max=mix(first.y_max, last.y_max),
    )


def smooth_tracks(
    per_frame: Mapping[int, Sequence[TrackedDetection]], *, window: int
) -> dict[int, list[TrackedDetection]]:
    """Average each track's box coordinates over a sliding window of frames.

    Changes no identity, no coverage and no frame membership -- only where the
    boxes sit. Steady motion is preserved because a centred moving average of a
    linear path is that same path; only frame-to-frame shake is removed, which is
    what :func:`~panaf_ape_detection.evaluation.tracking.measure_jitter` reports.

    The window stays **symmetric**: it shrinks equally on both sides at the ends
    of a track and wherever a frame is missing, rather than padding or reaching
    one-sidedly. A one-sided average is not a centred one -- it would pull the
    first and last boxes of every track inward and bend straight motion at the
    edges.

    Args:
        per_frame: ``{frame_index: tracked detections}`` for one clip.
        window: Frames to average over; must be odd. ``1`` disables smoothing.

    Returns:
        The same mapping with smoothed boxes.

    Raises:
        ValueError: If *window* is even or below 1. An even window has no centre,
            so it would shift every box half a frame forward in time.
    """
    if window < 1:
        msg = f"window must be >= 1, got {window}"
        raise ValueError(msg)
    if window % 2 == 0:
        msg = f"window must be odd so it has a centre frame, got {window}"
        raise ValueError(msg)
    if window == 1:
        return {index: list(detections) for index, detections in per_frame.items()}

    frame_indices = sorted(per_frame)
    tracks = _to_tracks(per_frame)
    half = window // 2

    for track in tracks.values():
        original = dict(track)
        for frame_index in list(track):
            # Grow a *symmetric* radius while both sides are still present. An
            # asymmetric window is not a centred average: at the ends of a track
            # it would drag the first and last boxes inward, and steady motion
            # would come out bent. Stopping at a missing frame is the same rule
            # that keeps the window from reaching across a gap.
            radius = 0
            while (
                radius < half
                and (frame_index - radius - 1) in original
                and (frame_index + radius + 1) in original
            ):
                radius += 1

            neighbours = [
                original[candidate]
                for candidate in range(frame_index - radius, frame_index + radius + 1)
            ]
            count = len(neighbours)
            track[frame_index] = track[frame_index].model_copy(
                update={
                    "box": BoundingBox(
                        x_min=sum(n.box.x_min for n in neighbours) / count,
                        y_min=sum(n.box.y_min for n in neighbours) / count,
                        x_max=sum(n.box.x_max for n in neighbours) / count,
                        y_max=sum(n.box.y_max for n in neighbours) / count,
                    )
                }
            )

    return _to_frames(tracks, frame_indices)


def refine(
    per_frame: Mapping[int, Sequence[TrackedDetection]],
    *,
    stitch_max_gap: int = 0,
    stitch_max_distance: float = 1.0,
    interpolate_max_gap: int = 0,
    smooth_window: int = 1,
) -> dict[int, list[TrackedDetection]]:
    """Apply the refinement steps in the one order that makes sense.

    ``stitch -> interpolate -> smooth``:

    * **Stitching first**, because it changes which boxes belong to the same
      track, and the other two operate per track. Interpolating first would fill
      toward a fragment's own end rather than across to its continuation.
    * **Interpolating before smoothing**, so the smoother sees a continuous run
      of frames and its window is not broken by holes that are about to be filled.

    ``drop_short_tracks`` belongs *between* the tracker and this, so fragments
    that stitch into one long track survive a length filter they would each have
    failed alone.

    Args:
        per_frame: ``{frame_index: tracked detections}`` for one clip.
        stitch_max_gap: See :func:`stitch_tracks`. ``0`` disables it.
        stitch_max_distance: See :func:`stitch_tracks`.
        interpolate_max_gap: See :func:`interpolate_gaps`. ``0`` disables it.
        smooth_window: See :func:`smooth_tracks`. ``1`` disables it.

    Returns:
        The refined mapping.
    """
    refined = stitch_tracks(per_frame, max_gap=stitch_max_gap, max_distance=stitch_max_distance)
    refined = interpolate_gaps(refined, max_gap=interpolate_max_gap)
    return smooth_tracks(refined, window=smooth_window)
