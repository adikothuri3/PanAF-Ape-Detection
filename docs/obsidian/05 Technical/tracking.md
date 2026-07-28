---
tags: [technical, tracking, phase1]
status: in-progress
updated: 2026-07-27
---

# Tracking

What the tracking stage does, what constrains it, and what has been measured. Detection is
covered in [[model]]; the artifact contract is in [[architecture]].

## The finding that reframed the stage

Frame-weighted tracking coverage on the 10-clip baseline was **0.7149**. Detection recall was
**0.7149**. Equal to four decimal places, and not a coincidence:

> Every true-positive detection was already inside some track, and no track covered a frame the
> detector missed.

Two things follow, and they set the whole agenda:

1. **Tuning the tracker cannot raise coverage.** Coverage is pinned to the detector by
   construction, because association can only ever link boxes that exist.
2. **The remaining problem is identity.** 54 predicted tracks for 23 real individuals, 36 ID
   switches, 64% of them in three multi-ape clips.

The only mechanism that can lift coverage above detection recall is **interpolation** — inventing a
box in a frame the detector missed — which is why it is treated as a prediction and flagged as
such rather than quietly counted.

## What ByteTrack will and will not let you do

Implementation is `supervision`'s ByteTrack, via
`panaf_ape_detection.tracking.bytetrack.ByteTrackTracker`. Five real knobs; before this work three
were unreachable from configuration and the fourth was welded to the detector's threshold.

Two limits are **literals in the upstream source**, not parameters:

```python
inds_low = scores > 0.1                                  # <= 0.1 discarded outright
self.det_thresh = self.track_activation_threshold + 0.1  # a new track needs activation + 0.1
```

Measured consequence on the 10-clip cache at confidence 0.05: **1,185 of 6,384 detections (18.6%)
are thrown away** before either association pass — by the low-score pass that is the entire reason
ByteTrack was chosen over SORT.

`ScoreFloor` works around the first. It maps scores into `[floor, 1]` on the way in and inverts
exactly on the way out, so ByteTrack sees boxes it would have discarded while **no artifact ever
records a rescaled score**. The activation threshold is mapped through the same transform, so it
keeps meaning a detector score.

One more upstream subtlety: `lost_track_buffer` is expressed at 30 fps and rescaled,
`max_time_lost = int(fps / 30 * buffer)`. At PanAf's 24 fps a buffer of 30 is **24 frames, 1.0 s**
— not the 1.25 s the code claimed until it was checked against the installed source.

## Measuring it honestly

`panaf_ape_detection.evaluation.tracking` scores predictions against the dataset's per-frame
`ape_id`. MOTA and IDF1 remain deliberately unimplemented — see the module docstring.

| Metric | What it catches |
| --- | --- |
| ID switches | The matched track id changed while the ape was covered |
| Fragmentation | Distinct tracks one ape was spread across; 1.00 ideal |
| Coverage | Share of an ape's frames any track covered |
| **Identity coverage** | Share held by its *single best* track — the tuning objective |
| **Track purity** | Share of a track's frames belonging to one ape |
| **ID merges** | Tracks matched to more than one ape |
| Jitter | Normalised box acceleration; steady motion scores 0 |

**Purity and merges exist because every other metric on that list rewards a specific wrong
answer.** Merge two apes into one track and the switches vanish, fragmentation falls to a perfect
1.00, and coverage does not move. Any step that joins tracks together can therefore score better by
being more wrong. `tests/test_tracking.py` pins this with the worked example.

**Identity coverage** is the objective because it cannot be gamed in either direction: splitting an
ape across tracks lowers it while coverage stays flat, and a track cannot hold two apes' frames at
once. Coverage minus identity coverage is the fragmentation tax.

## Refinement

`panaf_ape_detection.tracking.refine` — three pure functions, applied
`stitch → drop_short_tracks → interpolate → smooth`.

- **Stitching** joins fragments that do not overlap in time, whose motion extrapolates to where the
  next fragment begins, at compatible box sizes. The temporal-overlap prohibition is the safety
  rule: two tracks visible in the same frame are two animals, and no distance test may override it.
- **Interpolation** fills interior gaps only, never extending a track past its own span. Every
  synthesised box is marked `interpolated=True`, so Phase 2 pose work can exclude boxes no detector
  ever saw.
- **Smoothing** is a symmetric centred average. Symmetry matters: a one-sided window at the ends of
  a track would drag the first and last boxes inward and bend straight motion.

## Measured results — validated on all 500 clips

Tuned on 10 clips, then measured on the whole dataset: **500 clips, 874 annotated individuals**,
detector-only cache at confidence 0.05. Shipped settings versus
[`configs/tracking-candidate.yaml`](../../../configs/tracking-candidate.yaml):

| | shipped | candidate |
| --- | --- | --- |
| Identity coverage | 0.7436 | **0.8197** |
| Coverage | 0.8621 | 0.8709 |
| ID switches | 1910 | **409** |
| Fragmentation | 2.65 | **1.30** |
| Track purity | 0.9970 | 0.9913 |
| Tracks holding 2+ apes | 59 | **79** |
| Jitter | 0.0161 | **0.0037** |
| Mostly tracked | 640/874 | 670/874 |
| Mostly lost | 59/874 | 65/874 |

**The gain is real and it was oversold.** On the 10 tuning clips the margin was +11.1pp identity
coverage; on 500 it is **+7.6pp**. Roughly a third of the apparent improvement was fitting the
tuning set. Both arms also score higher in absolute terms here, because the 10 clips were
purposively chosen to be hard.

Three results worth keeping separate from the table:

- **Activation threshold did the most work**, raised to 0.40 — twice the detector threshold. Only a
  strong detection should open a new identity; faint ones should extend an existing track. A
  108-arm sweep over activation 0.10–0.30 never beat it (best 0.8018 against 0.8197).
- **Detect permissively, let the tracker filter.** Raw precision at confidence 0.05 is 0.64, but the
  *tracked* output beats the 0.20 pipeline at higher recall. The tracker discriminates using
  temporal consistency, which a per-frame threshold has no access to.
- **Stitching is redundant here, and that is a result.** It demonstrably works (untuned baseline:
  tracks 59 → 46, fragmentation 2.57 → 2.00) but changes nothing at the tuned operating point,
  because `lost_track_buffer: 120` already reconnects the same fragments — with a motion model
  inside the tracker rather than a repair afterwards.

> [!warning] One regression, and it is the one that matters
> Tracks holding two or more apes rose **59 → 79**, purity 0.9970 → 0.9913. As a rate: 2.5% of
> shipped tracks against 7.0% of candidate tracks. On the 10-clip sample merges had *improved*
> (3 → 2), so nothing before the full run hinted at it.
>
> Merging two animals into one track is the failure identity coverage can partly **reward** — both
> apes then count the merged track as dominant. The rule fixed before measuring was "maximise
> identity coverage, subject to merges not exceeding baseline", and by that rule the candidate
> fails. `track-sweep --max-merges` now enforces the ceiling rather than printing the column and
> hoping someone checks it.
>
> Open: whether a setting near the candidate keeps most of the +7.6pp while holding merges at 59.
> [`configs/sweeps/around-candidate.yaml`](../../../configs/sweeps/around-candidate.yaml) answers
> it, on CPU, over the cache already on disk. If nothing clears the ceiling, adopt the candidate
> and report the merge cost — do not pick settings that hide it.

## Related

- [[architecture]] — where tracking sits, and the artifacts contract
- [[model]] — the detector whose recall bounds coverage
- [[dataset]] — `ape_id`, and why PanAf500 is the only usable subset
- [[reproducibility]] — run metadata and provenance
