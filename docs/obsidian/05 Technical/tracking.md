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

## Measured results — 10 clips, pending validation

Staged sweep over the detector-only cache at confidence 0.05
(`artifacts/colab/variant-yolov10e/`), shipped settings versus
[`configs/tracking-candidate.yaml`](../../../configs/tracking-candidate.yaml):

| | shipped | candidate |
| --- | --- | --- |
| Identity coverage | 0.6449 | **0.7561** |
| Coverage | 0.7284 | 0.7639 |
| ID switches | 46 | **4** |
| Fragmentation | 2.57 | **1.13** |
| Track purity | 0.9963 | 0.9975 |
| Tracks holding 2+ apes | 3 | 2 |
| Jitter | 0.0230 | **0.0042** |
| Mostly tracked | 13/23 | 16/23 |
| Detection precision | 0.9530 | 0.9556 |
| Detection recall | 0.7246 | 0.7639 |

Three results worth keeping separately from the table:

- **Activation threshold did the most work**, raised to 0.40 — twice the detector threshold. Only a
  strong detection should open a new identity; faint ones should extend an existing track. Spurious
  track creation was the main source of fragmentation.
- **Detect permissively, let the tracker filter.** Raw precision at confidence 0.05 is 0.64, but the
  *tracked* output is 0.9556 — better than the 0.9308 the 0.20 pipeline reported, at higher recall.
- **Stitching turned out to be redundant here, and that is a result rather than an oversight.** It
  demonstrably works (on the untuned baseline: tracks 59 → 46, fragmentation 2.57 → 2.00) but
  changes nothing at the tuned operating point, because `lost_track_buffer: 120` already reconnects
  the same fragments — using a motion model inside the tracker rather than a repair afterwards.

> [!warning] Not adopted
> Every number above was tuned **and** measured on the same 10 clips, which were purposively chosen
> to be hard. That is the circumstance under which a tuned result means least. The candidate stays
> out of `base.yaml` until it is confirmed on clips it was never tuned on;
> [`configs/colab-full500.yaml`](../../../configs/colab-full500.yaml) produces them, and PanAf500's
> own train/validation/test split is what to divide by.
>
> PanAf500 is the ceiling for this, not a compromise: only that subset carries per-frame `ape_id`,
> so on the rest of PanAf20K these metrics are undefined rather than merely expensive. See
> [[dataset]].

## Related

- [[architecture]] — where tracking sits, and the artifacts contract
- [[model]] — the detector whose recall bounds coverage
- [[dataset]] — `ape_id`, and why PanAf500 is the only usable subset
- [[reproducibility]] — run metadata and provenance
