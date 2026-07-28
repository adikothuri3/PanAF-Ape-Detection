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

## Measured results — the whole dataset

Chosen by a 72-arm sweep over all **500 clips / 874 annotated individuals**, ranked by identity
coverage under a hard ceiling on merged tracks. 46 arms cleared the ceiling; the best is what
`configs/base.yaml` now ships. `configs/tracking-legacy.yaml` holds the settings it replaced, so
the comparison stays reproducible.

| | legacy | adopted |
| --- | --- | --- |
| Identity coverage | 0.7397 | **0.8230** |
| Coverage | 0.8504 | 0.8593 |
| ID switches | 2257 | **301** |
| Fragmentation | 2.48 | **1.27** |
| Track purity | 0.9969 | 0.9944 |
| Tracks holding 2+ apes | 57 | 56 |
| Jitter | 0.0153 | **0.0035** |
| Mostly tracked | 625/874 | 645/874 |
| Mostly lost | 65/874 | 74/874 |
| Tracked precision | 0.7940 | **0.8547** |
| Tracked recall | 0.8504 | **0.8593** |
| Tracked F1 | 0.8212 | **0.8570** |

Adopted settings: activation 0.45, lost-track buffer 120, matching threshold 0.8, minimum track
length 8, score floor 0.11, interpolation up to 24 frames, 5-frame smoothing, stitching off.

### What mattered, and what did not

- **Activation threshold** did the most, raised from 0.20 to 0.45. Since supervision needs
  `activation + 0.10` to *start* a track, only a 0.55 detection opens an identity while anything
  above the score floor can extend one. Spurious track creation was the fragmentation.
- **Matching threshold drives merged tracks**, and nothing else comes close. Mean merged tracks
  across the sweep were **46.7 / 54.9 / 71.9** at 0.7 / 0.8 / 0.9, while lost-track buffer 30 → 120
  moved them by about three. A 10-clip sweep had preferred 0.9; on the full dataset that is what
  fuses two animals into one identity. Permissive association merges apes — a long memory does not.
- **Detect generously, filter with the tracker.** Raw precision at confidence 0.05 is 0.4683, but
  the tracked output reaches 0.8547 at higher recall — above the 0.8349 F1 ceiling of the best
  possible single-frame threshold. See [[model]].
- **Stitching is redundant here.** It works (untuned baseline: tracks 59 → 46, fragmentation
  2.57 → 2.00) but changes nothing at these settings, because the 4-second buffer already
  reconnects the same fragments with a motion model rather than a repair afterwards. Ships off.

### Held out

PanAf500 ships a 400 / 25 / 75 train / validation / test split. Re-running the selection on the
training clips alone picks **exactly the same settings**, so the choice did not depend on seeing
the test clips. On those 75 held-out clips (137 apes):

| | legacy | adopted |
| --- | --- | --- |
| Identity coverage | 0.7641 | **0.8612** |
| ID switches | 342 | **60** |
| Fragmentation | 2.50 | **1.28** |
| Merged tracks | 5 | 9 |

The identity gain is **larger** on held-out data than on train (+9.7pp against +7.7pp). The merge
count is not: it fell 48 → 39 on train but rose 5 → 9 on test and 4 → 8 on validation. The
aggregate 57 → 56 was carried by the training split, and the honest reading is that the regression
was fixed where it was tuned and roughly tripled in rate where it was not.

### What it cost

Nine more apes are mostly lost, 65 → 74. Activation 0.45 is strict, so a faint animal that never
reaches 0.55 never gets a track started. `sitting_on_back` recall fell 0.227 → 0.207 for the same
reason — a rider is the most occluded case in the dataset.

Purity fell 0.9969 → 0.9944, though merged tracks went *down* (57 → 56), so the pipeline is not
fusing more animals than before.

> [!note] A third of the first result was overfitting
> Settings tuned on 10 clips appeared to lift identity coverage by **+11.1pp**. The same settings
> over 500 clips gave **+7.6pp**. That gap is what validating on unseen clips exists to find, and
> it is reported rather than quietly dropped. The shipped settings were re-chosen on the full
> dataset and give +8.3pp.

Full write-up: [phase1 findings 2026-07-28](../../../reports/phase1_findings_2026-07-28.md).

## Related

- [[architecture]] — where tracking sits, and the artifacts contract
- [[model]] — the detector whose recall bounds coverage
- [[dataset]] — `ape_id`, and why PanAf500 is the only usable subset
- [[reproducibility]] — run metadata and provenance
