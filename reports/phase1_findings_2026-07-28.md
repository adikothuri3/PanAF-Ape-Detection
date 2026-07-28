# Phase 1 Findings — MegaDetector V6 and ByteTrack on the whole of PanAf500

**2026-07-28. Supersedes [`phase1_findings_2026-07-26.md`](phase1_findings_2026-07-26.md),**
which measured 10 purposively-selected clips. This one measures all **500 clips, ~180,000 frames,
201,430 annotated ape boxes and 874 annotated individuals** — the entire densely-annotated subset,
and therefore the entire universe available for this task.

Every number here traces to a file under `artifacts/`. Nothing is estimated.

---

## 1. Headline

The Phase 1 pipeline — decode → MegaDetector V6 → ByteTrack → refine → annotated video — over the
full dataset, against the configuration that shipped before this work:

| | before | after |
| --- | --- | --- |
| **Detection precision** (tracked output) | 0.7940 | **0.8547** |
| **Detection recall** (tracked output) | 0.8504 | **0.8593** |
| **F1** | 0.8212 | **0.8570** |
| Mean IoU on matched pairs | 0.8366 | 0.8380 |
| **Identity coverage** | 0.7397 | **0.8230** |
| **ID switches** | 2257 | **301** |
| Fragmentation (tracks per ape) | 2.48 | **1.27** |
| Track purity | 0.9969 | 0.9944 |
| Tracks holding 2+ apes | 57 | 56 |
| **Jitter** (box shake) | 0.0153 | **0.0035** |
| Mostly tracked (≥80% of frames) | 625 / 874 | **645 / 874** |
| Mostly lost (≤20% of frames) | 65 / 874 | 74 / 874 |

Better on almost every axis. The two that got worse — purity by 0.0025, and nine more
mostly-lost apes — are discussed in §6 rather than left out.

No training, no fine-tuning, no new dependency. Pretrained inference, configuration, and two
post-processing steps.

---

## 2. The finding that reframed the work

Early on, tracking coverage over the 10-clip sample was **0.7149** and detection recall was
**0.7149**. Equal to four decimal places, and not a coincidence: every true-positive detection was
already inside some track, and no track ever covered a frame the detector missed.

Coverage was pinned to the detector by construction. No tracker setting could raise it. The
remaining problem was never coverage — it was **identity**, and the only way past the ceiling was
to synthesise boxes the detector never produced.

That is why the work below is about ID switches, fragmentation and interpolation, and not about
finding more animals.

---

## 3. Detection alone

Single-frame accuracy of the raw detector, `MDV6-yolov10-e`, over all 500 clips. The detections
were cached once at confidence 0.05, so every higher threshold is a free re-score of the same run.

| confidence | precision | recall | F1 |
| --- | --- | --- | --- |
| 0.05 | 0.4683 | 0.9223 | 0.6212 |
| 0.10 | 0.5979 | 0.9001 | 0.7185 |
| 0.20 | 0.7271 | 0.8715 | 0.7928 |
| 0.30 | 0.8073 | 0.8438 | 0.8251 |
| **0.40** | 0.8636 | 0.8080 | **0.8349** |
| 0.50 | 0.9029 | 0.7599 | 0.8253 |
| 0.60 | 0.9289 | 0.7033 | 0.8005 |
| 0.70 | 0.9520 | 0.6354 | 0.7621 |

> **Correction.** This repository asserted in `configs/base.yaml`, `configs/colab.yaml`, the
> experiment log and a check-in note that `MDV6-yolov10-e` peaks at confidence **0.20**. It does
> not: over the full dataset the single-frame peak is **0.40**.
>
> The old figure was wrong twice over. It came from 10 purposively-hard clips, *and* it was
> measured on a run with tracking enabled — so `drop_short_tracks` had already deleted false
> positives before the detector was scored. That inflated precision at low thresholds and moved
> the apparent optimum. Comparing a detector against itself requires the tracker to be off, which
> is why every detector number in this section comes from a detector-only run.

---

## 4. The pipeline beats any threshold

The pipeline is not a detector, and its output is not the detector's output: short tracks have
been dropped, gaps interpolated, positions smoothed. Measured on the *tracked* boxes, over the
same 500 clips:

| | precision | recall | F1 |
| --- | --- | --- | --- |
| Best possible single-frame threshold (0.40) | 0.8636 | 0.8080 | 0.8349 |
| Old pipeline — detect 0.20 + old tracker | 0.7940 | 0.8504 | 0.8212 |
| **Adopted — detect 0.05 + tuned tracker** | **0.8547** | **0.8593** | **0.8570** |

**Detect generously and let the tracker filter.** Raw precision at 0.05 is 0.4683 — more than half
the boxes are wrong. After tracking it is 0.8547, at higher recall than any threshold achieves, and
above the 0.8349 ceiling of the best single-frame operating point.

The reason is that a threshold and a tracker filter on different evidence. A false positive at 0.15
is one frame of flicker: it cannot start a track (activation 0.45 means a new identity needs a 0.55
box) and it cannot survive `minimum_track_length: 8`. A *real* ape at 0.15, sitting between two
confident frames of the same animal, is absorbed into the track it belongs to. Temporal consistency
is information a per-frame threshold has no access to.

This inverts the usual advice to tune the detector threshold for precision. With a tracker
downstream, the threshold's job is to avoid destroying data — and boxes below it are never written
to disk, so anything discarded there is unrecoverable without paying for the GPU again.

---

## 5. Tracking

874 annotated individuals. Metrics are defined in
[`docs/obsidian/05 Technical/tracking.md`](../docs/obsidian/05%20Technical/tracking.md); the ones
that matter here are **identity coverage** (the share of an ape's frames held by its *single best*
track) and **track purity** (the share of a track's frames belonging to one ape).

| | before | after |
| --- | --- | --- |
| Identity coverage | 0.7397 | **0.8230** |
| Coverage (any track) | 0.8504 | 0.8593 |
| ID switches | 2257 | **301** |
| Fragmentation | 2.48 | **1.27** |
| Jitter | 0.0153 | **0.0035** |

### What actually mattered

**Activation threshold, raised from 0.20 to 0.45.** The largest single effect. Because
`supervision` requires `activation + 0.10` to *start* a track, this means only a 0.55 detection
opens a new identity while anything down to the score floor can extend an existing one. Spurious
track creation was the main source of fragmentation.

**Association threshold, held at 0.8 rather than 0.9.** A sweep on 10 clips preferred 0.9. Over 500
clips that turned out to be the single biggest cause of two apes being fused into one track:

| `minimum_matching_threshold` | 0.7 | 0.8 | 0.9 |
| --- | --- | --- | --- |
| mean tracks holding 2+ apes | 46.7 | 54.9 | 71.9 |

I had predicted the long `lost_track_buffer` would be the culprit. It is not — buffer 30 → 120
moves merged tracks by about three. **Permissive association fuses animals; a long memory does
not.**

**Working around a hardcoded floor.** `supervision` discards every detection scoring at or below
`0.1` — a literal in its source, not a parameter — which at confidence 0.05 is 18.6% of all
detections, thrown away by the low-score association pass that is the entire reason ByteTrack was
chosen. A monotone affine map lifts scores into `[0.11, 1]` before association and inverts exactly
afterwards, so no artifact ever records a rescaled score.

**Interpolation is the only thing that raises coverage** above detection recall, because it is the
only step that produces a box where the detector produced none. Every synthesised box is flagged
`interpolated` in the artifacts, so Phase 2 can exclude boxes no detector ever saw.

**Smoothing cut jitter by 77% and slightly raised both precision and recall** — steadier boxes clear
the IoU 0.5 threshold more often. I expected it to be cosmetic.

**Stitching turned out to be redundant, and that is a result.** It demonstrably works — on the
untuned baseline it cuts tracks 59 → 46 and fragmentation 2.57 → 2.00 — but it changes nothing at
the adopted settings, because a 4-second lost-track buffer already reconnects the same fragments,
using a motion model inside the tracker rather than a repair afterwards. It ships disabled.

---

## 6. What it cost

**Nine more apes are mostly lost** (65 → 74 of 874). Activation 0.45 is strict: a faint animal that
never produces a 0.55 detection now never gets a track started at all. This is the direct price of
the fragmentation win and it is not recoverable by any other setting in the sweep.

**Track purity fell 0.9969 → 0.9944.** Small, and merged tracks actually went *down* (57 → 56), so
the pipeline is not fusing more animals than before. But it is worth stating that identity coverage
can be raised by merging two apes — both then count the merged track as their dominant one — which
is why the sweep was run under a hard ceiling on merged tracks rather than on identity coverage
alone. 46 of 72 arms cleared that ceiling.

**`sitting_on_back` got worse**, 0.227 → 0.207 recall. An ape riding on another's back is the most
occluded case in the dataset, and the stricter activation threshold means the rider rarely reaches
the score needed to open its own identity.

---

## 7. Where it still fails

Recall of the adopted pipeline, by the dataset's behaviour label:

| behaviour | found / annotated | recall |
| --- | --- | --- |
| sitting_on_back | 504 / 2430 | **0.207** |
| climbing_up | 2473 / 4098 | 0.603 |
| climbing_down | 940 / 1455 | 0.646 |
| hanging | 6428 / 8814 | 0.729 |
| camera_interaction | 1560 / 1939 | 0.805 |
| walking | 53116 / 62515 | 0.850 |
| sitting | 68384 / 76117 | 0.898 |
| standing | 37745 / 41934 | 0.900 |
| running | 1942 / 2128 | 0.913 |

And by how much of the frame the ape occupies:

| size | found / annotated | recall |
| --- | --- | --- |
| small | 44655 / 62790 | 0.711 |
| medium | 87054 / 94107 | 0.925 |
| large | 41383 / 44533 | 0.929 |

The pattern the project predicted from the start holds, now on the full dataset. The failures are
**occlusion and scale**, not species or site:

* **`sitting_on_back` at 0.207** is the hardest case by a wide margin. Two animals overlapping
  almost completely read as one.
* **The arboreal postures** — `climbing_up` 0.603, `climbing_down` 0.646, `hanging` 0.729 — are the
  next worst. Unusual poses against cluttered canopy.
* **Small subjects at 0.711** against 0.925 and 0.929 for medium and large. A distant ape is a
  handful of pixels.

MegaDetector is a general animal detector: its whole output space is `{animal, person, vehicle}`.
It has never seen a label for "chimpanzee hanging from a branch". That these are the failures is
expected; the value is in having them quantified over 201,430 boxes rather than assumed.

---

## 7b. Held out: the dataset's own splits

PanAf500 ships a split — **400 train / 25 validation / 75 test** — and everything above pools all
500 clips. The tracker settings were selected by a sweep that saw all of them, so those pooled
figures are not a held-out estimate.

Re-running the selection on the **400 training clips only**, under a merge ceiling recomputed on
train alone, picks **exactly the same settings**: activation 0.45, buffer 120, matching 0.8,
minimum track length 8. The choice never depended on seeing the test clips, so the test numbers
below are a fair estimate of settings that would have been chosen without them.

| split | | identity coverage | ID switches | fragmentation | merged tracks | jitter |
| --- | --- | --- | --- | --- | --- | --- |
| **train** (400 clips, 677 apes) | legacy | 0.7425 | 1709 | 2.41 | 48 | 0.0156 |
| | adopted | **0.8198** | **201** | **1.26** | **39** | **0.0036** |
| **validation** (25 clips, 60 apes) | legacy | 0.6567 | 206 | 3.30 | 4 | 0.0183 |
| | adopted | **0.7733** | **40** | **1.45** | 8 | **0.0042** |
| **test** (75 clips, 137 apes) | legacy | 0.7641 | 342 | 2.50 | 5 | 0.0129 |
| | adopted | **0.8612** | **60** | **1.28** | 9 | **0.0030** |

**The identity result holds, and is strongest on held-out data.** On the 75 test clips identity
coverage improves by **+9.7pp** against +7.7pp on train, and ID switches fall 82%.

> [!warning] The merge regression did not fully disappear on unseen data
> In aggregate merged tracks went 57 → 56, which cleared the ceiling. That was carried by the
> training split, where they fell 48 → 39. On the held-out splits they rose: **5 → 9 on test and
> 4 → 8 on validation.**
>
> As a rate on test that is 1.5% of tracks before against 5.1% after — a tripling, on small absolute
> counts (9 of roughly 175 tracks). So the honest statement is not "the regression was fixed" but
> "it was fixed on the data it was tuned against, and roughly tripled in rate on data that was not".
> The identity gain is large enough that the trade is still clearly worth making, and the numbers
> are here so a reader can disagree.

## 8. Honest limitations

**A third of the tracking gain was overfitting.** The settings were first tuned on 10 clips, where
they appeared to lift identity coverage by **+11.1pp**. Over 500 clips the same settings gave
**+7.6pp**. Two thirds of the apparent gain was real and one third was fitting the tuning set —
which is exactly what validating on unseen clips exists to discover. The adopted settings were then
re-chosen on the full dataset and give +8.3pp.

**The 10 tuning clips are inside the 500.** About 2% contamination. Negligible, but real.

**One corpus.** PanAf500 is 14 field sites and one camera-trap setup. Nothing here shows the
settings transfer to other footage.

**No training was done here, but MegaDetector's own training data is not fully public.** This
project runs pretrained inference only — no optimiser, no gradients, nothing fitted to these clips.
Whether PanAf footage appeared in the corpus MegaDetector V6 was originally trained on cannot be
verified from the published material, so it cannot be ruled out. If it did, the detection numbers
would be optimistic in a way no split of this dataset can detect. The *tracking* results are
unaffected either way: the tracker is not a learned model, and its settings were chosen here, on
data whose split is known.

**PanAf20K cannot extend this.** It is ~20,000 videos, but only this 500-clip subset carries
per-frame boxes and `ape_id` identities. Without identity ground truth, ID switches and
fragmentation are not expensive to compute — they are undefined. 500 clips is the ceiling, not a
compromise.

**Interpolated boxes are synthesised.** They are predictions, scored as predictions, and flagged in
the artifacts. Any downstream stage that needs real image evidence must exclude them.

**Behaviour labels come from the dataset, never the model.** MegaDetector cannot predict behaviour,
and the annotated videos show the dataset's label beside the box precisely so a viewer cannot
mistake one for the other.

---

## 9. Conclusion

Tracking is no longer the bottleneck. 301 ID switches across 874 individuals, fragmentation 1.27,
and a tracked F1 of 0.8570 that exceeds what any single-frame threshold can reach.

Detection is the limit again, and now the limit is characterised: occlusion and scale, concentrated
in `sitting_on_back`, the arboreal postures, and small subjects. That is a much better-founded case
for what to do next than the project had a week ago, and it rests on the whole dataset rather than
ten hand-picked clips.

The case for fine-tuning is correspondingly weaker than it looked. Tracking was the bottleneck and
configuration fixed it, without training anything. If fine-tuning is revisited, the table in §7 is
what it would have to improve — and `sitting_on_back` at 0.207 is where the headroom is.

---

## Reproducing this

```bash
# All 500 clips, detector-only, confidence 0.05 — one Colab A100 session, ~100 min.
python scripts/fetch_panaf500.py --all
panaf-phase1 detect --config configs/colab-full500.yaml

# Everything below is CPU, over the cache above.
panaf-phase1 track --config configs/base.yaml \
    --detections-dir artifacts/full500/detections \
    --metrics-dir artifacts/full500-adopted --write-detections
panaf-phase1 evaluate --config configs/base.yaml \
    --detections-dir artifacts/full500-adopted/detections --pooled-only

# The before/after, on identical inputs.
panaf-phase1 track --config configs/tracking-legacy.yaml \
    --detections-dir artifacts/full500/detections \
    --metrics-dir artifacts/full500-legacy --detection-floor 0.20
```

The sweep that chose the settings: `configs/sweeps/around-candidate.yaml`, 72 arms, ranked by
identity coverage under `--max-merges 59`. Its full record, including the arms that were rejected,
is written to `artifacts/metrics/tracking-sweep/`.

## The showcase clips

Three clips rendered from the shipped configuration on Apple MPS — `ICIHqd95OX`, `FgJpFLxSmH` and
`9uIpm1xLeI`, the crowded ones that held most of the original ID switches:

```bash
panaf-phase1 detect --config configs/base.yaml   # over a manifest of those three clips
```

They land in `artifacts/showcase/videos/`. Over those 10 individuals: **9 ID switches, 71
interpolated boxes of 1844, jitter 0.0060.** Rendering them is also what exposed the last bug in
this work — see below.

**They are not committed, deliberately.** Annotated footage is a derived work of a
non-commercially-licensed dataset, and this repository never redistributes it. Green is the
prediction, amber is the dataset's ground truth and behaviour label; the legend is drawn on every
frame so a still pulled out of the video is unambiguous about which box came from where.

## A bug this work found by looking at its own output

The showcase clips were rendered twice. The first attempt produced **jitter 0.0316 against 0.0035
dataset-wide, and zero boxes flagged `interpolated`** — which should have been impossible with
`interpolate_max_gap: 24` and `smooth_window: 5` in the config.

Stitching, interpolation and smoothing were applied only on the re-tracking path. `track` measured
one pipeline; `detect` produced artifacts from another. Every refinement setting in
`configs/base.yaml` was parsed, validated, and then ignored by the command that writes the
detections cache and the annotated video. Anyone cloning the repository and running `detect` would
have got numbers that did not match this report.

Both paths now call one `finalise_tracks()`, and a test asserts `detect` produces interpolated
boxes when the configuration asks for them. Re-rendered: jitter 0.0060, 71 interpolated boxes.

None of the measurements in this report were affected — all of them came through `track`, which did
refine, and the 500-clip cache is detector-only so refinement never applied to it. But it is the
third time in this project that a value was accepted, stored and never applied, after
PyTorch-Wildlife's `device=` and its `det_conf_thres`. The lesson keeps being the same one:
**verify by observing behaviour, never by the absence of an exception.**
