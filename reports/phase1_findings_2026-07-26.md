# Phase 1 Findings — Pretrained MegaDetector V6 and ByteTrack on PanAf500

**Author:** Aditya Kothuri
**Date:** 2026-07-26
**Commit:** see `artifacts/metadata/` (run metadata records commit and dirty flag)
**Run metadata:** `artifacts/metadata/phase1-baseline_*.json`

> Supersedes the 3-clip findings of 2026-07-25. Two conclusions from that write-up **did not survive
> the full sample** and are corrected in §5.1 and §5.2. The earlier numbers were correct for the
> clips they described; they were not representative.
>
> **§5 describes the detector at confidence 0.20. §8 shows that threshold was the single biggest
> problem with this baseline** — at 0.05 the "catastrophic" clips are largely recovered. Read §5 as
> *what one operating point does*, not as the detector's ceiling.

---

## 1. Objective

Establish, without any fine-tuning, how reliably pretrained MegaDetector V6 localises great apes in
PanAf500 camera-trap clips, whether a simple tracker can hold their identities, and under which
conditions each fails — so that a decision about fine-tuning rests on measurements rather than
impressions.

## 2. Data and method

**Data.** All 10 clips of a purposive PanAf500 sample, every frame: **3600 frames, 4985 annotated
ape boxes, 23 annotated individuals**. Every clip was run three times — detector-only at 0.20,
detector + ByteTrack at 0.20, and detector-only at 0.05 for the threshold sweep (§8). Selection was not random — candidate annotations were
profiled first and clips chosen greedily to span behaviour variety, both species, crowded frames,
subject size, and frames containing no ape. Reasons per clip are in `data/sample_manifest.csv`.

**Method.** Every frame decoded, run through pretrained `MDV6-yolov9-c` via PyTorch-Wildlife,
filtered at confidence ≥ 0.20 and to the `animal` class, then matched greedily against ground truth
at IoU ≥ 0.50. **No fine-tuning.** Behaviour labels shown in the annotated video are dataset ground
truth, never model output.

**Two passes, deliberately.** The pipeline was run twice over the same 10 clips:

| Pass | Configuration | Output |
| --- | --- | --- |
| **A — detector only** | `tracking.enabled: false` | `artifacts/no-tracking/` |
| **B — detector + ByteTrack** | `tracking.enabled: true`, `minimum_track_length: 5` | `artifacts/` |

This matters. With tracking on, `drop_short_tracks` removes tracks under five frames *before*
metrics are computed, so pass B measures the detector **and** the tracker acting as a temporal
filter. Reporting only pass B as "MegaDetector's accuracy" would be wrong. Running both isolates the
tracker's contribution instead of confounding it (§7).

## 3. Experimental setup

| | |
| --- | --- |
| Model | MegaDetectorV6 (PyTorch-Wildlife 1.3.0) |
| Variant | `MDV6-yolov9-c` |
| Confidence threshold | 0.20 |
| IoU threshold | 0.50 |
| Frame stride | 1 (every frame) |
| Tracker | ByteTrack (`supervision`), activation 0.20, buffer 30 frames, match 0.80, 24 fps |
| Minimum track length | 5 frames |
| Device | Apple MPS, **verified** by inspecting tensor placement (§5.5) |
| Hardware / OS | Apple M1, macOS 15.5 |
| Clips processed | 10 of 10 |
| Total frames | 3600 |
| Wall clock | ~20 min per pass |

## 4. What worked

**Precision is high: 0.874 detector-only, 0.917 with tracking.** Of 2203 detections, 1926 landed on
an annotated ape, and localisation is tight — **pooled mean IoU 0.825** over matched pairs.

**False positives on empty forest are rare.** Across **74 frames containing no ape at all**, only
**7** spurious detections detector-only, and **3** with tracking. The detector does not invent apes
in empty forest.

**It is reliable on clear, ground-level subjects.** Recall 1.000 on `camera_interaction` (34/34),
0.828 on `running` (775/936), 0.634 on `standing` (422/666).

**Where an ape is detected at all, ByteTrack holds it.** Mean fragmentation is **1.35 predicted
tracks per annotated individual** (ideal is 1.00), across 23 individuals and 29 tracks — see §6.

## 5. Failure cases

### Overall

| Metric | Detector only | + ByteTrack |
| --- | --- | --- |
| Precision | 0.8743 | **0.9165** |
| Recall | **0.3864** | 0.3525 |
| F1 | **0.5359** | 0.5091 |
| Pooled mean IoU (matched) | 0.8252 | 0.8322 |
| TP / FP / FN | 1926 / 277 / 3059 | 1757 / 160 / 3228 |
| FP on the 74 empty frames | 7 | 3 |

**Recall is the problem, not precision.** The detector missed **3059 of 4985** annotated apes.

### 5.1 Missed detections — by behaviour

Detector-only, pooled over 10 clips:

| Behaviour | Found / total | Recall |
| --- | --- | --- |
| `climbing_up` | 64 / 776 | **0.082** |
| `hanging` | 137 / 1346 | **0.102** |
| `sitting_on_back` | 17 / 154 | 0.110 |
| `sitting` | 12 / 86 | 0.140 |
| `climbing_down` | 54 / 200 | 0.270 |
| `walking` | 411 / 787 | 0.522 |
| `standing` | 422 / 666 | 0.634 |
| `running` | 775 / 936 | 0.828 |
| `camera_interaction` | 34 / 34 | 1.000 |

The gradient is real and consistent: **arboreal and seated postures fail badly while ground-level
locomotion mostly succeeds.**

> **Correction to the 3-clip write-up.** That report recorded `hanging` at **0.000 (0/213)** and
> called it "not degradation, blindness". Over 1346 instances the true figure is **0.102** — bad,
> but not blindness. The earlier zero was one clip's worth of an unusually hard condition, and
> generalising it was wrong. The ranking held; the absolute claim did not.

### 5.2 Missed detections — by subject size

| Size band (fraction of frame) | Found / total | Recall |
| --- | --- | --- |
| small (<2%) | 706 / 1979 | 0.357 |
| medium (2–10%) | 889 / 1462 | 0.608 |
| large (>10%) | 331 / 1544 | **0.214** |

**Read this row by row and it says large subjects are hardest. That conclusion is false**, and the
per-clip data shows why:

| Clip | Large boxes | Recall on them | Character |
| --- | --- | --- | --- |
| `FgJpFLxSmH` | 119 | **0.899** | daylight, gorillas, mixed distances |
| `1xDXmshd5P` | 90 | **0.856** | daylight, chimpanzees, mixed distances |
| `XmOoOk9n7t` | 360 | 0.361 | single chimp hanging, blown-out backlight |
| `RHH9DDfWZa` | 259 | 0.004 | infrared night |
| `z97mIEQzcL` | 356 | 0.045 | single chimp climbing/hanging, deep shade |
| `isfRigsIjO` | 360 | **0.000** | single chimp hanging, near-dark |

In the two clips where large and small subjects appear *together*, large subjects are detected
**~0.87 of the time** — the best of any band. The pooled 0.214 is a composition effect: 1335 of the
1544 large boxes come from four single-ape, close-up, badly-lit tree clips. **The failure is
lighting and contrast, not size.** This is the same confound flagged in the 3-clip report, and with
ten clips it is now measurable rather than suspected.

### 5.3 False positives

277 overall detector-only; only 7 on frames with no ape. Most false positives are therefore
**duplicate or mislocalised boxes on frames that do contain an ape**, not invented animals.

### 5.4 Difficult conditions

| Condition | At confidence 0.20 | At 0.05 (§8) | Evidence |
| --- | --- | --- | --- |
| Infrared / night | `RHH9DDfWZa`: 3 detections in 360 frames, recall **0.009** | recall **0.265** | `artifacts/*/metrics/RHH9DDfWZa.json` |
| Dark subject on dark ground | `isfRigsIjO`: **0 detections in 360 frames** | recall **0.433** | §11, figure 2 |
| Blown-out backlight | `XmOoOk9n7t`: recall 0.361 | recall **0.892** | §11, figure 3 |
| Arboreal posture + foliage | `climbing_up` 0.082, `hanging` 0.102 | 0.174, **0.527** | §5.1, §8 |
| Distance / small subjects | recall 0.357 | 0.440 | §5.2 |
| Motion blur | Not isolated; `running` scored highest, so blur alone is not obviously fatal | — | §5.1 |

The four worst clips share one property: **a dark animal against a background of similar luminance**,
whether the frame is dark overall (`isfRigsIjO`), infrared (`RHH9DDfWZa`) or backlit
(`XmOoOk9n7t`).

**But the right column is the finding.** These are not frames the detector cannot see — they are
frames it scores *quietly*, between 0.05 and 0.20, where the threshold then discards them. Low
contrast depresses confidence rather than destroying detection. A conclusion of "catastrophic
failure on night footage" drawn from the 0.20 column alone would have been wrong, and would have
pointed the next month of work at the wrong problem.

### 5.5 Device

PyTorch-Wildlife accepts `device=` and silently ignores it, loading on CPU. The adapter forces and
then **verifies** placement by inspecting tensor devices; both passes are confirmed on `mps:0`. Any
timing or device claim here rests on that check, not on what the library reported.

## 6. Tracking

Measured against the dataset's `ape_id`, over the same 10 clips:

| Metric | Value |
| --- | --- |
| Annotated individuals | 23 |
| Predicted tracks | 29 |
| Total ID switches | **17** |
| Mean fragmentation (tracks per individual) | **1.35** (max 4) |
| Pooled coverage of annotated individual-frames | **0.353** |
| Mostly tracked (≥80% covered) | 7 |
| Mostly lost (≤20% covered) | 9 |

**Fragmentation is low; coverage is not.** ByteTrack does not lose apes it can see — 1.35 tracks per
individual, and in the best clip (`FgJpFLxSmH`) four of five individuals are mostly tracked, three of
them above 0.92 coverage. But pooled coverage 0.353 is almost exactly detection recall (0.386):
**a tracker cannot associate a box that was never produced.** Three clips produced **no tracks at
all**, because they produced almost no detections.

The nine mostly-lost individuals split two ways, and the distinction matters: five are in the
badly-lit clips of §5.4, where nothing was detected; the other four are in daylight clips
(`1xDXmshd5P` ×3, `FgJpFLxSmH` ×1) and are the **distant** apes — the same small-subject failure as
§5.2, arriving as a tracking failure because a track needs detections to exist first.

Ten of the 17 ID switches fall in the two most crowded clips — `FgJpFLxSmH` (5 apes, 6 switches) and
`9uIpm1xLeI` (3 apes, 4 switches). That is the expected pattern: a switch needs two apes close
enough to confuse, and these are the clips with several apes in frame at once.

**MOTA and IDF1 are deliberately not reported.** Both are easy to get subtly wrong, and a metric
nobody has hand-checked is not a measurement. ID switches, fragmentation and coverage are each
tested against worked examples.

## 7. What tracking cost and bought

Tracking with `minimum_track_length: 5` removed **286 detections** — 117 false positives and 169
true positives:

| | Detector only | + ByteTrack | Δ |
| --- | --- | --- | --- |
| Precision | 0.8743 | 0.9165 | **+0.042** |
| Recall | 0.3864 | 0.3525 | **−0.034** |
| F1 | 0.5359 | 0.5091 | −0.027 |
| FP on empty frames | 7 | 3 | −4 |

The removed detections are **3.3× enriched in false positives** relative to the base rate (41% of
removals versus 12.6% of all detections), so the filter is doing real work rather than thinning at
random. But on this footage it trades away more recall than it gains precision, and F1 falls.

**Given that recall is the binding constraint, `minimum_track_length: 5` is too aggressive here.**
It is a config value, not a code change, and `panaf-phase1 track` re-runs tracking over saved
detections in seconds, so this is cheap to revisit.

### 7.1 ByteTrack has a hard floor at 0.1 — and it cancels the threshold win

§8 shows detector recall rises 0.386 → 0.563 at confidence 0.05. **None of that reaches the
tracker.** Tracking the 0.05 detections gives coverage 0.767 / 0.301 / 0.308 on the three clips that
track at all — the same, to three decimals, as at 0.20 — with *more* ID switches (32 tracks and more
switches versus 29). The clips that produced no tracks still produce none.

The cause is two hardcoded values in `sv.ByteTrack`, neither configurable:

```python
inds_low = scores > 0.1                                    # <= 0.1 is discarded outright
self.det_thresh = self.track_activation_threshold + 0.1    # new tracks need activation + 0.1
```

So lowering `track_activation_threshold` does **not** make the tracker consider everything the
detector kept — this module's original assumption, now corrected in its docstring. There is a floor:
nothing at or below 0.1 can ever start a track.

That floor lands exactly where the recovered detections are:

| Clip | Detections @ 0.05 | ≤ 0.10 (discarded) | 0.10–0.15 (cannot start a track) | > 0.15 (usable) |
| --- | --- | --- | --- | --- |
| `isfRigsIjO` | 742 | 649 (87%) | 81 | **12 (2%)** |
| `RHH9DDfWZa` | 98 | 73 | 20 | **5 (5%)** |
| `z97mIEQzcL` | 369 | 237 | 84 | 48 (13%) |
| `FgJpFLxSmH` | 716 | 80 | 39 | 597 (83%) |

**The low-confidence detections that rescue the hard clips are precisely the ones ByteTrack refuses
to use.** For a detection-only deliverable the lower threshold is a large win; for anything
downstream of the tracker it is worth nothing. Which of those matters is a decision about what
consumes the output, not about the detector.

Verified with `panaf-phase1 track --min-track-length 1|2|3|5` over the saved 0.05 detections: the
zero-track clips stay at zero for every value, so this is ByteTrack's floor, not
`drop_short_tracks`.

## 8. The confidence threshold — measured, not assumed

**This is the most important result in the study, and it required fixing a bug to get.**

The adapter filtered detections at `model.confidence_threshold` *after* inference but never passed
it to the model, so every run inferred at PyTorch-Wildlife's own default of 0.2 regardless of
configuration (§8.1). With that fixed, all 10 clips were re-run at 0.05 and the threshold swept over
the saved detections:

| Confidence | Precision | Recall | F1 |
| --- | --- | --- | --- |
| **0.05** | 0.6283 | **0.5633** | **0.5940** |
| 0.10 | 0.7914 | 0.4558 | 0.5784 |
| 0.15 | 0.8571 | 0.4116 | 0.5562 |
| **0.20** *(the operating point used above)* | 0.8743 | 0.3864 | 0.5359 |
| 0.25 | 0.8868 | 0.3661 | 0.5182 |
| 0.30 | 0.9030 | 0.3472 | 0.5016 |
| 0.40 | 0.9176 | 0.3081 | 0.4613 |
| 0.50 | 0.9426 | 0.2800 | 0.4318 |

**F1 rises monotonically as the threshold falls, and has not turned over at 0.05** — the best
operating point in this range is the lowest one tested, and the true optimum may be lower still.
Recall improves 46% relative (0.386 → 0.563) for a precision cost of 0.874 → 0.628.

### Where the recovered detections are

Not spread evenly — **concentrated almost entirely in the clips §5.4 called catastrophic**:

| Clip | Recall @ 0.20 | Recall @ 0.05 | Δ |
| --- | --- | --- | --- |
| `XmOoOk9n7t` — backlit, blown out | 0.361 | 0.892 | **+0.531** |
| `z97mIEQzcL` — deep shade | 0.044 | 0.492 | **+0.447** |
| `isfRigsIjO` — near-dark | **0.000** | 0.433 | **+0.433** |
| `RHH9DDfWZa` — infrared night | 0.009 | 0.265 | **+0.255** |
| `DGevz8OvXl` — daylight | 0.565 | 0.671 | +0.106 |
| `ICIHqd95OX` — daylight | 0.328 | 0.431 | +0.103 |
| `1xDXmshd5P` — daylight | 0.299 | 0.401 | +0.102 |
| `9uIpm1xLeI` — daylight | 0.772 | 0.864 | +0.092 |
| `zvwY5xoIli` — daylight | 0.217 | 0.265 | +0.048 |
| `FgJpFLxSmH` — daylight | 0.673 | 0.720 | +0.047 |

By behaviour, `hanging` goes **0.102 → 0.527** and `sitting` 0.140 → 0.360. By size, the large band —
the one §5.2 showed was really the badly-lit clips — goes **0.214 → 0.601**.

**So the detector was not blind to these apes. It was quiet about them.** Low contrast depresses
confidence; the threshold then throws the detection away. That is an operating-point problem, not a
capability gap, and it is fixed by editing one line of YAML rather than by training anything.

### 8.1 The bug this depended on

`YOLOV8Base.single_image_detection(img, det_conf_thres=0.2)`. The adapter called it without that
argument. Consequences:

- Every run before this one inferred at 0.2, whatever the config said. Because the configured value
  *was* 0.2, **every number in §4–§7 is still correct** — the two thresholds agreed by coincidence.
- A sweep below 0.2 was impossible and silently returned identical results. The first attempt at
  this experiment produced byte-identical output to the 0.20 run — 2203 detections, minimum
  confidence 0.2002 — which is how the bug was caught.

This is the **second** value PyTorch-Wildlife accepts and does not apply, after `device=`. The fix
passes the configured threshold to every inference call; three weights-free tests pin it, including
one asserting the recorded default still matches the installed library's signature.

## 9. Three improvement ideas

1. **Move the operating point to 0.05 — and sweep below it.** §8 measures a 46% relative recall gain
   for one YAML edit and no training. The curve has not turned over, so the next run should extend
   the sweep downward. Everything else in this list is more expensive than this. **But see §7.1:**
   this helps the detector, not the tracker, so the right threshold depends on which output is the
   product. If tracks are the product, the binding constraint is ByteTrack's hardcoded 0.1 floor, and
   the question becomes whether to replace the tracker rather than retune the detector.
2. **Then re-examine contrast preprocessing.** §5.4 identified low contrast as the common factor, and
   §8 shows the model already responds to those subjects — just weakly. CLAHE or histogram
   equalisation might lift those detections' scores rather than create them, which is a smaller and
   better-understood claim than "it recovers missed apes". Worth testing, no longer the first thing
   to try.
3. **Compare variants before fine-tuning.** `MDV6-yolov10-e` is larger and higher-resolution. Only
   the config changes. Fine-tuning should be considered only after the free levers are exhausted, and
   §8 shows one of them was still untouched.

## 10. Conclusion

Pretrained MegaDetector V6 is **precise but insensitive** on PanAf500: it rarely cries wolf
(precision 0.874, 7 false positives across 74 empty frames) but misses roughly **three of every five
annotated apes** (recall 0.386). ByteTrack holds identities well where detections exist —
fragmentation 1.35, 17 switches over 23 individuals — but covers only 0.353 of annotated
individual-frames, because coverage is capped by detection recall.

Failure is **not uniform**: it concentrates in **low-contrast footage** — night, infrared, deep
shade, blown-out backlight — with arboreal posture and small size as correlates rather than causes.

**But "insensitive" is a property of the operating point, not of the model.** At confidence 0.05 the
same weights on the same frames reach recall 0.563 and F1 0.594, and the gain lands almost entirely
on the clips that looked hopeless at 0.20: the near-dark clip goes from 0 detections in 360 frames to
recall 0.433, `hanging` from 0.102 to 0.527. The detector was scoring these apes quietly, not missing
them.

Three claims in this study's own lineage turned out to be artifacts rather than findings — the
3-clip `hanging` "blindness", the "large subject dip", and (nearly) a pooled mean-IoU collapse that
was really an averaging bug. Each looked like a result. **The pattern is worth more than any single
number here: a plausible failure story is easy to construct and expensive to act on.**

The main limitation remains sample size: **10 clips, purposively chosen to be hard.** These numbers
describe these clips, not PanAf500.

## 11. Figures

![Two gorillas detected at 0.92 and 0.74 with tracker ids #2 and #6, matching ground truth, while
the distant gorilla at top-left (small amber box) is missed. Green = prediction, amber = dataset
ground truth. Clip FgJpFLxSmH frame 100.](figures/FgJpFLxSmH_frame100_tracked.png)

**Figure 1 — what working looks like.** Track ids are stable across the clip; the distant third
gorilla is missed. This is the small-subject failure, visible.

![A near-black frame: a chimpanzee hanging in a tree, labelled by the dataset, with no prediction
box anywhere. Clip isfRigsIjO frame 180.](figures/isfRigsIjO_frame180.png)

**Figure 2 — what a badly-chosen threshold looks like.** The ape fills a third of the frame and a
human sees it immediately. At confidence 0.20 the detector produced nothing in all 360 frames. At
0.05, on the same frames, recall on this clip is **0.433** — the detections existed all along,
scoring below the threshold.

![An over-exposed infrared frame: a chimpanzee hanging, dark against blown-out white foliage, with
no prediction box. Clip XmOoOk9n7t frame 180.](figures/XmOoOk9n7t_frame180.png)

**Figure 3 — the same effect from the opposite direction.** Backlight blows out the background; the
ape is again a silhouette of near-uniform luminance against its surroundings. Recall on this clip
goes from 0.361 at confidence 0.20 to **0.892** at 0.05 — the largest single-clip gain in the study.

---

### Limitations statement

This used a small, purposively selected sample of 10 PanAf500 clips chosen to span expected failure
conditions, so the observations are not an unbiased estimate of performance on PanAf500. All figures
are at confidence 0.20 and IoU 0.50; different thresholds give different numbers. MegaDetector
detects `animal` / `person` / `vehicle` — it does not identify species, individuals, or behaviour, so
a detection counts as correct here only when it **localises** an annotated ape. Behaviour labels
shown in any annotated output are dataset ground truth, not model predictions. Track ids are the
tracker's own and are meaningful only within a single clip; they are compared against the dataset's
`ape_id` but never claimed to be individual identification.
