# Phase 1 Findings — Pretrained MegaDetector V6 on PanAf500

**Author:** Aditya Kothuri
**Date:** 2026-07-25
**Commit:** see `artifacts/metadata/` (run metadata records commit and dirty flag)
**Run metadata:** `artifacts/metadata/phase1-baseline_*.json`

---

## 1. Objective

Establish, without any fine-tuning, how reliably pretrained MegaDetector V6 localises great apes in
PanAf500 camera-trap clips, and under which conditions it fails — so that a decision about
fine-tuning rests on measurements rather than impressions.

## 2. Data and method

**Data.** 3 clips of a purposive 10-clip PanAf500 sample, 360 frames each (**1080 frames**,
1660 annotated ape boxes). Selection was not random: candidate annotations were profiled first and
clips chosen greedily to span behaviour variety, both species, crowded frames, subject size, and
frames containing no ape. Reasons per clip are in `data/sample_manifest.csv`.

**Method.** Every frame decoded, run through pretrained `MDV6-yolov9-c` via PyTorch-Wildlife,
filtered at confidence ≥ 0.2 and to the `animal` class, then matched greedily against ground truth
at IoU ≥ 0.5. **No fine-tuning.** Behaviour labels shown in the annotated video are dataset ground
truth, never model output.

## 3. Experimental setup

| | |
| --- | --- |
| Model | MegaDetectorV6 (PyTorch-Wildlife 1.3.0) |
| Variant | `MDV6-yolov9-c` |
| Confidence threshold | 0.20 |
| IoU threshold | 0.50 |
| Frame stride | 1 (every frame) |
| Tracker | none — deferred |
| Device | Apple MPS (verified; see §5.4) |
| Hardware / OS | Apple M1, macOS 15.5 |
| Clips processed | 3 of 10 |
| Total frames | 1080 |

## 4. What worked

**Precision is high: 0.854.** Of 799 detections, 682 landed on an annotated ape. When the detector
fires, it is usually right, and localisation is tight — **mean IoU 0.831** on matched pairs.

**False positives are rare.** Across 67 frames containing no ape at all, only **7** spurious
detections. The detector does not hallucinate apes in empty forest.

**It is excellent on clear, large, moving subjects.** Recall was 1.000 on `camera_interaction`
(34/34) and 0.889 on `running` (152/171).

## 5. Failure cases

### Overall

| Metric | Value |
| --- | --- |
| Precision | **0.854** |
| Recall | **0.411** |
| F1 | **0.555** |
| Mean IoU (matched) | 0.831 |
| TP / FP / FN | 682 / 117 / 978 |

**Recall is the problem, not precision.** The detector missed **978 of 1660** annotated apes.

### 5.1 Missed detections — by behaviour

| Behaviour | Found / total | Recall |
| --- | --- | --- |
| `hanging` | 0 / 213 | **0.000** |
| `climbing_up` | 3 / 177 | **0.017** |
| `sitting_on_back` | 17 / 154 | 0.110 |
| `climbing_down` | 10 / 55 | 0.182 |
| `walking` | 208 / 413 | 0.504 |
| `standing` | 258 / 443 | 0.582 |
| `running` | 152 / 171 | 0.889 |
| `camera_interaction` | 34 / 34 | 1.000 |

The gradient is stark and consistent: **arboreal postures fail almost completely** while
ground-level locomotion mostly succeeds. `hanging` was never detected once in 213 annotated
instances.

### 5.2 Missed detections — by subject size

| Size band (fraction of frame) | Found / total | Recall |
| --- | --- | --- |
| small (<2%) | 50 / 481 | **0.104** |
| medium (2–10%) | 447 / 711 | 0.629 |
| large (>10%) | 185 / 468 | 0.395 |

Small subjects fail as predicted. **The large-subject dip is confounded** — most large boxes in this
sample come from the one infrared night clip, so this row measures lighting as much as size. It
should not be read as "large subjects are hard" without a sample that separates the two.

### 5.3 False positives

117 overall; only 7 on frames with no ape. Most false positives are therefore **duplicate or
mislocalised boxes on frames that do contain an ape**, not invented animals.

### 5.4 Difficult conditions

| Condition | Observed effect | Evidence |
| --- | --- | --- |
| Infrared / night | **Catastrophic.** Clip `RHH9DDfWZa`: 3 detections in 360 frames, recall 0.009 | `artifacts/metrics/RHH9DDfWZa.json` |
| Occlusion (foliage, climbing) | Near-total failure on `hanging` and `climbing_up` | §5.1 |
| Distance / small subjects | Recall 0.104 | §5.2 |
| Multiple apes | Not isolated in this sample | — |
| Motion blur | Not isolated; `running` scored highest, so blur alone is not obviously fatal | §5.1 |

Note on the device: PyTorch-Wildlife accepts `device=` and silently ignores it, loading on CPU. The
adapter forces and then **verifies** placement by inspecting tensor devices; this run is confirmed on
`mps:0`. Any timing or device claim here rests on that check, not on what the library reported.

## 6. Comparison with provided behaviour labels

No behaviour was predicted — MegaDetector emits only `animal`. The comparison is an alignment check,
and it produced the clearest signal in this study: **detection success is strongly conditioned on the
ground-truth behaviour label** (§5.1), ranging from 0.000 to 1.000 recall across the nine classes.

An ape hanging in a tree is, to this detector, effectively invisible.

## 7. Three improvement ideas

1. **Lower the confidence threshold and re-measure.** Precision 0.854 with recall 0.411 says the
   operating point is badly mistuned for this footage — there is headroom to trade precision for
   recall. Cheap to test: `panaf-phase1 evaluate` recomputes from saved detections without re-running
   inference, so a sweep costs no GPU time.
2. **Compare variants before any fine-tuning.** `MDV6-yolov10-e` is larger and higher-resolution; if
   the small-subject and arboreal failures are capacity or resolution limits rather than domain
   mismatch, a variant swap is far cheaper than training. Only the config changes.
3. **If fine-tuning proceeds, target the failure modes, not the average.** The evidence points at
   arboreal postures, infrared, and small subjects — not at uniform weakness. A training set weighted
   toward `hanging` / `climbing_up` and night footage addresses the actual gap; more daylight
   ground-level footage would mostly reinforce what already works.

## 8. Conclusion

Pretrained MegaDetector V6 is **precise but insensitive** on PanAf500: it rarely cries wolf
(precision 0.854, 7 false positives across 67 empty frames) but misses roughly **three of every five
annotated apes** (recall 0.411). Failure is not uniform — it concentrates sharply in arboreal
postures (`hanging` 0.000, `climbing_up` 0.017), infrared night footage (one clip at recall 0.009),
and small distant subjects (0.104).

The main limitation is sample size: **3 clips, purposively chosen to be hard.** These numbers
describe these clips, not PanAf500. The most useful next step is the cheapest one — a confidence
threshold sweep over the detections already saved — before spending anything on training.

## 9. Figure

![MegaDetector on daylight gorilla footage: two detections at 0.92 and 0.74 matching ground truth,
while the distant gorilla at top-left (small amber box) is missed. Green = prediction, amber =
dataset ground truth. Clip FgJpFLxSmH frame 100, MDV6-yolov9-c at confidence 0.2.](figures/FgJpFLxSmH_frame100.png)

The same failure the numbers describe: near subjects found confidently, the distant one missed.

---

### Limitations statement

This used a small, purposively selected sample of 3 PanAf500 clips chosen to span expected failure
conditions, so the observations are not an unbiased estimate of performance on PanAf500. All figures
are at confidence 0.20 and IoU 0.50; different thresholds give different numbers. MegaDetector
detects `animal` / `person` / `vehicle` — it does not identify species, individuals, or behaviour, so
a detection counts as correct here only when it **localises** an annotated ape. Behaviour labels
shown in any annotated output are dataset ground truth, not model predictions.
