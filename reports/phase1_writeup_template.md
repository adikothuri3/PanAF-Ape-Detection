# Phase 1 Findings — <TITLE>

<!--
TEMPLATE. Copy to reports/phase1_findings_YYYY-MM-DD.md before filling in.
Do not fill this file in directly, and do not delete the TODO markers here.

Target: approximately one page once complete. Cut prose, not evidence.

Rules:
  * Every number must come from a recorded run. If you did not measure it, write "not measured".
  * State the model variant and confidence threshold wherever you state a result.
  * Describe observations about specific clips, not properties of PanAf500 as a whole.
  * The sample is 5-10 purposively chosen clips. It cannot support a generalisation. Say so.
-->

**Author:** TODO
**Date:** TODO
**Commit:** TODO (clean / dirty: TODO)
**Run metadata:** TODO — path(s) under `artifacts/metadata/`

---

## 1. Objective

TODO — one short paragraph. What question did this phase attempt to answer, and why that question?

_Reference: "Without any fine-tuning, how reliably does pretrained MegaDetector V6 localise great
apes in PanAf500 camera-trap clips, and under which conditions does it fail?"_

## 2. Data and method

**Data.** TODO — how many clips, from which subset, selected on what basis. Reference the manifest
rather than listing checksums here.

**Method.** TODO — one paragraph: frame extraction, pretrained detection, confidence filtering,
tracking, behaviour-label overlay. State plainly that no fine-tuning was performed and that
behaviour labels come from the dataset, not from the model.

## 3. Experimental setup

| | |
| --- | --- |
| Model | TODO |
| Variant | TODO (e.g. `MDV6-yolov9-c`) |
| Confidence threshold | TODO |
| Frame stride | TODO |
| Tracker | TODO (backend, or "none") |
| Device | TODO |
| Hardware / OS | TODO |
| Clips processed | TODO |
| Total frames | TODO |

## 4. What worked

TODO — be specific and bounded. "Detections were stable on clip X in daylight at threshold T", not
"the model worked well".

## 5. Failure cases

### 5.1 Missed detections

TODO — where the detector produced no box for a visibly present ape. Cite clip and frame ranges.
Estimate how often, and say how the estimate was made.

### 5.2 False positives

TODO — boxes with no animal. What was the detector firing on?

### 5.3 ID switches

TODO — tracker identity changes on the same individual, and identities reused across individuals.
Note whether the cause looked like detection dropout or association failure. Write "tracking not
implemented" if it was not run.

### 5.4 Difficult conditions

Fill in only what was observed. Leave a row as "not observed in this sample" rather than inventing
an outcome.

| Condition | Observed effect | Clips / evidence |
| --- | --- | --- |
| Darkness / infrared | TODO | TODO |
| Occlusion (vegetation, other apes) | TODO | TODO |
| Distance / small subjects | TODO | TODO |
| Motion blur | TODO | TODO |
| Multiple overlapping individuals | TODO | TODO |

## 6. Comparison with provided behaviour labels

TODO — the detector does not predict behaviour, so this is an alignment check, not an accuracy
measure. Did boxes appear on the individual the behaviour label refers to? Were there labelled
individuals with no detection? Did any behaviour class coincide with systematically worse detection
(e.g. climbing, hanging, camera interaction)?

State explicitly that no behaviour prediction was performed.

## 7. Three improvement ideas

Each should be actionable and follow from evidence above, not from general knowledge.

1. **TODO** — what, why (which observation motivates it), and how it would be tested.
2. **TODO** — same.
3. **TODO** — same.

## 8. Conclusion

TODO — three to five sentences. Answer the research question directly, including "we could not
determine X" where that is the honest answer. State the main limitation. State the single most
useful next step.

## 9. Figure / table

_One compact figure or table. A representative failure case is usually worth more than a montage of
successes. Regenerate from a recorded run; never edit the output by hand._

<!-- Place in reports/figures/ and reference it here:
![TODO caption stating clip id, variant and threshold](figures/TODO.png)
-->

TODO

---

### Limitations statement (do not delete)

This phase used a small, purposively selected sample of PanAf500 clips chosen to span expected
failure conditions. It does not constitute a quantitative evaluation of MegaDetector V6 on PanAf500,
and the observations above should not be read as accuracy estimates. MegaDetector detects
`animal` / `person` / `vehicle`; it does not identify species, individuals, or behaviour. Behaviour
labels shown in any annotated output are dataset ground truth, not model predictions.
