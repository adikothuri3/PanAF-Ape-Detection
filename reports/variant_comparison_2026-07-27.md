# Which detector should we use? A head-to-head test

**Date:** 2026-07-27 · **Author:** Aditya Kothuri
**Numbers from:** `artifacts/colab/sweep-conf005/` and `artifacts/colab/variant-yolov10e/`

---

## The short version

We swapped one line of configuration — the model variant — and **found nearly twice as many apes,
with no loss of accuracy in the boxes we drew.**

| At the same settings | Old model (`yolov9-c`) | New model (`yolov10-e`) |
| --- | --- | --- |
| How often a box we draw is really an ape | 87% | 86% |
| **How many of the apes present we actually find** | **39%** | **74%** |
| Combined score (F1) | 0.54 | **0.80** |

The cost was 22 seconds of extra compute on the whole dataset. Nothing was retrained.

---

## What was tested, in plain terms

A detector looks at a video frame and draws boxes around animals. Two things can go wrong: it can
**miss** an ape that is there, or it can **draw a box** where there is no ape.

Our first model was cautious. When it drew a box it was almost always right (87% of the time), but
it missed about three of every five apes. Missing is the expensive failure here — you cannot track,
pose-estimate or study an animal you never detected.

MegaDetector ships in several sizes. We had been using `MDV6-yolov9-c`, one of the smaller ones.
This test ran the larger `MDV6-yolov10-e` over **exactly the same 10 clips, the same 3,600 frames,
the same 4,985 labelled apes**, on the same Colab A100, in the same session. Only the model changed.

## What happened

**It finds far more apes, and the boxes are no worse.**

Recall — the share of real apes found — went from **39% to 74%**. Precision — how often our boxes are
correct — stayed essentially the same, 87% to 86%. That is unusual and worth saying plainly: normally
you buy recall by accepting more false boxes. Here we did not have to.

**The hardest footage improved the most.**

The clips we had written off as near-hopeless are where the gain landed:

| Clip | What it looks like | Apes found before | After |
| --- | --- | --- | --- |
| `RHH9DDfWZa` | infrared night footage | 19% | **100%** |
| `z97mIEQzcL` | chimp in deep shade | 55% | **100%** |
| `XmOoOk9n7t` | backlit, background blown out white | 85% | **100%** |
| `ICIHqd95OX` | daylight, distant subjects | 44% | 82% |

**Every behaviour improved.** The dataset labels what each ape is doing. Previously the detector was
close to blind on apes up trees; now it is merely worse than average on them:

| What the ape is doing | Found before | Found after |
| --- | --- | --- |
| sitting | 33% | **100%** |
| climbing up | 18% | **70%** |
| hanging | 53% | 74% |
| walking | 65% | 90% |
| standing | 67% | 94% |
| running | 91% | 98% |

**Small, distant apes improved too** — from 44% to 71% found — so this is not only a "big obvious
animal" effect.

## The one place it got worse

On `isfRigsIjO`, a very dark clip, the new model finds **fewer** apes (49% → 36%). But look at what
it was doing before: the old model drew **728 boxes** on a clip containing one ape per frame, and
only 24% of them were right. It was scattering boxes into the dark and getting lucky. The new model
draws 164 boxes and 80% of them are correct.

So it is not really a regression — it is a model that stopped guessing. That clip remains our worst
case and is the honest limit of what a pretrained detector does with near-darkness.

## What this does for tracking

Tracking follows each ape from frame to frame and gives it a stable ID. It can only follow what was
detected, so it was capped by the old model's misses.

| | Old model | New model |
| --- | --- | --- |
| Share of each ape's time on screen we successfully follow | 35% | **73%** |
| Individuals followed for most of their screen time (of 23) | 6 | **13** |
| Individuals we essentially lost (of 23) | 9 | **2** |
| Times we confused one ape for another | 19 | 46 |

Coverage roughly doubled, and we now lose only 2 individuals instead of 9. The cost is more identity
confusion — 46 mix-ups instead of 19 — because there are simply more apes being tracked at once, and
because two apes close together are easy to swap. That is a tracking problem to solve next, not a
detection problem.

## A bonus: the threshold problem mostly went away

Every detection carries a confidence score, and we choose a cut-off below which we ignore boxes.
With the old model that choice was painful: our default cut-off of 0.20 was badly wrong for this
footage, and performance kept improving as we lowered it — which meant we never found the right
setting.

The new model performs best **at the default 0.20**, and degrades gently either side of it:

| Cut-off | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.05 | 0.64 | 0.82 | 0.72 |
| 0.10 | 0.75 | 0.79 | 0.77 |
| **0.20** | **0.86** | **0.74** | **0.80** |
| 0.30 | 0.91 | 0.71 | 0.80 |
| 0.50 | 0.97 | 0.63 | 0.76 |

A model that is well-behaved at its default setting is much easier to trust and to hand to someone
else.

## What it cost

121 seconds instead of 99 for all 10 clips on an A100 — about 22% slower. Storage and code are
unchanged. **No training, no new dependencies, one line of configuration.**

## Honest limitations

- **10 clips, chosen to be difficult.** These numbers describe these clips, not all of PanAf500.
- **A detection counts as correct if it lands on a labelled ape.** MegaDetector reports only
  "animal" — it does not identify species, individuals, or behaviour. Behaviour labels come from the
  dataset.
- **More detections means more identity confusion.** Reported above rather than buried.
- **Not yet re-run end to end.** The comparison used saved detections; the annotated videos in the
  repo still come from the old model.

---

## Related

[Phase 1 findings](phase1_findings_2026-07-26.md) · [experiment log](../experiments/experiment_log.md) ·
[model notes](../docs/obsidian/05%20Technical/model.md)
