---
tags: [onboarding, phase-1, task-spec]
status: active
source: Geologic Dome Intern Onboarding.pdf (July 2026)
updated: 2026-07-24
---

# Phase 1 Task Spec — "See"

**Goal.** Run detection and tracking on real PanAf footage, overlay the behaviour labels, and write
up what you find.

## The six steps

Status below reflects **what actually exists in this repository**, not what is planned. Steps 1-5
are done and measured, and step 6 is written from those measurements.

| # | Step | What it asks for | Status |
|---|---|---|---|
| 1 | **Set up** | GitHub account; Python via uv or conda; VS Code or Cursor. No personal GPU needed — use Google Colab's free GPU for model work. | ✅ **Done** — uv + Python 3.11, locked env, [repo published](https://github.com/adikothuri3/PanAF-Ape-Detection), [Colab scaffold](../../../notebooks/README.md) |
| 2 | **Get the data** | Download a small slice of **PanAf500**: frame-by-frame boxes, individual IDs, species, and 9 action labels. Start with **5 to 10 clips**, not all 7 million frames. | ✅ **Done** — 10 clips, purposively selected, checksummed manifest; see [data/README.md](../../../data/README.md) |
| 3 | **Detect** | Run PyTorch-Wildlife (MegaDetector) on the clips, draw boxes on each frame, stitch back into an annotated video. | ✅ **Done** — `MDV6-yolov9-c`, every frame, annotated MP4 per clip |
| 4 | **Track** | Give each ape a stable ID across frames with a simple tracker — **SORT or ByteTrack**. | ✅ **Done** — ByteTrack via `supervision`; ID switches and fragmentation measured against `ape_id` |
| 5 | **Compare** | Show the dataset's action label next to your detections. **Does what you see match the label?** | ✅ **Done** — labels drawn beside detections; recall broken down by label |
| 6 | **Write it up** | One page: what worked, what failed (missed detections, ID switches, dark frames), and **three ideas** to make it better. | ✅ **Done** — [findings, 10 clips](../../../reports/phase1_findings_2026-07-26.md) |

The 9 action labels are listed in [[dataset|the nine action labels]].

## Deliverable

1. A **GitHub repo** — code plus a README.
2. **2 to 3 annotated clips or GIFs.**
3. The **one-page write-up**.

## Done means

> Someone else can clone your repo, follow the README, and reproduce one annotated clip.

This is the acceptance test. It is stricter than "the code runs on my machine", and it is why the
repo has a locked environment, a checksummed manifest and a
[reproducibility contract](../05%20Technical/reproducibility.md).

## Stretch

If the above is solid, run an **animal pose model (DeepLabCut or ViTPose) on one clip** to get
skeletons. That is the on-ramp to Phase 2 — see [[Four Phase Arc]].

## What this phase is not

Worth stating because the tooling makes it easy to overclaim:

- MegaDetector outputs `animal` / `person` / `vehicle`. It does **not** identify species,
  individuals, or behaviour — see [[model|MegaDetector variants]] and [model docs](../05%20Technical/model.md).
- Behaviour labels come from the **dataset**, never from the model. Step 5 is a *comparison*, not a
  prediction task.
- No fine-tuning. Phase 1 is pretrained inference only.

## Related

[[Four Phase Arc]] · [[How We Work]] · [[dataset|the nine action labels]] · [[Reading List]] · [[PanAf Command Center]]
