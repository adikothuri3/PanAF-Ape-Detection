---
tags: [reference, dataset, verified]
status: verified
source: Geologic Dome Intern Onboarding.pdf (July 2026)
updated: 2026-07-24
---

# PanAf500 Action Labels

**Nine action labels**, listed verbatim in the onboarding document:

| # | Label |
|---|---|
| 1 | sitting |
| 2 | standing |
| 3 | walking |
| 4 | running |
| 5 | climbing up |
| 6 | climbing down |
| 7 | hanging |
| 8 | sitting on back |
| 9 | camera interaction |

Alongside these, PanAf500 provides **frame-by-frame boxes, individual IDs, and species**.

## Provenance

Source: the onboarding document, which states PanAf500 "has frame-by-frame boxes, individual IDs,
species, and 9 action labels" and then names all nine. This is a **verified** list — it is why
[[Phase 1 Task Spec|step 5]] (compare detections against the dataset's action label) is
well-defined.

## What is still unverified

The **label list** is verified. The **annotation file format** is not:

- Which file holds these labels, and in what encoding
- Whether labels are per-frame, per-track, or both
- The exact strings used in the files (`climbing up` vs `climbing_up` vs an integer id)
- How multiple simultaneous behaviours are represented, if at all

Do not code against a guessed schema. Confirm by opening the real files, then record the answer
here and in [dataset docs](../docs/dataset.md).

## Why this matters for Phase 1

Step 5 displays these labels **beside** detections. They are **dataset ground truth, never model
predictions** — MegaDetector cannot predict behaviour, and any annotated clip must make that
distinction visible. See [[MegaDetector Variants]] and [model docs](../docs/model.md).

Note also that several of these labels name exactly the conditions expected to break a detector:
`climbing up`/`climbing down` and `hanging` imply occlusion and unusual poses, and
`camera interaction` means a subject filling or exceeding the frame.

## Related

[[Phase 1 Task Spec]] · [[MegaDetector Variants]] · [[PanAf20K Paper]] · [dataset docs](../docs/dataset.md)
