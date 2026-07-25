---
tags: [reference, glossary]
status: active
updated: 2026-07-24
---

# Glossary

Terms that appear across the onboarding, the dataset and the robotics stack. Kept short; where a
term has a fuller treatment elsewhere, this links there rather than restating it.

## Animals and the dataset

**Great ape** — chimpanzees, gorillas, bonobos, orangutans. The onboarding is explicit that these
are **great apes, not monkeys**. The IUCN lists every great ape species as Endangered or Critically
Endangered.

**PanAf20K** — the full dataset: ~20,000 coarsely annotated camera-trap videos, >7 million frames,
14 African field sites.

**PanAf500** — the densely annotated 500-video subset with per-frame boxes, identity tracks and
behaviour labels. **The only subset Phase 1 uses.** See [[PanAf500 Action Labels]].

**Camera trap** — a fixed camera triggered by motion. It records whatever passes, so some clips
contain no animal at all — which is what makes false-positive behaviour measurable.

**Tracklet** — a short sequence of boxes belonging to one individual across consecutive frames.

## Detection and tracking

**MegaDetector** — a general animal/person/vehicle detector for camera-trap imagery. Not a species
classifier. See [[MegaDetector Variants]].

**Confidence threshold** — the score above which a detection is kept. Trades recall against
precision continuously; a detection count is meaningless without it.

**ID switch** — a tracker assigning a new identity to the same individual, or reusing an identity
across individuals. A primary Phase 1 failure mode.

**SORT / ByteTrack** — the two simple trackers the onboarding names for
[[Phase 1 Task Spec|step 4]]. The choice is deliberately deferred until detections are known to be
stable.

**NMS (non-maximum suppression)** — post-processing that removes overlapping duplicate boxes. It
cannot distinguish two genuinely overlapping apes from one duplicated detection, which is why dense
groups are a known failure case.

## Pose, prediction and robotics

**Pose estimation** — locating an animal's joints to build a skeleton, turning movement into joint
data. [[Four Phase Arc|Phase 2]]; see [[DeepLabCut]].

**World model** — a model that predicts what happens next, rather than classifying what is happening
now. [[Four Phase Arc|Phase 3]]; see [[World Models]].

**Retargeting** — mapping motion from one body onto a different one. The onboarding names this
directly as the hard part of [[Four Phase Arc|Phase 4]]: a climbing, four-limbed ape onto a bipedal
humanoid.

**Unitree G1** — the humanoid robot platform. **Pemba** is Geologic Dome's G1; see
[[Geologic Dome Context]].

**MuJoCo** — the physics simulator the G1 is driven in. See [[MuJoCo]].

**DimensionalOS (dimos)** — the humanoid and quadruped stack that runs on Pemba, shipping a built-in
G1 MuJoCo simulation. See [[DimensionalOS dimos]].

**Edge node** — a self-contained field deployment: solar power, satellite link, on-device AI. The
conservation analogue is [[SPARROW]].

## Method

**Pretrained inference** — running a model as shipped, with no fine-tuning. The whole of Phase 1.

**Fine-tuning** — further training on task-specific data. **Out of scope** and deliberately absent
from this repo.

**Run metadata** — the record written alongside every run: commit, config, variant, threshold, seed,
input checksums. See [reproducibility docs](../docs/reproducibility.md).

## Related

[[Geologic Dome Context]] · [[Four Phase Arc]] · [[MegaDetector Variants]] · [[Reading List]]
