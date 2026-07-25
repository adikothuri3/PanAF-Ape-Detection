---
tags: [onboarding, roadmap]
status: reference
source: Geologic Dome Intern Onboarding.pdf (July 2026)
updated: 2026-07-24
---

# Four Phase Arc

The project runs in four phases. **Phase 1 is the current and only active phase.**

| # | Phase | What it means |
|---|---|---|
| 1 | **See** | Detect and track great apes in wild camera-trap video and read their behaviour, using the PanAf20K dataset. |
| 2 | **Pose** | Add a skeleton to each animal, turning movement into joint data — the same representation a robot uses. |
| 3 | **Predict** | Build a small world model that predicts what the animal does next. |
| 4 | **Embody** | Translate that movement onto a Unitree G1 humanoid in MuJoCo simulation with DimensionalOS (dimos), the stack that runs on Pemba. |

## Why the phases are staged

The onboarding is direct about this:

> Phases 3 and 4 are real research problems. A full next-frame video model is hard, and mapping a
> climbing, four-limbed ape onto a bipedal humanoid (this is called **retargeting**) is harder. We
> stage them on purpose. **Your job right now is Phase 1, done well.**

The through-line is [[Geologic Dome Context|perceive → predict → act]]: Phase 1 perceives, Phase 2
converts perception into the representation robots use, Phase 3 predicts, Phase 4 acts.

## Correction to an earlier repo error

An earlier version of this repository's README invented a different roadmap — "Phase 2 =
quantitative evaluation, Phase 3 = fine-tuning". **That was wrong** and has been corrected.

Those two items are real and worth doing, but they are *rigour within Phase 1*, not phases of the
project. The repo now files them under "Beyond Phase 1". This note is the authority on phase
numbering; if anything disagrees with it, this note wins.

## Phase 2 on-ramp

The Phase 1 stretch goal — run an animal pose model (DeepLabCut or ViTPose) on one clip to get
skeletons — is explicitly the on-ramp to Phase 2. See [[Phase 1 Task Spec]].

## Related

[[Phase 1 Task Spec]] · [[Geologic Dome Context]] · [[Reading List]] · [[PanAf Command Center]]
