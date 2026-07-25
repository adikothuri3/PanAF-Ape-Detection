---
tags: [template, check-in]
status: template
updated: 2026-07-24
---

# Check-in Template

Copy the block below into a new dated file in `03 Check-ins/`. Naming: `YYYY-MM-DD Check-in.md`.

Weekly, per [[How We Work]] — and **especially when stuck**, which is the case this is really for.
Keep it short enough to read on a phone.

---

```md
---
tags: [check-in]
date: YYYY-MM-DD
week: <n>
phase: "Phase 1 — See"
status: on-track | stuck | blocked
---

# YYYY-MM-DD — Check-in

## Where I am
One or two lines. Which of the six steps in [[Phase 1 Task Spec]] are done, and which is active.

## Done since last check-in
- ...

## In progress
- ...

## Stuck on
Say this plainly. Being stuck is normal and it is most of the job — reporting it late is the only
real mistake.

### What I tried
- ...

### Exact error
```text
verbatim traceback or output — not a paraphrase
```

### What I expected
- ...

## Questions
- ...

## Next
The single next thing I plan to do.

## Evidence
Link a log entry, a run, or a frame. If a detection box landed cleanly on an ape for the first time,
that goes here — it is a real milestone, not a footnote.
```

## Why the three-field structure

The onboarding asks for help to be requested in exactly this shape:

> Say **what you tried**, paste the **exact error**, and say **what you expected**. That unblocks
> you fast.

All three are required. Two out of three still leaves the reader guessing.

## Standing questions

Carry these forward until answered:

- **Reading list ambiguity** — the onboarding says to read "the two Foundations items and the two
  *Our world* items" this week, but only one item is listed under *Our world*. Which two were meant?
  See [[Reading List]].
- **[[DimensionalOS dimos]] repository location** — not given in the onboarding.
- **[[SPARROW]] canonical URL** — not given in the onboarding.

## Related

[[How We Work]] · [[Phase 1 Task Spec]] · [[PanAf Command Center]] · [experiment log](../../../experiments/experiment_log.md)
