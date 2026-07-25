---
tags: [onboarding, process]
status: reference
source: Geologic Dome Intern Onboarding.pdf (July 2026)
updated: 2026-07-24
---

# How We Work

Four expectations from the onboarding document, and where each one lives in this repo.

## 1. Keep a log

> One running doc of what you tried each session, **dead ends included**. That is how research
> actually works.

**Where:** [experiments/experiment_log.md](../experiments/experiment_log.md) — the single running
log. There is deliberately **no second log** in this vault; a duplicate log is a log nobody trusts.

The template at the top of that file already has the fields this asks for: objective, hypothesis,
environment, commands run, observations, **failures and dead ends**, **exact errors**,
interpretation, next action.

Write it during the session, not afterwards. Run `/log-session` to have Claude draft the entry.

## 2. Check in weekly

> Share progress, and **especially share it when you are stuck**.

**Where:** `03 Check-ins/`, one dated note per week, from [[Check-in Template]].

Being stuck is the thing to report, not the thing to hide until it is resolved.

## 3. Ask for help well

> Say **what you tried**, paste the **exact error**, and say **what you expected**. That unblocks
> you fast.

Three fields, all required. "It doesn't work" is not a question anyone can answer. The
[[Check-in Template]] has this structure built in, and the same discipline is why the experiment log
demands verbatim tracebacks rather than paraphrases.

## 4. Being stuck is normal

> Being stuck is normal, and it is most of the job. **Small wins compound.** Send me the first frame
> where your box lands cleanly on an ape. That is a real milestone.

Worth taking literally: the first clean detection is a deliverable to share, not something to sit on
until the pipeline is finished.

## How this interacts with the repo's honesty rules

These four practices and the repo's existing rules point the same way:

- Log dead ends → the log records failures, and `/log-session` will **not** manufacture entries.
- Paste exact errors → never paraphrase a traceback.
- Report when stuck → never quietly narrow the task to the part that worked.
- Never invent a number. If it was not measured, write **"not measured"**.

## Related

[[Phase 1 Task Spec]] · [[Check-in Template]] · [[Geologic Dome Context]] · [[PanAf Command Center]]
