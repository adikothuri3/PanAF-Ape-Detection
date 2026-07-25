---
tags: [reading, edge-ai, context]
depth: read
phase: context
status: not-started
url: "confirm — search Microsoft Research for the SPARROW project"
updated: 2026-07-24
---

# Microsoft SPARROW project

**Depth:** `[read]` — read closely · **Phase:** Context · **This week:** yes

> Solar power, Starlink, and a Jetson running wildlife AI in the field. **This is the conservation
> version of a Geologic Dome node.**

## Why it matters here

This is the closest public analogue to what Geologic Dome builds — see
[[Geologic Dome Context|autonomous field nodes]]. It is also the deployment context that makes the
Phase 1 constraints real: a model that only runs on a workstation is useless on a solar-powered
Jetson at a remote site, which is why detector variant and threshold are recorded per run rather
than tuned by feel.

The onboarding notes SPARROW runs [[PyTorch-Wildlife and MegaDetector|the same models]] this project
uses.

## URL

Not given in the onboarding document. Find it via Microsoft Research rather than guessing — this
note deliberately does not assert a URL it has not verified.

## Notes

_To answer by reading: what runs on-device vs. off, what the power and bandwidth budget looks like,
and what they do about false positives in the field._

## Questions

_Anything to raise at the next check-in._

## Related

[[Geologic Dome Context]] · [[PyTorch-Wildlife and MegaDetector]] · [[Reading List]]
