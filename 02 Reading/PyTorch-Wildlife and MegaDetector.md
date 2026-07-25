---
tags: [reading, models, phase-1]
depth: read
phase: phase-1
status: in-progress
url: "https://github.com/microsoft/Pytorch-Wildlife"
updated: 2026-07-24
---

# Microsoft PyTorch-Wildlife and MegaDetector

**Depth:** `[read]` — read closely · **Phase:** 1 · **This week:** yes

> The exact models [[SPARROW]] runs on camera-trap images. **You will run these yourself.**

- PyTorch-Wildlife — <https://github.com/microsoft/Pytorch-Wildlife>
- MegaDetector — <https://github.com/agentmorris/MegaDetector>

## Status: partly read

The API surface has been **verified against the installed PyTorch-Wildlife 1.3.0** — variant
strings, class vocabulary, and two broken upstream defaults are recorded in
[[MegaDetector Variants]]. But **no model has actually been run**, which is the entire point of
`[read]` here.

## What is already recorded elsewhere

- Variant strings, weights, broken defaults → [[MegaDetector Variants]]
- What MegaDetector is and is not → [model docs](../docs/model.md)
- Install gotchas (undeclared `soundfile`/`librosa`, `setuptools<81`) → [CLAUDE.md](../CLAUDE.md)

## Notes

_To answer by reading: how does the detector API expose batching, what does it return exactly, how
should video be fed through it, and what does the repo recommend for thresholds?_

## Questions

_Anything to raise at the next check-in._

## Related

[[MegaDetector Variants]] · [[SPARROW]] · [[PanAf20K Paper]] · [[Phase 1 Task Spec]] · [[Reading List]]
