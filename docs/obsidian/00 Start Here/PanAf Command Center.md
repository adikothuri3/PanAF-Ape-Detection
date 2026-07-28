---
tags: [command-center, start-here]
status: active
phase: "Phase 1 — See · steps 1-6 complete; cheap experiments before any fine-tuning"
updated: 2026-07-28
---

# PanAf Command Center

> **Find your next step.**
> Single entry point for this project. Start here every session — `/start-session` reads this note.

---

## Current Status

🟢 **Full Phase 1 pipeline working and measured on all 10 clips.**

Detection, tracking, overlay, video export and evaluation all run end to end. Tests and checks are
green (**281 tests, 19 verification checks**). The repo is published at
[adikothuri3/PanAF-Ape-Detection](https://github.com/adikothuri3/PanAF-Ape-Detection).

**All 10 clips, 3600 frames, 4985 annotated apes**, confidence 0.20 / IoU 0.50, verified `mps:0`,
using the current default `MDV6-yolov10-e` with ByteTrack:

| | Precision | Recall | F1 | Pooled mean IoU |
|---|---|---|---|---|
| **Current (`MDV6-yolov10-e`)** | **0.931** | **0.715** | **0.809** | 0.852 |
| Previous (`MDV6-yolov9-c`) | 0.917 | 0.353 | 0.509 | 0.832 |

3564 of 4985 apes found, 265 false positives. Tracking as shipped: 54 tracks over 23 individuals,
36 ID switches, **coverage 0.715**, 13 mostly-tracked, only 2 mostly-lost (was 9).

**Tracking coverage equalled detection recall to four decimal places (0.7149 vs 0.7149)** — every
detected ape was already tracked, and no track ever covered a frame the detector missed. Coverage
could not be improved by tuning; only identity could. That finding drove the tracking work below,
which has a candidate at **4 ID switches** pending validation on the full dataset. See [[tracking]].

Four clips are now above 0.95 recall, including the infrared night clip (0.963) that scored **0.009**
when this project started. The remaining weak spots are `isfRigsIjO` (0.211 recall, near-darkness --
though precision is 1.000, so what it does find is right) and `zvwY5xoIli` (0.348, small distant
subjects).

Full analysis: [findings write-up](../../../reports/phase1_findings_2026-07-26.md).

## Current Phase

**[[Four Phase Arc|Phase 1 — See]]**, all six steps complete.

Detect and track great apes in PanAf500 camera-trap video, overlay the dataset's behaviour labels,
and write up what you find. The full arc is 1 See → 2 Pose → 3 Predict → 4 Embody.

## Current Active Task

**Settle the merge regression, then adopt. One CPU sweep, no GPU.**

Tracking was validated on the full dataset — **500 clips, 874 individuals**. It holds:
identity coverage 0.7436 → **0.8197**, ID switches 1910 → **409**, fragmentation 2.65 → **1.30**,
jitter down 77%. Details in [[tracking]] and the 2026-07-28 log entry.

Two honest caveats, both of which belong in the write-up:

1. **The margin shrank.** +11.1pp on the 10 tuning clips, **+7.6pp** on 500. About a third of the
   apparent gain was overfitting; two thirds is real.
2. **Merged tracks rose 59 → 79** (purity 0.9970 → 0.9913). This appeared only at scale — on the
   10-clip sample merges had improved. It is the one metric that must not be traded away, because
   merging two apes is the failure identity coverage can partly reward.

What to run:

```
panaf-phase1 track-sweep --grid configs/sweeps/around-candidate.yaml \
    --config configs/tracking-candidate.yaml \
    --detections-dir artifacts/full500/detections --jobs 4 --max-merges 59
```

~30 minutes on CPU over the cache already on disk. If an arm clears the ceiling and keeps most of
the gain, adopt it into `base.yaml` and `colab.yaml`. If none does, adopt the candidate and report
the merge cost rather than choosing settings that hide it.

## Next Recommended Task

**After adoption: re-render the showcase clips, then Phase 2.**

1. **Re-render 2–3 annotated clips** from the adopted settings. Jitter fell 77% — this is the
   deliverable a reader actually watches.
2. **Phase 2 — Pose** is unblocked. It needed trustworthy tracks: you cannot build a movement
   trajectory for an animal whose identity flips every few seconds. At 409 switches across 874
   individuals, that is no longer the obstacle. Note the `interpolated` flag on tracked detections —
   those boxes were synthesised to bridge detector gaps, and pose work must be able to exclude them.
3. **Fine-tuning remains the weakest case yet.** Tracking was the bottleneck and configuration fixed
   it. Detection is the limit again, and now measured over the whole dataset rather than 10 clips.

The threshold question is reopened usefully. 0.20 is right for *reporting* single-frame detection
accuracy; it is the wrong threshold for *running* the pipeline, because detecting at 0.05 and
letting the tracker discard the junk gives better tracked precision at higher recall.

## Reading Progress

Four items are due this week. See [[Reading List]] for the full table and one unresolved ambiguity
about which items were meant.

| This week                             | Depth    | Status        |
| ------------------------------------- | -------- | ------------- |
| [[HF Deep RL Course]]                 | `[read]` | ⬜ not started |
| [[Spinning Up Key Concepts]]          | `[skim]` | ⬜ not started |
| [[PanAf20K Paper]]                    | `[skim]` | 🟡 partly     |
| [[PyTorch-Wildlife and MegaDetector]] | `[read]` | 🟡 partly     |

## Deliverable Checklist

From [[Phase 1 Task Spec]]:

- [ ] GitHub repo with code and README — *repo exists; pipeline code does not*
- [ ] 2–3 annotated clips or GIFs
- [ ] One-page write-up
- [ ] **Done means:** someone else can clone, follow the README, and reproduce one annotated clip

## Logs

| What | Where |
|---|---|
| Running research log — **the only one** | [experiments/experiment_log.md](../../../experiments/experiment_log.md) |
| Weekly check-ins | `03 Check-ins/`, from [[Check-in Template]] |
| Write-up template | [reports/phase1_writeup_template.md](../../../reports/phase1_writeup_template.md) |
| Run metadata (once inference exists) | `artifacts/metadata/` — not yet produced |

Log every session, dead ends included — [[How We Work]]. `/log-session` drafts the entry.

## Source of Truth

| Topic | Note |
|---|---|
| Company, mentor, the big idea | [[Geologic Dome Context]] |
| Phase numbering | [[Four Phase Arc]] |
| What Phase 1 asks for | [[Phase 1 Task Spec]] |
| Working practice | [[How We Work]] |
| The 9 behaviour labels | [[dataset|the nine action labels]] |
| Detector variants and defaults | [[model|MegaDetector variants]] |
| Terms | [[Glossary]] |
| Module design | [architecture](../05%20Technical/architecture.md) |
| Dataset background and ethics | [dataset](../05%20Technical/dataset.md) |
| What MegaDetector is and is not | [model](../05%20Technical/model.md) |
| Reproducibility contract | [reproducibility](../05%20Technical/reproducibility.md) |
| Three separate licences | [licensing](../05%20Technical/licensing.md) |

## Where documentation lives

**All project documentation is in this vault, at `docs/obsidian/`.** Open that directory as the
vault in Obsidian — not the repository root.

> **Notes never go inside `.obsidian/`.** That directory is Obsidian's *configuration* (app.json,
> core-plugins.json, workspace state). Obsidian does not index it as notes, so anything filed there
> disappears from the file explorer, search and the graph, and every `[[wikilink]]` to it breaks.
> Notes are **siblings** of `.obsidian/`, which is why the numbered folders sit next to it.

| Folder | Holds |
|---|---|
| `00 Start Here/` | This note — the entry point |
| `01 Onboarding/` | Project context, the four-phase arc, the Phase 1 spec, working practice |
| `02 Reading/` | The onboarding reading list, one note per item |
| `03 Check-ins/` | Weekly check-in template and notes |
| `04 Reference/` | Cross-cutting reference — currently the [[Glossary]] |
| `05 Technical/` | Architecture, dataset, model, licensing, reproducibility |

Four things live outside the numbered folders because tooling pins them there, **not** because they
are a second documentation system:

| Path | Pinned by |
|---|---|
| `README.md` | `pyproject.toml` (`readme =`) — the wheel build fails without it; also the GitHub landing page |
| `CLAUDE.md` | Claude Code loads project memory from the repository root |
| `experiments/experiment_log.md` | The single running log; `make verify` and `/log-session` require this path |
| `reports/phase1_writeup_template.md` | The Phase 1 deliverable; `make verify` requires this path |

Small `README.md` files inside `data/`, `notebooks/` and `experiments/` are **directory signposts**,
not documentation — GitHub renders them when browsing that folder, and they exist to be read at the
moment you are about to touch those files. Substantive documentation belongs in a numbered folder.

**The rule:** a fact has exactly one home. Everything else links to it. `04 Reference/` previously
restated the detector variants and behaviour labels that `05 Technical/` already documented; those
notes were merged into `model.md` and `dataset.md` rather than kept in sync by hand.

## Claude Workflow

| Command | Does |
|---|---|
| `/start-session` | Reads this note, reports phase/task/reading, runs `doctor`, checks the tree is clean |
| `/log-session` | Drafts an entry into the experiment log and updates this note |
| `/finish-phase` | Full gate, deliverable check, dated write-up, drafts a check-in |

`make quality` (ruff · format · mypy · pytest) and `make verify` are the gates. A PostToolUse hook
runs ruff on edited Python only — the full gate stays manual.

| Smoke test | Downloads weights | Proves |
|---|---|---|
| `make smoke-inference` | no | The extra imports; ByteTrack, NumPy interop, video round-trip and GIF export all work. Also a weekly CI job. |
| `make smoke-detect` | **yes (~1 GB)** | Real weights load and inference runs end to end, on the verified device. |

## Rules That Must Not Be Broken

1. **Never invent results.** No fabricated metrics, example detections, or filled-in templates.
   "Not measured" is honest; a plausible number is not.
2. **Never commit data or weights.** No video, frames, annotations, `*.pt`/`*.pth`, `artifacts/`,
   `.env`, or notebook output cells.
3. **No fine-tuning.** Phase 1 is pretrained inference only.
4. **No stubbed CLI commands.** Unimplemented stages are absent, not registered-and-raising.
5. **One log, one write-up, one home per fact.** Documentation lives in the numbered vault folders.
   If a note would restate another note, link instead — see [[#Where documentation lives]].
6. **`data/raw/` is immutable.** Derived frames go to `data/interim/` or `artifacts/`.
7. **Always pass the model variant explicitly** — the upstream defaults raise `ValueError`.

## Known gaps

- **PyTorch-Wildlife ignores `device=`.** Verified upstream bug: the model loads on CPU while
  reporting whatever you asked for. On Colab that means CPU speed with CUDA in the metadata. The
  three-line fix and the verification helper are in [[model|MegaDetector variants]]; Phase 1c must record
  the device `runtime.module_device()` reports, never the requested one.
- Annotation **file format** is unverified; the label list is not. See [[dataset|the nine action labels]].
- Tracker backend chosen from 1c evidence: **ByteTrack**, via `supervision`, no new dependency. Its
  low-confidence second association pass targets exactly the detector flicker a recall-0.4 baseline
  produces.
- **MOTA and IDF1 are not implemented**, deliberately. ID switches, fragmentation and coverage are,
  each checked against a worked example.
- **Tracked results must use `TrackedFrameDetections`.** A plain `FrameDetections` silently drops
  `track_id` and `behavior_label` on write; see [architecture](../05%20Technical/architecture.md).
- Manifest **loading and checksum verification** are not implemented — `manifest.py` is the schema
  only. `provenance.file_sha256()` computes the digests in the meantime.
- No code licence selected; the repo is public under default copyright. See
  [licensing](../05%20Technical/licensing.md).

## Related

[[Phase 1 Task Spec]] · [[Four Phase Arc]] · [[How We Work]] · [[Reading List]] · [[Check-in Template]] · [[Glossary]]
