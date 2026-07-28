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

🟢 **Phase 1 "See" is complete, and measured on the entire dataset.**

All **500 PanAf500 clips / ~180,000 frames / 201,430 annotated boxes / 874 annotated individuals**.

| | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|
| Best single-frame threshold (0.40) | 0.864 | 0.808 | 0.835 | — |
| Pipeline before 2026-07-28 | 0.794 | 0.850 | 0.821 | 0.837 |
| **Pipeline as shipped** | **0.855** | **0.859** | **0.857** | 0.838 |

Tracking: **301 ID switches** (was 2257), fragmentation **1.27** (was 2.48), identity coverage
**0.823** (was 0.740), jitter down **77%**, 645 of 874 apes mostly-tracked.

**The pipeline is more accurate than any confidence threshold can be.** It detects generously at
0.05 — raw precision there is 0.468 — and lets the tracker discard what does not persist. That
beats the 0.835 F1 ceiling of the best single-frame operating point, because temporal consistency
is evidence a per-frame cut cannot use.

What still fails is occlusion and scale: `sitting_on_back` 0.207, arboreal postures 0.60–0.73,
small subjects 0.711 against 0.93 for medium and large.

**On the dataset's own 75 held-out test clips**, where re-running the selection on train alone
picks the same settings: identity coverage 0.7641 → **0.8612**, ID switches 342 → **60**. The
identity gain is larger there than on train. Merged tracks went the other way on held-out data
(5 → 9 on test, 4 → 8 on validation) even though the aggregate improved — that is in the write-up
rather than smoothed over.

Full analysis: [findings write-up](../../../reports/phase1_findings_2026-07-28.md). Tracking detail:
[[tracking]].

## Current Phase

**[[Four Phase Arc|Phase 1 — See]]**, all six steps complete, measured on the whole dataset.

## Current Active Task

**None. Phase 1 is closed.**

The deliverables are in place: a public repo, annotated showcase clips rendered from the shipped
config, and a write-up whose every number traces to a file under `artifacts/`.

## Next Recommended Task

**Phase 2 — Pose.** It was blocked on trustworthy tracks: you cannot build a movement trajectory
for an animal whose identity flips every few seconds. At 301 switches across 874 individuals and
fragmentation 1.27, that is no longer the obstacle.

One thing to carry forward: tracked detections may be flagged `interpolated`. Those boxes were
synthesised to bridge frames the detector missed — they are the only reason coverage can exceed
detection recall — and there is **no image evidence underneath them**. Pose work must be able to
exclude them, which is why the flag exists.

If detection is revisited before then, §7 of the write-up is the table to improve, and
`sitting_on_back` at 0.207 recall is where the headroom is. The case for fine-tuning is weaker than
it looked a week ago: tracking was the bottleneck, and configuration fixed it without training
anything.

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
