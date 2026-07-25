---
tags: [command-center, start-here]
status: active
phase: "Phase 1 — See · scaffold complete (1a), clip selection next (1b)"
updated: 2026-07-24
---

# PanAf Command Center

> **Find your next step.**
> Single entry point for this project. Start here every session — `/start-session` reads this note.

---

## Current Status

🟡 **Repository scaffold complete. No detection has been run.**

The environment, typed config, setup CLI, tests, CI and documentation are in place and green
(79 tests, 13 verification checks, CI passing on `main`). The repo is published at
[adikothuri3/PanAF-Ape-Detection](https://github.com/adikothuri3/PanAF-Ape-Detection).

**Nothing in this repository has produced a detection.** No dataset has been downloaded, no model
weights loaded, no clip annotated. Steps 2–6 of [[Phase 1 Task Spec]] are all outstanding.

## Current Phase

**[[Four Phase Arc|Phase 1 — See]]**, sub-phase **1a complete**.

Detect and track great apes in PanAf500 camera-trap video, overlay the dataset's behaviour labels,
and write up what you find. The full arc is 1 See → 2 Pose → 3 Predict → 4 Embody.

## Current Active Task

**None active.** The scaffold task closed on 2026-07-24.

## Next Recommended Task

**Phase 1b — clip selection and frame extraction.**

1. Install FFmpeg (`brew install ffmpeg`) — [[#Known gaps|not currently present]].
2. Obtain PanAf500 access from the [Bristol deposit](https://data.bris.ac.uk/data/dataset/1h73erszj3ckn2qjwm4sqmr2wt)
   under its own licence, and download **5–10 clips only**.
3. Fill in `data/sample_manifest.csv` from the template, with SHA-256 checksums and a
   `selected_reason` per clip — selection axes are in [dataset docs](../docs/dataset.md).
4. Implement `panaf_ape_detection.data.manifest` then `.video`, tested against **synthetic** video
   so CI never needs the dataset.

Do manifest before frame extraction: it makes the checksum contract real before anything consumes
the files.

## Reading Progress

Four items are due this week. See [[Reading List]] for the full table and one unresolved ambiguity
about which items were meant.

| This week | Depth | Status |
|---|---|---|
| [[HF Deep RL Course]] | `[read]` | ⬜ not started |
| [[Spinning Up Key Concepts]] | `[skim]` | ⬜ not started |
| [[PanAf20K Paper]] | `[skim]` | 🟡 partly |
| [[PyTorch-Wildlife and MegaDetector]] | `[read]` | 🟡 partly |

## Deliverable Checklist

From [[Phase 1 Task Spec]]:

- [ ] GitHub repo with code and README — *repo exists; pipeline code does not*
- [ ] 2–3 annotated clips or GIFs
- [ ] One-page write-up
- [ ] **Done means:** someone else can clone, follow the README, and reproduce one annotated clip

## Logs

| What | Where |
|---|---|
| Running research log — **the only one** | [experiments/experiment_log.md](../experiments/experiment_log.md) |
| Weekly check-ins | `03 Check-ins/`, from [[Check-in Template]] |
| Write-up template | [reports/phase1_writeup_template.md](../reports/phase1_writeup_template.md) |
| Run metadata (once inference exists) | `artifacts/metadata/` — not yet produced |

Log every session, dead ends included — [[How We Work]]. `/log-session` drafts the entry.

## Source of Truth

| Topic | Note |
|---|---|
| Company, mentor, the big idea | [[Geologic Dome Context]] |
| Phase numbering | [[Four Phase Arc]] |
| What Phase 1 asks for | [[Phase 1 Task Spec]] |
| Working practice | [[How We Work]] |
| The 9 behaviour labels | [[PanAf500 Action Labels]] |
| Detector variants and defaults | [[MegaDetector Variants]] |
| Terms | [[Glossary]] |
| Module design | [architecture](../docs/architecture.md) |
| Dataset background and ethics | [dataset](../docs/dataset.md) |
| What MegaDetector is and is not | [model](../docs/model.md) |
| Reproducibility contract | [reproducibility](../docs/reproducibility.md) |
| Three separate licences | [licensing](../docs/licensing.md) |

## Claude Workflow

| Command | Does |
|---|---|
| `/start-session` | Reads this note, reports phase/task/reading, runs `doctor`, checks the tree is clean |
| `/log-session` | Drafts an entry into the experiment log and updates this note |
| `/finish-phase` | Full gate, deliverable check, dated write-up, drafts a check-in |

`make quality` (ruff · format · mypy · pytest) and `make verify` are the gates. A PostToolUse hook
runs ruff on edited Python only — the full gate stays manual.

## Rules That Must Not Be Broken

1. **Never invent results.** No fabricated metrics, example detections, or filled-in templates.
   "Not measured" is honest; a plausible number is not.
2. **Never commit data or weights.** No video, frames, annotations, `*.pt`/`*.pth`, `artifacts/`,
   `.env`, or notebook output cells.
3. **No fine-tuning.** Phase 1 is pretrained inference only.
4. **No stubbed CLI commands.** Unimplemented stages are absent, not registered-and-raising.
5. **One log, one write-up, one set of docs.** If a note would restate `docs/` or `experiments/`,
   link instead.
6. **`data/raw/` is immutable.** Derived frames go to `data/interim/` or `artifacts/`.
7. **Always pass the model variant explicitly** — the upstream defaults raise `ValueError`.

## Known gaps

- **FFmpeg is not installed** on this machine — needed before any video decoding or export.
- Annotation **file format** is unverified; the label list is not. See [[PanAf500 Action Labels]].
- Tracker backend undecided by design — chosen from 1c evidence.
- No code licence selected; the repo is public under default copyright. See
  [licensing](../docs/licensing.md).

## Related

[[Phase 1 Task Spec]] · [[Four Phase Arc]] · [[How We Work]] · [[Reading List]] · [[Check-in Template]] · [[Glossary]]
