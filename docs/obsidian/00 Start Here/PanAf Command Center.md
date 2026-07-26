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

🟢 **Detection pipeline working and measured on real PanAf500 footage.**

The environment, typed config, runtime layer, schemas, setup CLI, tests, CI and documentation are in
place and green (**190 tests, 18 verification checks**, CI passing on `main`). The repo is published
at [adikothuri3/PanAF-Ape-Detection](https://github.com/adikothuri3/PanAF-Ape-Detection).

10 PanAf500 clips downloaded (23.4 MB), **3 processed over 1080 frames**. Measured at confidence
0.20 / IoU 0.50 on verified MPS:

| | Precision | Recall | F1 | mean IoU |
|---|---|---|---|---|
| **Overall** | **0.854** | **0.411** | **0.555** | 0.831 |

**Precise but insensitive** — it rarely invents an ape (7 false positives across 67 empty frames) but
misses about three in five. Failure concentrates in arboreal postures (`hanging` 0.000,
`climbing_up` 0.017), infrared night footage (one clip at recall 0.009) and small subjects (0.104).

Full analysis: [findings write-up](../../../reports/phase1_findings_2026-07-25.md).
**7 clips remain**, to be run on the Colab T4.

## Current Phase

**[[Four Phase Arc|Phase 1 — See]]**, sub-phase **1a complete**.

Detect and track great apes in PanAf500 camera-trap video, overlay the dataset's behaviour labels,
and write up what you find. The full arc is 1 See → 2 Pose → 3 Predict → 4 Embody.

## Current Active Task

**None active.** Phase 1b–1f closed on 2026-07-25 with the detection pipeline and first measurements.

## Next Recommended Task

**Finish the sample, then sweep the threshold — before any fine-tuning.**

1. Run the remaining **7 clips** on a Colab T4 via [the notebook](../../../notebooks/phase1_colab.ipynb).
2. **Confidence sweep.** Precision 0.854 against recall 0.411 says the operating point is mistuned
   for this footage. `panaf-phase1 evaluate` recomputes from saved detections, so a sweep costs no
   GPU time at all. Cheapest possible experiment; do it first.
3. **Variant comparison.** `MDV6-yolov10-e` is larger and higher-resolution. Config-only change.
4. Only then consider fine-tuning, and target the measured gaps — arboreal postures, infrared, small
   subjects — rather than the average.

Tracking (ByteTrack) is still deferred; `ape_id` ground truth is waiting for it.

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
- Tracker backend undecided by design — chosen from 1c evidence. `sv.ByteTrack` is confirmed working
  under the locked NumPy 2.x.
- Detection **quality is entirely unmeasured** — the only frames run through the model were
  synthetic noise.
- **Tracked results must use `TrackedFrameDetections`.** A plain `FrameDetections` silently drops
  `track_id` and `behavior_label` on write; see [architecture](../05%20Technical/architecture.md).
- Manifest **loading and checksum verification** are not implemented — `manifest.py` is the schema
  only. `provenance.file_sha256()` computes the digests in the meantime.
- No code licence selected; the repo is public under default copyright. See
  [licensing](../05%20Technical/licensing.md).

## Related

[[Phase 1 Task Spec]] · [[Four Phase Arc]] · [[How We Work]] · [[Reading List]] · [[Check-in Template]] · [[Glossary]]
