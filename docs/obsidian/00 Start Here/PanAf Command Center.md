---
tags: [command-center, start-here]
status: active
phase: "Phase 1 — See · steps 1-6 complete; cheap experiments before any fine-tuning"
updated: 2026-07-26
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

**All 10 clips, 3600 frames, 4985 annotated boxes**, confidence 0.20 / IoU 0.50, verified MPS. Run
twice, so the tracker's effect is isolated rather than confounded:

| | Precision | Recall | F1 | Pooled mean IoU |
|---|---|---|---|---|
| Detector only | 0.874 | 0.386 | 0.536 | 0.825 |
| + ByteTrack | **0.917** | 0.353 | 0.509 | 0.832 |

Tracking against `ape_id`: 23 individuals, 29 tracks, **17 ID switches**, fragmentation **1.35**,
coverage 0.353. ByteTrack holds an ape it can see; coverage is capped by detection recall.

**Precise but insensitive at 0.20 — and that threshold turned out to be the problem.** Swept over the
same 10 clips, F1 peaks at the *lowest* value tested:

| Confidence | Precision | Recall | F1 |
|---|---|---|---|
| **0.05** | 0.628 | **0.563** | **0.594** |
| 0.20 *(config default)* | 0.874 | 0.386 | 0.536 |
| 0.50 | 0.943 | 0.280 | 0.432 |

The recovered detections land almost entirely on the clips that looked hopeless: the near-dark clip
goes from **0 detections in 360 frames to recall 0.433**, `hanging` from 0.102 to 0.527. Low contrast
depresses confidence rather than preventing detection.

Getting that number required fixing a **second silently-ignored PyTorch-Wildlife parameter**:
`det_conf_thres` was never passed, so every run inferred at the library's 0.2 default. See
[[model]].

Full analysis: [findings write-up](../../../reports/phase1_findings_2026-07-26.md).

## Current Phase

**[[Four Phase Arc|Phase 1 — See]]**, all six steps complete.

Detect and track great apes in PanAf500 camera-trap video, overlay the dataset's behaviour labels,
and write up what you find. The full arc is 1 See → 2 Pose → 3 Predict → 4 Embody.

## Current Active Task

**None — the variant comparison is done, and it settled the open question.**

`MDV6-yolov10-e` nearly doubles recall at unchanged precision, for one line of config and 22 seconds
of extra compute on the whole sample. Plain-English write-up:
[variant comparison](../../../reports/variant_comparison_2026-07-27.md).

The decision to make: adopt it as the default variant (recommended — see below).

## Next Recommended Task

**Adopt the better model, then attack tracking — detection is no longer the bottleneck.**

1. **Switch `configs/base.yaml` and `colab.yaml` to `MDV6-yolov10-e`**, keeping the 0.20 threshold,
   which is now the *right* threshold: the new model peaks there and degrades gently either side.
   Then re-run all 10 clips once, so the annotated videos and headline numbers come from the model
   we actually recommend.
2. **Tracking is now the limit.** Coverage doubled to 0.728, but ID switches went 19 → 46 — more
   apes on screen means more chances to confuse two of them. Free things to try first, all over
   saved detections with `panaf-phase1 track`: ByteTrack activation at 0.20 rather than 0.05
   (already measured as better on every axis) and a lower `minimum_track_length`.
3. **Only then consider fine-tuning.** The case is much weaker than it looked: the gap was
   **capacity, not domain mismatch**, and a config change closed most of it. See
   [the write-up](../../../reports/variant_comparison_2026-07-27.md) for what fine-tuning would and
   would not fix.

The threshold question is closed. `yolov9-c`'s F1 rose all the way down to 0.05, so its optimum was
never found; `yolov10-e` peaks at the shipped default. The mistuned operating point was a symptom of
the smaller model, not a property of the footage.

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
