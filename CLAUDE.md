# CLAUDE.md

Guidance for Claude Code sessions in this repository.

The project brain lives in the **Obsidian vault at `docs/obsidian/`** — start at
[PanAf Command Center](docs/obsidian/00%20Start%20Here/PanAf%20Command%20Center.md). Run
`/start-session` at the beginning of a session and `/log-session` at the end.

**All documentation is in the numbered folders under `docs/obsidian/`** (`00 Start Here/` …
`05 Technical/`). This file, `README.md`, `experiments/experiment_log.md` and
`reports/phase1_writeup_template.md` sit outside the vault only because tooling pins them there.
When adding documentation, put it in a numbered folder and link to it — never restate an existing
note.

Notes are siblings of `docs/obsidian/.obsidian/`, never inside it: that directory is Obsidian's
**configuration**, and anything placed there is not indexed as a note, so links and search would
silently stop working.

> **Current phase: Phase 1 "See" — steps 1-5 implemented; 1f (write-up) in progress.** The full
> pipeline runs on real PanAf500 footage: clip selection, decoding, MegaDetector V6 inference,
> ByteTrack tracking, behaviour-label overlay, annotated video, and accuracy measured against the
> dataset's ground-truth boxes. Every number in the docs must trace to a file under
> `artifacts/metrics/`; nothing is estimated.

## The project

Geologic Dome builds edge infrastructure and robots for extreme and regulated environments —
autonomous field nodes and humanoids like **Pemba**, a Unitree G1. The thesis: the same AI that lets
a robot move through the world can help watch over it. Perceive → predict → act.

Mentor: **Pabs** (p@geologicdome.com). Context: [Geologic Dome Context](docs/obsidian/01%20Onboarding/Geologic%20Dome%20Context.md).

### Four phases — get this right

| # | Phase | Meaning |
|---|---|---|
| **1** | **See** | Detect and track great apes in camera-trap video, read behaviour. **← here** |
| 2 | Pose | Skeletons — movement as joint data, the representation a robot uses |
| 3 | Predict | A small world model of what the animal does next |
| 4 | Embody | Retarget onto a Unitree G1 in MuJoCo via DimensionalOS (dimos) |

An earlier README invented "Phase 2 = quantitative evaluation, Phase 3 = fine-tuning". **That was
wrong.** Those are rigour *within* Phase 1, filed under "Beyond Phase 1".
Authority: [Four Phase Arc](docs/obsidian/01%20Onboarding/Four%20Phase%20Arc.md).

### Phase 1, in six steps

1. **Set up** — ✅ done (uv, Python 3.11, locked env, published repo)
2. **Get the data** — ✅ done (10 purposively selected PanAf500 clips, checksummed manifest)
3. **Detect** — ✅ done (PyTorch-Wildlife MegaDetector V6 per frame, stitched to annotated video)
4. **Track** — ✅ done (ByteTrack via `supervision`; ID switches and fragmentation measured)
5. **Compare** — ✅ done (dataset action label drawn beside detections; recall broken down by label)
6. **Write it up** — in progress: `reports/phase1_findings_*.md`

**Deliverable:** a GitHub repo, 2–3 annotated clips or GIFs, the write-up.
**Done means:** someone else can clone, follow the README, and reproduce one annotated clip.
Full spec: [Phase 1 Task Spec](docs/obsidian/01%20Onboarding/Phase%201%20Task%20Spec.md).

## How we work

Log every session including dead ends · check in weekly, especially when stuck · ask for help with
what you tried + the exact error + what you expected · being stuck is normal.
Details: [How We Work](docs/obsidian/01%20Onboarding/How%20We%20Work.md).

The running log is **`experiments/experiment_log.md` and nowhere else.** Do not start a second log.

## Architecture

Full detail: [architecture](docs/obsidian/05%20Technical/architecture.md). The rules that constrain edits:

1. **Foundation modules import nothing from the project.** `types.py`, `config.py`, `paths.py` are leaves.
2. **Stage modules never import each other.** `tracking/` consumes `Detection` from `types.py`, not `inference/`.
3. **Only `pipeline/` knows the stage order.**
4. **`cli.py` stays thin** — parse, load config, delegate, format. No experiment logic.
5. **Heavy ML imports are lazy**, inside functions. `panaf-phase1 doctor` must work after a bare
   `uv sync` with no PyTorch installed; `scripts/verify_repository.py` enforces this.
6. **No global model loading.** Detectors are constructed inside functions, from config.

Keep inference, tracking, visualization and evaluation modular. If changing the tracker requires
editing the video writer, fix the boundary.

## Commands

```bash
uv sync                      # lightweight dev environment
uv sync --extra inference    # heavy ML stack (~190 packages)

make quality                 # ruff + format + mypy + tests  (what CI runs)
make verify                  # repository + vault invariants
make lock                    # re-resolve uv.lock AND refresh requirements-colab.txt

uv run panaf-phase1 doctor
uv run panaf-phase1 validate-config --config configs/base.yaml
uv run python scripts/verify_repository.py --only vault
```

Run `make quality && make verify` before declaring any change done.

## Hard rules

### Do not invent results

The most important rule. **No fabricated metrics, example detections, sample outputs, or filled-in
report templates.** Not as illustration, not as a placeholder, not "for now".

If asked to show output that does not exist, say it does not exist. "Not measured" is honest; a
plausible number gets quoted later by someone who assumes it was real. This extends to
documentation: never describe an unimplemented stage as working.

### Do not commit data or weights

Never commit dataset files, video, frames, annotations, `data/sample_manifest.csv`, weights
(`*.pt`, `*.pth`, `*.onnx`, `*.safetensors`), anything under `artifacts/`, notebook output cells, or
`.env`. `data/raw/` is immutable; derived frames go to `data/interim/` or `artifacts/`.

### Never run inference on CPU — use Colab

**Model inference runs on an accelerator or not at all.** CUDA on Colab, Apple MPS locally. A CPU
run does not fail: it produces correct results several times slower, so the cost shows up as a lost
afternoon rather than an error. `runtime.require_accelerator()` therefore **refuses** CPU, and
`MegaDetectorV6Runner` calls it — there is no path to a CPU run through the CLI.

Anything heavier than a few frames belongs on a **Colab GPU** via
[`notebooks/phase1_colab.ipynb`](notebooks/phase1_colab.ipynb) — Runtime → Change runtime type →
GPU, *before* running any cell. All 10 clips run end to end there. The A100 is the runtime
actually used for this project; a T4 works and is roughly 3x slower.

`PANAF_ALLOW_CPU_INFERENCE=1` exists only for debugging a non-performance bug on a machine with no
accelerator. **Nothing measured under it is reportable**, least of all a timing.

`model.device: auto` prefers CUDA, then MPS, and raises rather than silently choosing CPU. Never
change a config to `cpu` to make a run "work".

Configs for Colab runs: `configs/colab.yaml` (baseline), `configs/colab-sweep-conf005.yaml` and
`configs/colab-variant-yolov10e.yaml` (the two arms of the variant comparison, both detector-only at
confidence 0.05 so they compare at every threshold).

### Do not fine-tune

Phase 1 is **pretrained inference only**. No training loops, optimisers, or `requires_grad`
toggling. Fine-tuning is a Beyond-Phase-1 question, and only if a variant comparison justifies it.

### The artifacts contract — one directory, one schema

`artifacts/metrics/` holds **detection** metrics; `artifacts/metrics/tracking/` holds **track**
metrics. They are different shapes that share `clip_id`, `frames_evaluated` and `iou_threshold` —
enough to look interchangeable, not enough to be. They were once distinguished only by a filename
suffix, and a notebook cell that globbed the directory crashed with `KeyError: 'overall'` for every
user who enabled tracking.

**Never read `artifacts/` by hand.** `panaf_ape_detection.reporting` has the loaders, they are
tested, and they tolerate the legacy layout. Adding a third artifact shape means a third directory
and a `schema` field, not a fourth filename convention.

### Notebook code is code

`notebooks/` is **not** excluded from ruff, and `make verify` statically checks every cell: it must
parse, use no name an earlier cell has not defined, invoke no `uv` (Colab has none), reference only
configs that exist and only registered CLI commands. `tests/test_notebook_cells.py` then *executes*
every read-only cell against a synthetic `artifacts/` tree.

That machinery exists because three notebook defects reached the user in a row. Analysis logic goes
in `reporting.py` where it is typed and tested; a notebook cell should be a thin call.

### Do not stub CLI commands

Unimplemented pipeline commands are **absent**, not registered-and-raising. `make verify` fails if
the CLI exposes anything outside `{doctor, validate-config, show-paths}`. Register a command in the
same change that implements and tests it, updating the allowlist then.

### Do not download data or weights during development

No dataset or weight downloads in tests or CI. There is deliberately **no download script** —
acquisition is manual and documented in [data/README.md](data/README.md).

### One source of truth

A fact has exactly one home. If a note would restate another note, or restate
`experiments/experiment_log.md` or `reports/`, **link instead**.

## Licensing

Three separate licences — [licensing](docs/obsidian/05%20Technical/licensing.md).

- **Code: no licence chosen.** No `LICENSE` file, so default copyright applies and the repo is
  public under it. **Do not add a LICENSE** unless asked — it has institutional implications.
- **Dataset:** Bristol deposit, non-commercial. Never redistributed through this repo. Annotated
  clips are derived works.
- **Weights: licence varies by variant.** Verify per variant before deployment or commercial use.

## What MegaDetector actually does

`{0: "animal", 1: "person", 2: "vehicle"}`. That is the whole output space. It does **not** identify
species, individuals, or behaviour. Behaviour labels come from the **dataset**, never the model —
never write documentation or captions implying otherwise.

Verified variants and the two broken upstream defaults:
[`docs/obsidian/05 Technical/model.md`](docs/obsidian/05%20Technical/model.md). **Always pass `version=`
explicitly** — both classes' defaults raise `ValueError`.

The nine dataset behaviour labels:
[`docs/obsidian/05 Technical/dataset.md`](docs/obsidian/05%20Technical/dataset.md).

## Requirements for future changes

- **Tests** that pass without the inference extra, without a GPU, without network. Weights are never
  downloaded in tests; use a stub detector satisfying the protocol.
- **Docstrings** on public modules, classes and functions (Google style, Ruff `D`).
- **Documentation updates** where behaviour changed — including flipping a stage from "not
  implemented" in the README table, [architecture](docs/obsidian/05%20Technical/architecture.md) and the notebook.
- **An experiment-log entry** for anything touching data or models, including failures and verbatim
  errors.
- **`make lock`** if dependencies changed, committing `uv.lock` and `requirements-colab.txt`.
- **Vault notes stay linked** — `make verify` fails on a broken `[[wikilink]]`.

## Dependency gotchas (verified, not theoretical)

The `inference` extra carries three workarounds, each commented in `pyproject.toml`:

- `soundfile` and `librosa` — PytorchWildlife 1.3.0 eagerly imports its `bioacoustics` subpackage
  from `__init__.py` but does not declare these.
- `setuptools<81` — PytorchWildlife pulls in `yolov5`, which still does `import pkg_resources`;
  setuptools 83 removed it.

**A successful `uv lock` does not mean the package imports.** After any change to the extra, run
`make smoke-inference` — imports, ByteTrack, NumPy 2.x interop and a video round-trip, no weights.

### `device=` is ignored — this will bite Phase 1c

`MegaDetectorV6(device="cuda")` stores the value and **never applies it**; the line that would is
commented out upstream (`yolov8_base._load_model`). The model loads on CPU, nothing raises, and
`detector.device` still claims CUDA. On Colab that is CPU speed with GPU metadata.

The fix, applied after setup — all three lines are required:

```python
detector.predictor.model.to(torch_device)  # weights
detector.predictor.device = torch_device  # inputs (omit -> forward pass crashes)
detector.predictor.args.device = device  # args
```

Never record the *requested* device in `RunMetadata` — record what
`runtime.module_device(detector)` reports. `scripts/smoke_detect.py` is the working reference.
Full detail: [model docs](docs/obsidian/05%20Technical/model.md).

### `det_conf_thres` is ignored unless passed — the same trap, second instance

`single_image_detection(img, det_conf_thres=0.2)`. Omit it and **inference runs at 0.2 whatever
`model.confidence_threshold` says**; a post-inference filter then hides this completely while the
configured value happens to be 0.2, and every threshold below it returns identical results.

Always pass it: `single_image_detection(frame, det_conf_thres=config.model.confidence_threshold)`.

**The rule this library keeps teaching: an argument being accepted is no evidence it is applied.**
Verify by observing behaviour — where the tensors live, what scores come back — never by the absence
of an exception.

## Smoke tests

| Command | Downloads weights | Purpose |
|---|---|---|
| `make smoke-inference` | no | The extra imports and works. Also a weekly CI job. |
| `make smoke-detect` | **yes (~1 GB)** | Real weights load and inference runs. Local only. |

Run `make smoke-detect` before the first real clip, so a Phase 1c failure is distinguishable from a
broken stack.

## Conventions

- Python 3.11, `src/` layout, `from __future__ import annotations`, modern typing.
- `pathlib` only; paths via `panaf_ape_detection.paths`, never `os.getcwd()` or absolute local paths.
- `logging.getLogger(__name__)` in library code; `rich` output only in `cli.py`.
- Ruff lint + format (line length 100), mypy strict over `src/`.
- Config is strict: unknown keys are errors. A new YAML key means a new model field.
- Vault notes: YAML frontmatter (`tags`, `status`, `updated`), `[[wikilinks]]`, `## Related` footer.
