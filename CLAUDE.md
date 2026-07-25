# CLAUDE.md

Guidance for Claude Code sessions in this repository.

The project brain lives in the **Obsidian vault at the repository root** — start at
[PanAf Command Center](00%20Start%20Here/PanAf%20Command%20Center.md). Run `/start-session` at the
beginning of a session and `/log-session` at the end.

> **Current phase: Phase 1 "See" — sub-phase 1a (scaffold) complete and audited, 1b (clip
> selection) next.** The stack is proven: `make smoke-detect` loads real weights and runs inference
> end to end. But **no dataset has been downloaded and no clip annotated** — the only frames ever
> put through the model were synthetic noise, so detection quality is entirely unmeasured.

## The project

Geologic Dome builds edge infrastructure and robots for extreme and regulated environments —
autonomous field nodes and humanoids like **Pemba**, a Unitree G1. The thesis: the same AI that lets
a robot move through the world can help watch over it. Perceive → predict → act.

Mentor: **Pabs** (p@geologicdome.com). Context: [Geologic Dome Context](01%20Onboarding/Geologic%20Dome%20Context.md).

### Four phases — get this right

| # | Phase | Meaning |
|---|---|---|
| **1** | **See** | Detect and track great apes in camera-trap video, read behaviour. **← here** |
| 2 | Pose | Skeletons — movement as joint data, the representation a robot uses |
| 3 | Predict | A small world model of what the animal does next |
| 4 | Embody | Retarget onto a Unitree G1 in MuJoCo via DimensionalOS (dimos) |

An earlier README invented "Phase 2 = quantitative evaluation, Phase 3 = fine-tuning". **That was
wrong.** Those are rigour *within* Phase 1, filed under "Beyond Phase 1".
Authority: [Four Phase Arc](01%20Onboarding/Four%20Phase%20Arc.md).

### Phase 1, in six steps

1. **Set up** — ✅ done (uv, Python 3.11, locked env, published repo)
2. **Get the data** — 5–10 PanAf500 clips, not all 7 million frames
3. **Detect** — PyTorch-Wildlife (MegaDetector) per frame, boxes, stitched to video
4. **Track** — stable IDs with SORT or ByteTrack
5. **Compare** — show the dataset's action label beside detections; does it match?
6. **Write it up** — one page: what worked, what failed, three improvements

**Deliverable:** a GitHub repo, 2–3 annotated clips or GIFs, the write-up.
**Done means:** someone else can clone, follow the README, and reproduce one annotated clip.
Full spec: [Phase 1 Task Spec](01%20Onboarding/Phase%201%20Task%20Spec.md).

## How we work

Log every session including dead ends · check in weekly, especially when stuck · ask for help with
what you tried + the exact error + what you expected · being stuck is normal.
Details: [How We Work](01%20Onboarding/How%20We%20Work.md).

The running log is **`experiments/experiment_log.md` and nowhere else.** Do not start a second log.

## Architecture

Full detail: [architecture](docs/architecture.md). The rules that constrain edits:

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

### Do not fine-tune

Phase 1 is **pretrained inference only**. No training loops, optimisers, or `requires_grad`
toggling. Fine-tuning is a Beyond-Phase-1 question, and only if a variant comparison justifies it.

### Do not stub CLI commands

Unimplemented pipeline commands are **absent**, not registered-and-raising. `make verify` fails if
the CLI exposes anything outside `{doctor, validate-config, show-paths}`. Register a command in the
same change that implements and tests it, updating the allowlist then.

### Do not download data or weights during development

No dataset or weight downloads in tests or CI. There is deliberately **no download script** —
acquisition is manual and documented in [data/README.md](data/README.md).

### One source of truth

If a vault note would restate `docs/`, `experiments/` or `reports/`, **link instead**. The vault
adds the onboarding, reading, check-in and verified-reference layer; it does not mirror the repo.

## Licensing

Three separate licences — [licensing](docs/licensing.md).

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
[MegaDetector Variants](04%20Reference/MegaDetector%20Variants.md). **Always pass `version=`
explicitly** — both classes' defaults raise `ValueError`.

The nine dataset behaviour labels:
[PanAf500 Action Labels](04%20Reference/PanAf500%20Action%20Labels.md).

## Requirements for future changes

- **Tests** that pass without the inference extra, without a GPU, without network. Weights are never
  downloaded in tests; use a stub detector satisfying the protocol.
- **Docstrings** on public modules, classes and functions (Google style, Ruff `D`).
- **Documentation updates** where behaviour changed — including flipping a stage from "not
  implemented" in the README table, [architecture](docs/architecture.md) and the notebook.
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
Full detail: [model docs](docs/model.md).

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
