# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## Scientific goal

Phase 1 ("See") of a wildlife computer-vision and robotics project. Establish a reproducible
baseline for detecting great apes in wild camera-trap video by running **pretrained MegaDetector V6**
(via Microsoft PyTorch-Wildlife) over ~5–10 clips from **PanAf500**, the densely annotated subset of
PanAf20K.

Research question: *without fine-tuning, how reliably does MegaDetector V6 localise great apes in
PanAf500 clips, and under which conditions does it fail?*

Later phases add tracking, behaviour-label overlay, quantitative evaluation, and eventually
deployment. Not now.

## Current phase and status

**Phase 1a complete: repository scaffold.**

Implemented and tested:

- `config.py` — typed YAML configuration (strict, root-relative paths, range-validated)
- `paths.py` — repository-root discovery and canonical layout
- `types.py` — `Detection`, `TrackedDetection`, `FrameDetections`, `RunMetadata` schemas
- `cli.py` — `doctor`, `validate-config`, `show-paths`

**Not implemented:** frame extraction, MegaDetector inference, tracking, behaviour overlay, video
export, evaluation. **No detection has ever been produced by this repository.**

Next task: clip selection + manifest, then frame extraction (`data/video.py`).

## Architecture

Full detail in [`docs/architecture.md`](docs/architecture.md). The rules that constrain edits:

1. **Foundation modules import nothing from the project.** `types.py`, `config.py`, `paths.py` are
   leaves.
2. **Stage modules never import each other.** `tracking/` does not import `inference/`; it consumes
   `Detection` from `types.py`.
3. **Only `pipeline/` knows the stage order.**
4. **`cli.py` stays thin** — parse, load config, delegate, format output. No experiment logic.
5. **Heavy ML imports are lazy**, inside functions. Importing `panaf_ape_detection` or running
   `panaf-phase1 doctor` must work after a bare `uv sync`, with no PyTorch installed.
   `scripts/verify_repository.py` enforces this.
6. **No global model loading.** Detectors are constructed inside functions, from config.

Keep inference, tracking, visualization and evaluation modular. If changing the tracker requires
editing the video writer, the boundary has been broken — fix the boundary.

## Commands

```bash
uv sync                      # lightweight dev environment (default)
uv sync --extra inference    # heavy ML stack (~180 packages)

make quality                 # lint + format-check + mypy + tests  (what CI runs)
make verify                  # repository structural invariants
make lock                    # re-resolve uv.lock AND refresh requirements-colab.txt

uv run panaf-phase1 doctor
uv run panaf-phase1 validate-config --config configs/base.yaml
uv run panaf-phase1 show-paths
```

Run `make quality && make verify` before declaring any change done.

## Hard rules

### Do not invent results

The single most important rule. **No fabricated metrics, example detections, sample outputs, or
filled-in report templates.** Not as illustration, not as a placeholder, not "for now".

If asked to demonstrate output that does not exist, say it does not exist. An empty field is honest;
a plausible number gets quoted later by someone who assumes it was measured.

This extends to documentation: never describe an unimplemented stage as working. The README, docs
and docstrings all currently mark unimplemented stages explicitly — keep it that way.

### Do not commit data or weights

Never commit dataset files, video, frames, annotations, `data/sample_manifest.csv`, model weights
(`*.pt`, `*.pth`, `*.onnx`, `*.safetensors`), anything under `artifacts/`, notebook output cells, or
`.env`.

`data/raw/` is immutable. Extracted frames go to `data/interim/` or `artifacts/`.

### Do not fine-tune

Phase 1 is **pretrained inference only**. No training loops, no optimisers, no fine-tuning scripts,
no `requires_grad` toggling. Fine-tuning is Phase 3 and only if a variant comparison justifies it.

### Do not stub CLI commands

Unimplemented pipeline commands are **absent**, not registered-and-raising. `make verify` fails if
the CLI exposes anything outside `{doctor, validate-config, show-paths}`. Register a command in the
same change that implements and tests it, and update the allowlist then.

### Do not download data or weights during development

No dataset downloads, no weight downloads in tests or CI. There is deliberately **no download
script** — acquisition is manual and documented in `data/README.md` until the endpoint and terms are
verified.

## Licensing restrictions

Three separate licences; see [`docs/licensing.md`](docs/licensing.md).

- **Repository code: no licence chosen.** No `LICENSE` file exists, so no reuse rights are granted.
  **Do not add a LICENSE file** unless explicitly asked — the choice has institutional implications.
- **Dataset:** Bristol deposit, non-commercial. Not covered by the code licence. Never
  redistributed through this repository. Annotated clips are derived works.
- **Model weights: licence varies by variant.** YOLOv9/YOLOv10-derived variants and the Apache
  RT-DETR variants differ. Verify per variant before deployment or commercial use.

## What MegaDetector actually does

`{0: "animal", 1: "person", 2: "vehicle"}`. That is the whole output space.

It does **not** identify species (no chimpanzee vs gorilla), individuals, or behaviour. Behaviour
labels displayed in output come from the **dataset**, never the model. Never write documentation or
captions implying otherwise.

Verified variant strings (PyTorch-Wildlife 1.3.0):

- `MegaDetectorV6`: `MDV6-yolov9-c`, `MDV6-yolov9-e`, `MDV6-yolov10-c`, `MDV6-yolov10-e`,
  `MDV6-rtdetr-c`
- `MegaDetectorV6Apache`: `MDV6-apa-rtdetr-c`, `MDV6-apa-rtdetr-e`

**Both classes' upstream defaults are broken** (`yolov9c` and `MDV6-rtdetr-x-apache` are not accepted
by their own validation and raise `ValueError`). Always pass the variant explicitly, from config.

## Requirements for future changes

Any change must come with:

- **Tests** that pass without the inference extra, without a GPU, and without network access.
  Weights are never downloaded in tests; use a stub detector satisfying the protocol.
- **Docstrings** on public modules, classes and functions (Google style, enforced by Ruff `D`).
- **Documentation updates** where behaviour changed — including flipping a stage from "not
  implemented" to implemented in the README table, `docs/architecture.md`, and the notebook.
- **An experiment-log entry** for anything touching data or models, including failures and verbatim
  errors.
- **`make lock`** if dependencies changed, committing both `uv.lock` and `requirements-colab.txt`.

## Dependency gotchas (verified, not theoretical)

The `inference` extra carries three workarounds, each commented in `pyproject.toml`:

- `soundfile` and `librosa` — PytorchWildlife 1.3.0 eagerly imports its `bioacoustics` subpackage
  from `__init__.py` but does not declare these; without them `import PytorchWildlife` fails.
- `setuptools<81` — PytorchWildlife pulls in `yolov5`, which still does `import pkg_resources`;
  setuptools 83 removed it.

**A successful `uv lock` does not mean the package imports.** After any change to the extra, verify
with `uv run --extra inference python -c "import PytorchWildlife"`.

## Conventions

- Python 3.11, `src/` layout, `from __future__ import annotations`, modern typing.
- `pathlib` only; paths resolved via `panaf_ape_detection.paths`, never `os.getcwd()` or absolute
  local paths.
- `logging.getLogger(__name__)` in library code; `rich` output only in `cli.py`.
- Ruff for lint and format (line length 100), mypy strict over `src/`.
- Config is strict: unknown keys are errors. Adding a YAML key means adding a model field.
