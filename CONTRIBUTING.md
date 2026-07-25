# Contributing

This is a research repository. The standards below exist so that results stay trustworthy and
reproducible, not to add ceremony.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you do not have uv
git clone <repository-url> panaf-ape-detection
cd panaf-ape-detection
make setup          # uv sync -- lightweight, no ML stack
make hooks          # install pre-commit hooks
make doctor         # confirm the environment
make quality        # lint, format check, mypy, tests
```

Add the heavy stack only when you need to run detection:

```bash
make setup-inference    # uv sync --extra inference
```

You also need FFmpeg for video work (`brew install ffmpeg`, `apt-get install ffmpeg`).

**Everything except inference must work without the extra.** If a change makes `make quality` or
the CLI require PyTorch, the change is wrong.

## Branches

Branch from `main`. Do not commit to `main` directly — a pre-commit hook blocks it.

```text
feat/<short-description>      new capability          feat/frame-extraction
fix/<short-description>       bug fix                 fix/manifest-checksum-validation
docs/<short-description>      documentation           docs/annotation-schema
exp/<short-description>       experiment work         exp/threshold-sweep
chore/<short-description>     tooling, deps, CI       chore/bump-ruff
```

## Quality commands

```bash
make lint            # ruff check .
make format          # ruff format .
make format-check    # ruff format --check .
make typecheck       # mypy src
make test            # pytest with coverage
make quality         # all of the above -- what CI runs
make verify          # repository structural invariants
```

Run `make quality && make verify` before opening a PR. CI runs the same commands, so a local failure
is a CI failure.

## Code standards

- **`src/` layout, modern typing.** `list[str]`, `X | None`, `from __future__ import annotations`.
- **Docstrings on public modules, classes and functions.** Google style; enforced by Ruff's `D`
  rules.
- **`pathlib`, never string path arithmetic.** Enforced by Ruff's `PTH` rules.
- **Paths resolve from the repository root**, via `panaf_ape_detection.paths`. Never
  `os.getcwd()`, never a hard-coded absolute path, never `~/Desktop/...`.
- **Structured logging, not scattered prints.** `logging.getLogger(__name__)` in library code.
  `rich` printing is fine in `cli.py`, where user-facing output *is* the job.
- **Small, testable functions.** If it needs a five-line docstring to explain, it is doing too much.
- **No global model loading.** Never load weights at import time or in a module-level constant. A
  detector is constructed inside a function, from config.
- **Heavy imports are lazy.** `import torch` goes inside the function that needs it, never at module
  scope in anything the CLI imports. `make verify` enforces this.
- **No credentials in source control.** Ever. See `.env.example`.
- **No business logic in notebooks.** See `notebooks/README.md`.

## Testing

- Tests live in `tests/`, mirroring the module they cover.
- **Tests must pass without the `inference` extra, without a GPU, and without network access.**
  This is not negotiable — it is what keeps CI honest and fast.
- **Tests never download model weights.** Test the adapter against a stub satisfying the detector
  protocol. If a real-weights test is ever added, mark it and exclude it from default CI.
- **Tests never require dataset files.** Generate small synthetic videos and detections in the test
  itself.
- New behaviour needs a test. Bug fixes need a test that fails before the fix.
- Test the failure paths, not just the happy one. Most of `tests/test_config.py` is invalid input,
  deliberately.

## Commits

Present tense, imperative, explaining *why* where it is not obvious.

```text
feat: extract frames honouring frame_stride
fix: reject manifest rows with mismatched checksums
docs: record verified MDV6 variant strings
chore: pin setuptools<81 for yolov5 pkg_resources import
```

Keep commits focused. A commit that changes a config default, adds a module, and reformats three
files is three commits.

Commit the experiment-log entry alongside the code or config it describes.

## Data handling

Read [`data/README.md`](data/README.md) and [`docs/obsidian/05 Technical/licensing.md`](docs/obsidian/05%20Technical/licensing.md) before touching
data.

**Never commit:**

- dataset files — video, frames, annotations, or your `data/sample_manifest.csv`
- model weights — `*.pt`, `*.pth`, `*.onnx`, `*.safetensors`, ...
- anything under `artifacts/`
- notebook output cells (they can embed dataset frames)
- `.env`, credentials, tokens

**Also:**

- `data/raw/` is immutable. Nothing writes to it. Derived frames go to `data/interim/` or
  `artifacts/`.
- Phase 1 uses ~5–10 PanAf500 clips. Do not download the full PanAf20K dataset.
- Record every clip in the manifest with checksums and a `selected_reason`.
- Annotated clips are derived works of the dataset. Check the licence before sharing them.

Pre-commit hooks and `make verify` block most of this, but they are a backstop, not a substitute for
paying attention.

## Logging experiments

Every session that touches data or models gets an entry in
[`experiments/experiment_log.md`](experiments/experiment_log.md), using the template at the top.

- Write the objective and hypothesis **before** running.
- Paste **exact** commands and **verbatim** errors.
- Record the model variant and confidence threshold — a detection count without them is not a
  result.
- **Record failures and dead ends.** An entry with no failures usually means it was written from
  memory.
- **Never invent a number.** "Not measured" is a valid entry; a plausible-looking estimate is a
  fabrication that will be quoted later.

## Pull requests

Fill in the PR template honestly — especially the **Honesty** section.

Expectations:

1. `make quality` and `make verify` pass.
2. New behaviour has tests; the suite still runs without the inference extra.
3. Documentation updated where behaviour changed.
4. No data, weights or secrets in the diff. Check `git diff --stat`.
5. **Nothing unimplemented is described as working.** If a stage is a stub, the docstring, the docs
   and the README all say so.
6. If dependencies changed: `make lock` run, both `uv.lock` and `requirements-colab.txt` committed,
   and `import PytorchWildlife` verified if the extra changed. A successful resolve is not a
   successful import — we have already been caught by that three times (see the log entry for
   2026-07-24).
7. If inference ran: variant, threshold, clips, device and run-metadata path in the PR description,
   plus an experiment-log entry.

Small PRs get reviewed. A PR that implements frame extraction, adds a tracker, and rewrites the
README will not.

## The one rule that matters most

**Do not fabricate results.** No invented metrics, no illustrative example detections presented as
real, no filled-in report templates, no plausible placeholder numbers.

An empty field is honest. A number nobody measured is not, and it will be quoted by someone who
assumes it was.
