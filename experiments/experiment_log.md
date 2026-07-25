# Experiment log

A running research notebook. Newest entries go at the **bottom**, so the file reads chronologically.

**Rules**

1. One entry per working session that touches data, models or configuration.
2. Write the entry as you work, not afterwards. Reconstructed logs are fiction.
3. **Record failures and dead ends.** A dead end you wrote down is a result; one you deleted gets
   rediscovered in three weeks.
4. **Paste exact error messages**, not paraphrases. "It crashed on video loading" is unactionable;
   a traceback is a starting point.
5. **Never invent numbers.** If you did not measure it, write "not measured". An empty field is
   honest; a plausible-looking number is not.
6. Record the model variant and confidence threshold in every entry that runs inference.
7. Link to the run-metadata file in `artifacts/metadata/` once runs produce one.

---

## Entry template

Copy this block for each new entry.

```markdown
## YYYY-MM-DD HH:MM (timezone) — <short title>

**Objective**
What I set out to do in this session.

**Hypothesis / question**
What I expected to happen, or what I wanted to find out. Written *before* running.

**Environment**
- Machine / OS:
- Device (cpu / cuda / mps):
- Python:
- Commit:
- Working tree clean? yes / no

**Data / clip IDs**
Which clips, from which manifest. Note if checksums were verified.

**Model and variant**
- Framework:
- Model:
- Variant:
- Confidence threshold:

**Configuration**
Config file used, plus any environment overrides.

**Commands run**
```bash
# exact commands, copy-pasted
```

**Observations**
What actually happened, including things I did not expect.

**Quantitative results**
Numbers, with units and the threshold they were computed at. "Not measured" if not measured.

**Qualitative results**
What the output looked like. Which clips, which frames, what was visibly right or wrong.

**Failures and dead ends**
What did not work, and what I tried that led nowhere.

**Exact errors**
```text
# verbatim tracebacks and error messages
```

**Interpretation**
What I think this means. Flag speculation as speculation.

**Next action**
The single next thing to do.
```

---

## 2026-07-24 — Repository scaffold created

**Objective**
Create the Phase 1 repository scaffold: package layout, locked environment, typed configuration,
setup CLI, tests, CI, and documentation.

**Hypothesis / question**
None — this was setup work, not an experiment.

**Environment**
- Machine / OS: macOS (Darwin 25.5.0), arm64
- Device: n/a (no inference run)
- Python: 3.11.15
- Commit: initial commit
- Working tree clean? n/a at time of creation

**Data / clip IDs**
**None.** No dataset was downloaded, accessed, or processed. `data/raw/` is empty.

**Model and variant**
**None loaded.** No model weights were downloaded and no inference was run. `configs/base.yaml`
specifies `MegaDetectorV6` / `MDV6-yolov9-c` / threshold `0.2` as the *intended* Phase 1 baseline;
this has not been executed.

**Configuration**
`configs/base.yaml` and `configs/colab.yaml` created and validated.

**Commands run**
```bash
uv sync
uv sync --extra inference
uv run panaf-phase1 doctor
uv run panaf-phase1 validate-config --config configs/base.yaml
uv run panaf-phase1 show-paths
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run python scripts/verify_repository.py
```

**Observations**
- Default (`uv sync`) environment installs 30-odd packages; the `inference` extra pulls in ~180,
  because PytorchWildlife depends on `ultralytics`, `yolov5`, `lightning` and `gradio`.
- `doctor` correctly reports the inference stack as absent after a bare `uv sync` and present after
  `uv sync --extra inference`.
- On this machine: CUDA not available, Apple MPS available, FFmpeg **not** installed.
- Verified from the installed PyTorch-Wildlife 1.3.0 source that `MegaDetectorV6` accepts
  `MDV6-yolov9-c`, `MDV6-yolov9-e`, `MDV6-yolov10-c`, `MDV6-yolov10-e`, `MDV6-rtdetr-c`, and that
  `MegaDetectorV6Apache` accepts `MDV6-apa-rtdetr-c`, `MDV6-apa-rtdetr-e`. Class vocabulary is
  `{0: animal, 1: person, 2: vehicle}` — no species, no behaviour.

**Quantitative results**
Not applicable — no inference was run. Test suite: 79 tests passing, 94% statement coverage of
`src/`. These are software metrics, not research results.

**Qualitative results**
Not applicable. No frames, detections, or clips were produced.

**Failures and dead ends**
Three real dependency problems in the `inference` extra, all found by trying to import the package
rather than by assuming the resolve was enough:

1. `import PytorchWildlife` failed on a missing `soundfile`. PytorchWildlife 1.3.0's
   `__init__.py` eagerly imports its `bioacoustics` subpackage, which imports `soundfile` at module
   level, but `soundfile` is not a declared dependency.
2. After adding `soundfile`, the same chain failed on a missing `librosa`, undeclared for the same
   reason.
3. After adding `librosa`, the import failed inside `yolov5`, which does `import pkg_resources`.
   setuptools 83 (resolved by default) removed `pkg_resources`.

Resolved by adding `soundfile`, `librosa` and `setuptools<81` to the `inference` extra, each with a
comment in `pyproject.toml` explaining that it is an upstream workaround rather than a real
dependency of this project. Import verified afterwards.

Also noted: `MegaDetectorV6.__init__` defaults to `version='yolov9c'` and
`MegaDetectorV6Apache.__init__` to `version='MDV6-rtdetr-x-apache'`, and **neither default is
accepted by its own validation** — both fall through to `raise ValueError`. The variant must be
passed explicitly.

**Exact errors**
```text
ModuleNotFoundError: No module named 'soundfile'
  File ".../PytorchWildlife/data/bioacoustics/bioacoustics_annotations.py", line 3, in <module>
    import soundfile as sf

ModuleNotFoundError: No module named 'librosa'
  File ".../PytorchWildlife/data/bioacoustics/bioacoustics_spectrograms.py", line 24, in <module>
    import librosa

ModuleNotFoundError: No module named 'pkg_resources'
  File ".../yolov5/utils/general.py", line 34, in <module>
    import pkg_resources as pkg
```

**Interpretation**
The scaffold is complete and every check passes, but **nothing about the research question has been
touched.** No claim about MegaDetector's performance on PanAf500 is supported by anything in this
repository.

The dependency findings are worth carrying forward: the `inference` extra is fragile in ways that
are upstream's fault and not visible from a successful `uv lock`. Resolution succeeding is not the
same as import succeeding — check imports after any dependency change. The broken upstream variant
defaults independently justify keeping `model.variant` mandatory in config.

FFmpeg is missing on this machine and will be needed before video decoding or export.

**Next action**
Install FFmpeg. Then obtain PanAf500 access, select 5–10 clips against the axes in
`docs/dataset.md`, and populate `data/sample_manifest.csv` with checksums. Then implement frame
extraction (`data/video.py`) with tests against synthetic video.
