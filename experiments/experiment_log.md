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
`docs/obsidian/05 Technical/dataset.md`, and populate `data/sample_manifest.csv` with checksums. Then implement frame
extraction (`data/video.py`) with tests against synthetic video.

---

## 2026-07-25 — Pre-inference infrastructure audit

**Objective**
Audit the scaffold by exercising it rather than reading it, before running any real inference, so
that a Phase 1c failure is attributable to new pipeline code and not to the stack underneath it.

**Hypothesis / question**
The gates are green, but green gates only prove the things they check. Two questions: does the
heavy `inference` extra actually *work* (not merely resolve), and are there latent defects in the
foundation that would surface mid-implementation?

**Environment**
- Machine / OS: macOS (Darwin 25.5.0), arm64, Apple M1
- Device: MPS available, no CUDA
- Python: 3.11.15
- Commit: eb1e9c2 (working tree dirty throughout the audit)
- FFmpeg: now installed at /opt/homebrew/bin/ffmpeg

**Data / clip IDs**
**None.** No dataset was accessed. All checks used synthetic frames and synthetic video.

**Model and variant**
`MegaDetectorV6` / `MDV6-yolov9-c`, threshold 0.2, loaded from real weights for the first time in
this project via `scripts/smoke_detect.py`. Run on a **synthetic noise frame only**.

**Configuration**
`configs/base.yaml`, unmodified.

**Commands run**
```bash
uv run ruff check . && uv run mypy src && uv run pytest && uv run python scripts/verify_repository.py
git check-ignore -q reports/figures/failure.png
make smoke-inference
make smoke-detect
```

**Observations**

Two real bugs, both silent:

1. **`PathsConfig` was dead code.** Nothing read `config.paths.*`. `doctor`, `show-paths` and
   `Config.repository_paths()` all used the hardcoded checkout layout, and `describe()` did not
   display the paths at all. Setting `paths.artifacts_dir` or `PANAF_ARTIFACTS_DIR` validated and
   was then discarded, with no visible symptom.
2. **`reports/figures/*.png` was git-ignored.** The write-up template asks for a figure there and
   references `figures/TODO.png`; it would have vanished with no error.

Then, from actually loading the model — the most important finding:

3. **PyTorch-Wildlife 1.3.0 ignores `device=` entirely.** In
   `yolov8_base._load_model` the line that would apply it is commented out:
   `# self.predictor.args.device = device # Will uncomment later`. The value is accepted, stored on
   `self.device`, and never used. The weights load on CPU; nothing raises; `detector.device` still
   reports what you asked for. **On a Colab GPU runtime this is CPU speed with CUDA in the metadata.**

Dependency risk was lower than feared. `supervision` 0.23.0 (pinned by PytorchWildlife, from
Aug 2024) and `yolov5` 7.0.10 were scanned for numpy-2.0-removed aliases against the locked numpy
2.4.6 — none found — and `sv.ByteTrack` then tracked a synthetic box across 10 frames with a stable
id, so the pairing works in practice and not just on paper.

**Quantitative results**
- Test suite: 79 → 128 passing. Verification checks: 17 → 18.
- `MDV6-yolov9-c` first load (incl. download): 72.6 s; cached load: 6.3 s.
- Inference on one synthetic 1280×720 frame: **2.13 s on CPU, 1.53 s on MPS** after forcing the
  device. The speedup is the evidence the move actually took effect.
- Detections on random noise: 0 — correct; any detection there would be a false positive.

These are software and timing measurements. **Nothing here measures detection quality**, which
remains entirely unmeasured.

**Qualitative results**
`make smoke-detect` now walks the whole chain — config → device → weight download → model load →
inference → run metadata — and prints a real `RunMetadata` including the git commit and
`git_dirty: True` (accurate; the tree was dirty).

**Failures and dead ends**

- First attempt at the device fix set `predictor.args.device` after construction. **No effect** —
  `setup_model()` has already run by then.
- Second attempt moved the weights with `predictor.model.to(device)` but not the inputs. Weights
  moved, then the forward pass crashed (see below). The working fix needs **three** lines: move the
  model, set `predictor.device`, and set `args.device`.
- My own `runtime.module_device()` initially returned `None` for the real detector, and the smoke
  test treated `None` as success — so the first "passing" run was reporting `ok` on a model that
  was actually on CPU. Cause: ultralytics nests weights at `.predictor.model.model`, and the
  intermediate `AutoBackend.parameters()` yields an empty generator, so a shallow search finds a
  parameters() method that reports nothing. Fixed with a bounded breadth-first search that keeps
  descending past empty wrappers, plus treating `None` as a failure rather than a pass.
- `scripts/smoke_inference.py` first probed imageio's `pyav` plugin, which is not what the extra
  ships — we declare `imageio[ffmpeg]`. Switched to the FFMPEG plugin, which works, and added a GIF
  export check since GIF is an accepted deliverable format.

**Exact errors**
```text
RuntimeError: slow_conv2d_forward_mps: input(device='cpu') and weight(device=mps:0')
must be on the same device

ImportError: The `pyav` plugin is not installed. Use `pip install imageio[pyav]` to install it.
```

**Interpretation**
The scaffold was sound in the places the gates covered and defective in three places they did not.
All three shared a failure mode: **something was accepted, stored, and then silently not used.** A
config section, a gitignore negation, a device argument. None would have raised; all three would
have produced confusing behaviour attributed to whatever code was being written at the time.

The device bug is the one that matters most. Phase 1c must record the *verified* device from
`runtime.module_device()`, never the requested one, or every performance claim in the write-up will
be unfounded.

The stack itself is in good shape: imports work, ByteTrack works under numpy 2.x, video round-trips
through OpenCV and imageio, GIF export works, and real weights load and run.

**Next action**
Phase 1b. Obtain PanAf500 access, select 5–10 clips against the axes in `docs/obsidian/05 Technical/dataset.md`, populate
`data/sample_manifest.csv` with checksums (`provenance.file_sha256` now computes them), then
implement `data/manifest.py` before `data/video.py`, tested against synthetic video.

---

## 2026-07-25 — Full codebase audit

**Objective**
Audit the entire repository, not just the inference path, to confirm it is a sound base for
continuing research.

**Hypothesis / question**
The gates are green and the stack is proven, but green gates only cover what they check. What is
wrong in the parts nothing checks — the schemas, the docs, the guards themselves?

**Environment**
- Machine / OS: macOS (Darwin 25.5.0), arm64, Apple M1
- Device: MPS available, no CUDA
- Python: 3.11.15
- Commit: cd3c9f6 (working tree dirty throughout)

**Data / clip IDs**
**None.** No dataset was accessed. Synthetic frames and synthetic trees only.

**Model and variant**
`MegaDetectorV6` / `MDV6-yolov9-c`, threshold 0.2 — loaded once via the new opt-in weights test, on
a synthetic noise frame.

**Configuration**
`configs/base.yaml`, unmodified.

**Commands run**
```bash
uv run ruff check . && uv run mypy src && uv run pytest && uv run python scripts/verify_repository.py
uv run pytest -m weights --collect-only          # before: collected nothing
uv run --extra inference pytest -m weights       # after: 2 passed
make smoke-inference && make smoke-detect
```

**Observations**

One data-loss bug and six smaller defects.

1. **Serialization dropped track identity.** `FrameDetections.detections` was annotated
   `list[Detection]`. A `TrackedDetection` survived *in memory* — pydantic keeps the subclass — but
   `model_dump()` serializes to the **declared** type. Verified: dumped keys were
   `['box', 'category_id', 'category_name', 'confidence']`. Track ids and behaviour labels, which
   are the entire Phase 1d/1e deliverable, would have been destroyed on write with no error.
2. **Detection records were not self-describing.** No frame dimensions anywhere, so a saved record
   could not be normalised or bounds-checked without re-opening the video. This blocked "small
   distant subjects", a failure axis `docs/obsidian/05 Technical/model.md` names, since measuring it needs box area
   *relative to frame area*.
3. **The Colab notebook predated the runtime layer** and never mentioned the device bug — despite
   Colab being exactly where that bug costs most.
4. **The `weights` marker was registered and documented but unused**; `pytest -m weights` collected
   nothing.
5. **Manifest columns were defined in three places** with nothing keeping them in sync.
6. **`scripts/verify_repository.py` (668 lines) had no tests.**
7. `filterwarnings` module patterns were misleading — the third field matches the module that
   *raises* the warning, often not the one named.

**Quantitative results**
- Tests 128 → 190 (plus 2 opt-in weights tests). Verification checks: 18, now themselves tested.
- 37 Markdown files scanned for broken relative links: the only hit is a deliberate placeholder
  inside an HTML comment in the write-up template.
- No TODO/FIXME in `src/`. Every `config` field is unread outside `config.py`, which is correct —
  the pipeline does not exist yet, and it confirms nothing is mis-wired.

Software metrics only. **Detection quality remains entirely unmeasured.**

**Qualitative results**
`TrackedFrameDetections` now round-trips through JSON with `track_id` and `behavior_label` intact,
proven by a test that fails against the old schema.

**Failures and dead ends**

- First fix for the serialization bug overrode `detections: list[TrackedDetection]`. mypy rejected
  it — correctly. `list` is invariant, so code holding the object as a `FrameDetections` could
  append a plain `Detection` into a tracked frame, and pydantic does not validate `.append()`.
  Changed the base field to `Sequence[Detection]`; the read-only interface makes the override sound
  and blocks the mutation. Runtime container is still `list`.
- Writing the weights test immediately exposed a **fourth upstream bug**: `thop`, pulled in by
  `yolov5`, pulled in by PytorchWildlife, calls `distutils.version.LooseVersion` at import. Under
  pytest's `filterwarnings = ["error"]` that `DeprecationWarning` becomes a hard error, so the
  entire inference stack could not be imported *from a test at all* — while working fine as a
  script. This is exactly the latent risk noted as finding 7, and it bit within minutes of there
  being a test to bite.
- My first `test_tracked_frame_rejects_plain_detections` asserted the error mentions `track_id`; it
  actually names `TrackedDetection`. Behaviour was right, assertion was wrong.

**Exact errors**
```text
error: Incompatible types in assignment (expression has type "list[TrackedDetection]",
base class "FrameDetections" defined the type as "list[Detection]")
note: Consider using "Sequence" instead, which is covariant

DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  .venv/lib/python3.11/site-packages/thop/profile.py:12: in <module>
    if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
```

**Interpretation**
Every defect this audit found was silent. The schema accepted data and dropped part of it; the
marker was registered and selected nothing; the guards were unguarded. That is the same pattern as
the previous audit (a value accepted, stored, then not used), which suggests it is the failure mode
this codebase is prone to, and that "it ran without error" is the weakest possible evidence here.

The variance error is worth remembering: mypy caught a real correctness hole that no test would
have, because the unsound path required `.append()` on a downcast reference.

**Next action**
Phase 1b. Obtain PanAf500 access, select 5–10 clips against the axes in `docs/obsidian/05 Technical/dataset.md`, fill
`data/sample_manifest.csv` (schema now in `manifest.py`, digests from `provenance.file_sha256`),
then implement `data/manifest.py` loading before `data/video.py`, tested against synthetic video.

---

## 2026-07-25 — Phase 1: PanAf500 acquired, MegaDetector run, accuracy measured

**Objective**
Obtain PanAf500 access, run pretrained MegaDetector V6 over real clips, produce annotated video, and
measure accuracy against ground truth well enough to inform a fine-tuning decision.

**Hypothesis / question**
From `05 Technical/model.md`, written before any run: night/infrared, occlusion, and small distant
subjects were expected to degrade detection. Which failure mode actually dominates, and is it
uniform or concentrated?

**Environment**
- Machine / OS: macOS 15.5, Apple M1, arm64
- Device: **mps:0, verified** by inspecting tensor placement (not by trusting the library)
- Python: 3.11.15
- Working tree clean? No — dirty during the run; recorded in run metadata

**Data / clip IDs**
10 clips downloaded from the Bristol deposit (23.4 MB); **3 processed** this session:
`FgJpFLxSmH` (gorilla, daylight, camera interaction/running, some small subjects),
`RHH9DDfWZa` (chimpanzee, **infrared night**, hanging/climbing_up, 35 empty frames),
`1xDXmshd5P` (chimpanzee, climbing_down/climbing_up/sitting_on_back).
Checksums verified before the run; manifest at `data/sample_manifest.csv`.

**Model and variant**
- Framework: PyTorch-Wildlife 1.3.0
- Model: MegaDetectorV6
- Variant: `MDV6-yolov9-c`
- Confidence threshold: 0.20 · IoU threshold for matching: 0.50

**Configuration**
`configs/base.yaml`, frame_stride 1 (every frame).

**Commands run**
```bash
uv run python scripts/fetch_panaf500.py --count 10 --pool 150
uv run --extra inference panaf-phase1 detect -c configs/base.yaml --clips 1 --frames 5 --overwrite
uv run --extra inference panaf-phase1 detect -c configs/base.yaml --clips 3 --overwrite
```

**Observations**

The deposit turned out to expose a **browsable file tree**, so clips can be fetched individually
(~1–6 MB each) instead of the 42.2 GiB archive. That is what made a targeted 10-clip sample
practical, and it retires the "no download script" caveat that had stood since the scaffold.

The annotation schema is now **verified from real files**, closing the longest-standing unknown in
this project. Two traps were found and handled in `data/annotations.py`:
`frame_id` is **1-based** (everything else here is 0-based), and behaviour labels use **underscores**
on disk (`climbing_up`) while the onboarding PDF writes them as prose.

The device bug fired exactly as documented, on the first real run:
`PyTorch-Wildlife ignored device='mps' (weights on 'cpu'); forcing it` → `weights forced onto mps:0`.

**Quantitative results**

3 clips, 1080 frames, 1660 annotated boxes, at confidence 0.20 / IoU 0.50:

| | Precision | Recall | F1 | mean IoU |
|---|---|---|---|---|
| **Overall** | **0.854** | **0.411** | **0.555** | 0.831 |
| FgJpFLxSmH | 0.876 | 0.673 | 0.762 | 0.900 |
| 1xDXmshd5P | 0.793 | 0.299 | 0.435 | 0.777 |
| RHH9DDfWZa | 1.000 | 0.009 | 0.018 | 0.816 |

TP 682 / FP 117 / FN 978. Only **7** false positives across **67** frames containing no ape.

Recall by behaviour: `hanging` 0.000 (0/213) · `climbing_up` 0.017 · `sitting_on_back` 0.110 ·
`climbing_down` 0.182 · `walking` 0.504 · `standing` 0.582 · `running` 0.889 ·
`camera_interaction` 1.000.

Recall by subject size: small 0.104 · medium 0.629 · large 0.395.

Throughput: ~210 ms inference per frame on MPS, ~2.2 min per 360-frame clip.

**Qualitative results**

Inspected frames directly rather than trusting the numbers alone. `FgJpFLxSmH` frame 100 shows two
gorillas detected at 0.92 and 0.74 matching ground truth, with a **distant third gorilla missed** —
the small-subject failure, visible. `RHH9DDfWZa` frame 0 is a **monochrome infrared night frame**: a
dark chimpanzee hanging against dark bark, ground truth present, no detection. That inspection is
what confirmed the 0.009 recall was real difficulty rather than a pipeline bug.

Saved to `reports/figures/FgJpFLxSmH_frame100.png`.

**Failures and dead ends**

- **My evaluator was wrong on the first run.** A 5-frame smoke run reported recall 0.013 with
  precision 1.000. Cause: it compared the 5 processed frames against all 360 annotated frames,
  counting 355 never-processed frames as misses — so "recall" was really the fraction of frames
  processed. Fixed to intersect processed frames with annotated ones; regression test added. Had
  this gone unnoticed, every strided or capped run would have under-reported recall.
- The clip selector estimates frame size from the largest box extent (annotations carry no
  dimensions), so its "large subject" label for `RHH9DDfWZa` (57.8%) was wrong — true relative area
  is 13.9% against the real 720×404. Selection was still sound; the reported figure was not. The
  by-size table in the write-up flags the resulting confound.
- The overlay initially risked reading as though the model produced behaviour labels. Resolved by
  colouring predictions and ground truth differently and burning a legend into every frame.

**Exact errors**
```text
# The upstream device bug, caught and corrected at runtime:
WARNING panaf_ape_detection.inference.megadetector:
  PyTorch-Wildlife ignored device='mps' (weights on 'cpu'); forcing it
INFO    panaf_ape_detection.inference.megadetector: weights forced onto mps:0
```

**Interpretation**

MegaDetector V6 is **precise but insensitive** here: it rarely invents an ape, and misses about three
in five. Crucially the failure is **not uniform** — it concentrates in arboreal postures, infrared
night footage, and small distant subjects. That shape matters more than the headline F1, because it
says the gap is domain-specific rather than general weakness.

The predictions in `model.md` held up: night/infrared, occlusion and small subjects were all
confirmed as failure modes. What was not predicted is how *absolute* the arboreal failure is —
0 of 213 `hanging` instances is not degradation, it is blindness.

Before any fine-tuning, the cheap experiments come first: a confidence-threshold sweep (free —
`panaf-phase1 evaluate` recomputes from saved detections) and a variant comparison (config-only).
3 purposively-hard clips cannot support a claim about PanAf500 as a whole.

**Next action**
Run the remaining 7 clips on the Colab T4 using `notebooks/phase1_colab.ipynb`, then a threshold
sweep over the saved detections before deciding anything about fine-tuning.
