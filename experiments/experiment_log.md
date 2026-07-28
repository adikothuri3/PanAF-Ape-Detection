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

---

## 2026-07-26 — Tracking implemented; all 10 clips run twice; two earlier conclusions overturned

**Objective**
Close Phase 1 step 4 (tracking), run the complete 10-clip sample rather than 3, and separate what
the detector does from what the tracker does.

**Hypothesis / question**
Written down before the run: **recall is 0.411, so tracks will fragment.** A tracker cannot
associate a detection that was never made, so many short tracks per individual were expected, and
that should be read as a consequence of detection recall rather than a tracker defect.

**Environment**
- Machine / OS: macOS 15.5, Apple M1, arm64
- Device: **mps:0, verified** by inspecting tensor placement
- Python: 3.11.15
- Wall clock: ~20 min per 10-clip pass (`elapsed_seconds` in run metadata)

**Data / clip IDs**
**All 10** manifest clips, every frame: 3600 frames, 4985 annotated boxes, 23 annotated individuals.

**Model and variant**
- PyTorch-Wildlife 1.3.0 · MegaDetectorV6 · `MDV6-yolov9-c`
- Confidence 0.20 · IoU 0.50 · ByteTrack activation 0.20, buffer 30, match 0.80, 24 fps
- `minimum_track_length: 5`

**Commands run**
```bash
# pass A -- detector only, artifacts/no-tracking/
uv run --extra inference panaf-phase1 detect -c artifacts/detector_only.yaml --overwrite
# pass B -- detector + ByteTrack, artifacts/
uv run --extra inference panaf-phase1 detect -c configs/base.yaml --overwrite
```

**Observations**

| | Detector only | + ByteTrack |
|---|---|---|
| Precision | 0.8743 | 0.9165 |
| Recall | 0.3864 | 0.3525 |
| F1 | 0.5359 | 0.5091 |
| Pooled mean IoU | 0.8252 | 0.8322 |
| TP / FP / FN | 1926 / 277 / 3059 | 1757 / 160 / 3228 |
| FP on the 74 empty frames | 7 | 3 |

Tracking: 23 individuals, 29 predicted tracks, **17 ID switches**, mean fragmentation **1.35**,
pooled coverage **0.353**, 7 mostly-tracked, 9 mostly-lost.

**The hypothesis was wrong in an informative way.** Fragmentation is *low* (1.35, ideal 1.00), not
high. ByteTrack holds an ape it can see. What low recall caps is **coverage** (0.353 ≈ recall
0.386), not track continuity. Three clips produced no tracks at all because they produced almost no
detections.

**Two conclusions from the 2026-07-25 write-up did not survive the full sample**

1. `hanging` recall was reported as **0.000 (0/213)** and described as "blindness". Over 1346
   instances it is **0.102**. Bad, but not blindness — the zero was one clip.
2. The by-size table showed a "large subject dip" (0.395), flagged then as *possibly* confounded.
   With 10 clips it is **0.214**, and the confound is now measurable: 1335 of 1544 large boxes come
   from four single-ape, close-up, badly-lit tree clips. In the two clips containing large *and*
   small subjects together, large-subject recall is **0.856 and 0.899** — the best of any band.
   **The failure is contrast, not size.**

**Failures and dead ends**

- **`configs/base.yaml` still had `tracking.enabled: false`.** Only `colab.yaml` had been updated
  when tracking shipped. The first 10-clip run was launched, ran four minutes, and was producing
  *no tracks at all* while every document claimed tracking was on. Killed and restarted after
  confirming the tracker logs its settings on construction. Added a verification check that the two
  shipped configs must agree about which pipeline stages run.
- **`data.max_clips: 8` silently capped the "full" run at 8 of 10 clips.** Raised to 10, with a
  comment: this must never sit below the manifest size or a full run quietly omits clips.
- **The pooled `mean_iou` statistic was wrong.** `summarise()` averaged per-clip means, and a clip
  with *no* matched pairs has `mean_iou = 0.0` by construction — so three zero-match clips dragged
  the tracked figure to 0.579 and made it look as though the tracker had wrecked localisation. It
  had not: weighting by matched pairs gives 0.8252 detector-only versus 0.8322 tracked. Fixed to
  weight by true positives, with a regression test. **I nearly reported the artifact as a finding.**
- **Enabling tracking changes the detection metrics**, because `drop_short_tracks` runs before
  evaluation. This is why both passes exist. Reporting pass B alone as "MegaDetector's accuracy"
  would have been wrong.

**Exact errors**
```text
# mypy, on the tracked-detection narrowing in the runner -- list is invariant:
src/panaf_ape_detection/pipeline/runner.py:230: error: Incompatible types in assignment
  (expression has type "list[TrackedDetection]", variable has type "list[Detection]")  [assignment]
src/panaf_ape_detection/pipeline/runner.py:230: note: "list" is invariant
src/panaf_ape_detection/pipeline/runner.py:230: note: Consider using "Sequence" instead, which is covariant
```

**Interpretation**

The detector is **precise but insensitive** (P 0.874, R 0.386), and the failure is concentrated in
**low-contrast footage** — night, infrared, deep shade, blown-out backlight — with arboreal posture
and small size as correlates rather than causes. Two frames inspected directly confirm it:
`isfRigsIjO` frame 180 is a black chimpanzee against a black tree, obvious to a human, and produced
**zero detections in all 360 frames**; `XmOoOk9n7t` frame 180 is the same silhouette problem from
the opposite direction, backlit against blown-out white foliage.

`minimum_track_length: 5` trades away more recall than it buys precision (+0.042 P, −0.034 R, F1
falls). The removals are 3.3× enriched in false positives, so the filter is not thinning at random —
but with recall as the binding constraint it is too aggressive here. That is a config value, and
`panaf-phase1 track` re-runs tracking over saved detections in seconds.

**Next action**
Confidence sweep — see the next entry, which is where it went wrong and then went right.

Full analysis: [findings 2026-07-26](../reports/phase1_findings_2026-07-26.md).

---

## 2026-07-26 (later) — The threshold was the problem, and a second ignored parameter

**Objective**
Sweep the confidence threshold. Precision 0.874 against recall 0.386 said the operating point was
mistuned; this is the cheapest possible experiment and needed doing before anything else.

**Hypothesis / question**
There is precision to spend. Lowering the threshold should trade some of it for recall. Unknown:
whether the recovered detections land on the easy clips (uninteresting) or the badly-lit ones
(important).

**Commands run**
```bash
# boxes below 0.20 were never saved, so a downward sweep needs a re-run
uv run --extra inference panaf-phase1 detect -c artifacts/sweep_conf005.yaml --overwrite
for c in 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50; do
  uv run panaf-phase1 evaluate -c artifacts/sweep_conf005.yaml --confidence $c
done
```

**The first attempt was silently a no-op**

The 0.05 run produced **byte-identical output to the 0.20 run**: 2203 detections, minimum confidence
0.2002, zero boxes below 0.20. The sweep at 0.05, 0.10, 0.15 and 0.20 returned the same three
numbers to four decimal places, which is what gave it away.

Cause, confirmed by reading the installed source:

```text
YOLOV8Base.single_image_detection(self, img, img_path=None, det_conf_thres=0.2, id_strip=None)
```

The adapter called `single_image_detection(frame)` with no `det_conf_thres`, so **every run in this
repository's history inferred at PyTorch-Wildlife's default of 0.2**, regardless of
`model.confidence_threshold`. The configured value was only ever applied as a post-inference filter.

Because the configured value *was* 0.2, every previously reported number is still correct — the two
thresholds agreed by coincidence. What was impossible was going below it.

**This is the second parameter this library accepts and ignores**, after `device=`. Same shape:
accepted, stored, never applied where it matters; fails silently; only detectable by observing
behaviour rather than the absence of an exception. Fixed by passing the configured threshold to
every inference call, with three weights-free tests — including one that asserts the recorded
default still matches the installed library's signature, so an upstream change breaks a test rather
than quietly shifting results.

**Results, after the fix**

| Confidence | Precision | Recall | F1 |
|---|---|---|---|
| **0.05** | 0.6283 | **0.5633** | **0.5940** |
| 0.10 | 0.7914 | 0.4558 | 0.5784 |
| 0.15 | 0.8571 | 0.4116 | 0.5562 |
| 0.20 | 0.8743 | 0.3864 | 0.5359 |
| 0.25 | 0.8868 | 0.3661 | 0.5182 |
| 0.30 | 0.9030 | 0.3472 | 0.5016 |
| 0.40 | 0.9176 | 0.3081 | 0.4613 |
| 0.50 | 0.9426 | 0.2800 | 0.4318 |

F1 rises monotonically as the threshold falls and **has not turned over at 0.05**, so the optimum in
this range is the lowest point tested and may be lower still.

**The recovered detections are exactly where they matter**

| Clip | R@0.20 | R@0.05 | Δ |
|---|---|---|---|
| `XmOoOk9n7t` (backlit) | 0.361 | 0.892 | **+0.531** |
| `z97mIEQzcL` (deep shade) | 0.044 | 0.492 | **+0.447** |
| `isfRigsIjO` (near-dark) | **0.000** | 0.433 | **+0.433** |
| `RHH9DDfWZa` (infrared) | 0.009 | 0.265 | **+0.255** |
| six daylight clips | — | — | +0.05 to +0.11 |

`hanging` 0.102 → **0.527**. The "large" size band — which §5.2 of the write-up showed is really the
badly-lit clips — 0.214 → **0.601**.

**Interpretation**

The earlier entry concluded that failure "concentrates in low-contrast footage". That is true, and
the mechanism is now clear and much less dramatic: **low contrast depresses the confidence score; the
threshold then discards the detection.** The detector was not blind to the ape in `isfRigsIjO` — it
was scoring it between 0.05 and 0.20 for all 360 frames.

Had the bug not been caught, the conclusion would have been "MegaDetector cannot see apes in
low-contrast footage; fine-tune or preprocess" — a month of work aimed at a problem that a one-line
YAML change substantially fixes. **Three claims in this study's lineage have now turned out to be
artifacts rather than findings** (3-clip `hanging` blindness, the large-subject dip, the pooled
mean-IoU collapse). Each was plausible, and each would have been expensive to act on.

**Tracked run at 0.05 — the win does not survive the tracker**

Hypothesis: precision 0.628 is low, but `minimum_track_length` is a temporal filter that should
remove exactly the single-frame low-confidence noise a lower threshold admits, so tracking at 0.05
should beat both.

**Wrong, and the reason is worth more than the hypothesis was.** Tracking the 0.05 detections gives
coverage 0.767 / 0.301 / 0.308 on the clips that track at all — identical to 0.20 to three decimals —
with *more* ID switches (9uIpm1xLeI 4 → 8). The clips that produced no tracks still produce none.
Lowering the detector threshold buys the tracker exactly nothing.

Cause, read from the installed `supervision` source rather than guessed:

```python
inds_low = scores > 0.1                                    # <= 0.1 discarded outright
self.det_thresh = self.track_activation_threshold + 0.1    # new tracks need activation + 0.1
```

Neither is configurable. **`track_activation_threshold` does not do what its name suggests** — it
does not open the tracker to everything the detector kept, which is exactly what this repo's
`bytetrack.py` docstring claimed. Docstring corrected.

Isolated it from my own filter with `panaf-phase1 track --min-track-length 1|2|3|5` over the saved
0.05 detections: the zero-track clips stay at zero for every value. It is ByteTrack's floor, not
`drop_short_tracks`.

The floor lands precisely on the recovered detections:

| Clip | Detections @0.05 | ≤0.10 (discarded) | 0.10-0.15 (cannot start a track) | >0.15 (usable) |
|---|---|---|---|---|
| `isfRigsIjO` | 742 | 649 (87%) | 81 | **12 (2%)** |
| `RHH9DDfWZa` | 98 | 73 | 20 | **5 (5%)** |
| `FgJpFLxSmH` | 716 | 80 | 39 | 597 (83%) |

**Third instance of the same failure mode in one day** — after `device=` and `det_conf_thres`, a
parameter that is accepted and does not mean what its name implies. The first two were upstream's;
this one I built on top of, by assuming a configurable threshold was the only threshold.

**Next action**
Extend the sweep below 0.05 for the detector-only case. For tracking, the question is no longer the
detector threshold but whether ByteTrack's 0.1 floor makes it the wrong tracker for footage whose
hard cases score at 0.05-0.10. Then the variant comparison.

---

## 2026-07-27 — Variant comparison: `MDV6-yolov10-e` nearly doubles recall for free

**Objective**
Answer the last cheap question before any fine-tuning: is the detection gap a capacity limit, or
domain mismatch? Run the larger variant over the identical sample and compare.

**Hypothesis / question**
Written down before the run: the deciding number is not recall but the **score distribution** —
whether the larger model scores the low-contrast subjects *above* ByteTrack's 0.15 floor. A variant
that only finds fainter boxes changes nothing downstream.

**Environment**
Colab A100, `cuda` (verified), both arms in one session. yolov9-c 99 s, yolov10-e 121 s for
10 clips / 3600 frames.

**Commands run**
```bash
# both arms, detector-only at 0.05 so they compare at every threshold
panaf-phase1 detect -c configs/colab-sweep-conf005.yaml --overwrite
panaf-phase1 detect -c configs/colab-variant-yolov10e.yaml --overwrite
# then, locally, over the saved detections
panaf-phase1 evaluate -c <arm> --confidence 0.05|0.10|0.20|0.30|0.40|0.50
panaf-phase1 track -c <arm>
```

**Results — at the same 0.20 operating point**

| | yolov9-c | yolov10-e |
|---|---|---|
| Precision | 0.8743 | 0.8621 |
| Recall | 0.3864 | **0.7446** |
| F1 | 0.5359 | **0.7991** |

Recall nearly doubles at **unchanged precision**. At 0.05: P 0.6337 → 0.6419, R 0.5639 → 0.8221.

**The predicted deciding number came out decisively.** Detections clearing 0.15: **54% → 73%**.

By behaviour (both at 0.05): `sitting` 0.326 → 1.000, `climbing_up` 0.180 → 0.700,
`sitting_on_back` 0.104 → 0.364, `hanging` 0.531 → 0.737, `standing` 0.670 → 0.943. Every class
improves. By size: small 0.441 → 0.710, medium 0.693 → 0.945, large 0.600 → 0.850 — a uniform
~+0.25, so this is not a large-subject effect.

**Tracking, measured rather than inferred**

| | yolov9-c | yolov10-e (activation 0.20) |
|---|---|---|
| Coverage of annotated individual-frames | 0.351 | **0.728** |
| Mostly tracked / 23 | 6 | **13** |
| Mostly lost / 23 | 9 | **2** |
| Predicted tracks | 30 | 59 |
| ID switches | 19 | 46 |

Coverage doubles; identity stability worsens. Both are consequences of having more detections, and
the honest reading is that the bottleneck has moved from detection to association.

Running ByteTrack with activation 0.20 rather than 0.05 is better on every axis (coverage 0.728 vs
0.718, switches 46 vs 54, mostly-lost 2 vs 3): faint boxes should *extend* tracks, not start them,
which is what ByteTrack was designed for.

**The threshold problem largely dissolved.** yolov9-c's F1 rose monotonically as the threshold fell,
so the optimum was never found. yolov10-e peaks at **0.20–0.30**, the shipped default, and degrades
gently either side.

**Checks before believing any of it**

- Three clips report recall 1.000, which is the kind of number that is usually an artifact. Checked
  detections per frame: `RHH9DDfWZa` 417 detections for 325 annotated boxes at 1.16/frame on a
  one-ape clip, precision 0.779. Not box flooding — the recall is real.
- The A100 baseline agrees with the earlier local MPS baseline: P 0.6337 vs 0.6283, R 0.5639 vs
  0.5633 at 0.05. **The two execution paths do not disagree**, so no published number is
  device-specific.
- `isfRigsIjO` is the one regression: recall 0.492 → 0.364. But yolov9-c drew **728 boxes on a
  one-ape clip at precision 0.243** — scattering boxes into the dark. yolov10-e draws 164 at
  precision 0.799. A model that stopped guessing, not a model that got worse.

**Failures and dead ends**

- The first download contained the **baseline arm twice**: Drive names every arm's subfolder
  `metrics`/`detections`, so two arms produce identically-named zips. Caught by reading
  `model.variant` inside the files rather than trusting filenames. Had it gone unnoticed, the
  "comparison" would have compared a run against itself and reported no difference.

**Interpretation**

The pretrained gap was **capacity, not domain mismatch.** That is the cleanest possible answer to
the question Phase 1 existed to ask, and it arrived from a one-line config change rather than
GPU-hours of training. It also retires the low-contrast narrative: night, shade and backlit clips go
to 100%, 100% and 100% recall with a bigger model.

**Next action**
Adopt `MDV6-yolov10-e` as the default variant, re-run the annotated videos from it, and re-open the
tracking question — association, not detection, is now the limit.

---

## 2026-07-27 (later) — Re-ran all 10 clips on the adopted variant

**Objective**
Regenerate the annotated videos and headline numbers from `MDV6-yolov10-e`, so the repository's
outputs come from the model it actually recommends.

**Environment**
Local, verified `mps:0`. **~1.05 s/frame inference** against ~0.25 s for `yolov9-c` -- YOLOv10x is
roughly 4x the work, and MPS does not parallelise it the way the A100 does (which did the same 10
clips in 121 s). Total wall clock about 70 minutes.

**Results — 10 clips, 3600 frames, 4985 annotated apes, confidence 0.20, tracked**

| | `yolov9-c` | `yolov10-e` |
|---|---|---|
| Precision | 0.9165 | **0.9308** |
| Recall | 0.3525 | **0.7149** |
| F1 | 0.5091 | **0.8087** |
| Pooled mean IoU | 0.8322 | 0.8519 |
| Track coverage | 0.351 | **0.715** |
| Mostly-lost individuals | 9 / 23 | **2 / 23** |

3564 of 4985 apes found against 1757 before, at *higher* precision. Four clips now exceed 0.95
recall, including `RHH9DDfWZa` at 0.963 -- the infrared night clip that scored **0.009** at the start
of this project. Inspected frame 100 directly: a clean, tight prediction box on the hanging chimp,
where the old model produced nothing across all 360 frames.

Weak spots that remain: `isfRigsIjO` 0.211 recall in near-darkness (though precision 1.000 -- what it
finds is right), and `zvwY5xoIli` 0.348 on small distant subjects.

**Failures and dead ends**

- **Two background runs were killed part-way**, at 8 of 10 clips. Recovered by running detached with
  `nohup`.
- **Resuming without `--overwrite` would have silently skipped the two unfinished clips**, because
  the runner skips any clip that already has detections *and* a video -- regardless of which model
  wrote them. The stale outputs had to be deleted first. A plain resume would have "completed" while
  leaving two clips on the old model.
- **A metrics directory updated clip by clip is a mixture, and nothing showed it.** Reading the
  numbers mid-run pooled 3 new clips with 7 stale ones and produced a figure describing no model at
  all. Each file was individually valid. `detections/*.json` records the variant -- which is the only
  reason the affected clips could be identified -- but `metrics/*.json` did not.

  Fixed: `ClipEvaluation` now carries `model_variant`, it is written into every metrics file, and
  `reporting.variants_in()` exposes a mixture. The notebook **halts** rather than pooling one. Three
  tests cover it. Note that the metrics files from *this* run pre-date the field and report an empty
  variant; the detections files identify them.
- My "all 10 videos are there" check counted two `_h264` re-encodes made earlier for playback
  testing. The glob was `*.mp4` rather than `*_annotated.mp4`. Corrected; the stray files are removed.

**Interpretation**

The adopted variant reproduces its Colab result on entirely different hardware: A100 detector-only
gave P 0.862 / R 0.745, local MPS tracked gives P 0.931 / R 0.715. The two paths agree, so nothing
here is device-specific.

**Next action**
Tracking, not detection, is the remaining limit: 36 ID switches across 23 individuals. The free
experiments come first -- ByteTrack activation and `minimum_track_length`, both re-run over saved
detections in seconds.

---

## 2026-07-27 (later still) — Tracking: 36 ID switches → 4, and coverage finally broke its ceiling

**Goal**
Take the "free experiments" from the last entry as far as they go: make tracking good rather than
merely working. Tune, do not fine-tune — Phase 1 stays pretrained-inference only.

**The measurement that reframed the whole thing**

Frame-weighted tracking coverage was **0.7149**. Detection recall was **0.7149**. Equal to four
decimal places, over `artifacts/metrics/` and `artifacts/metrics/tracking/`.

That is not a near-miss, it is an identity: every true-positive detection was already inside some
track, and no track ever covered a frame the detector missed. So **no tracker setting can raise
coverage** — it is pinned to the detector by construction. The remaining problem was never coverage;
it was identity (54 tracks for 23 apes, 36 switches), plus the fact that the only way past the
ceiling is to *interpolate* boxes the detector never produced.

I had been about to tune for coverage. That would have been a week of moving a number that cannot
move.

**Two bugs found before any tuning started**

- **`panaf-phase1 evaluate` was broken against the repository's own default artifacts.** `cli.py`
  did `Detection(**d)`, `Detection` is `extra="forbid"`, and every record written with tracking on
  carries `track_id` and `behavior_label`:

  ```
  ValidationError: 2 validation errors for Detection
  track_id
    Extra inputs are not permitted [type=extra_forbidden, input_value=1, input_type=int]
  ```

  The whole test suite passed throughout, because every CLI fixture was written with tracking
  *off*. Fixed with a shared `reporting.detection_fields()`, and the fixture now writes track ids.
- **A resumed run reported no metrics at all.** `run_clip`'s skip-if-present branch returned
  `evaluation=None`. It would have silently hollowed out the 500-clip Colab run this work depends
  on — the exact situation a dropped session produces. It now re-measures from the saved
  detections, which costs no GPU and no video decode. Verified against the baseline: 10/10 tracking
  metric files byte-identical, and detection metrics now additionally carry the `model_variant` the
  old files were missing.

**What was actually wrong with the tracker**

Three of ByteTrack's five knobs were unreachable from configuration, and the fourth
(`track_activation_threshold`) was welded to `model.confidence_threshold`, so it could not be varied
against it. All four are now config fields, defaulting to the previously hardcoded values so an old
config reproduces the old behaviour exactly.

Then the upstream floor. `supervision` has `inds_low = scores > 0.1` as a **literal**: on the
10-clip cache at confidence 0.05 that discards **1,185 of 6,384 detections, 18.6%** — thrown away
by the low-score pass that is the entire reason ByteTrack was picked over SORT. `ScoreFloor` maps
scores into `[floor, 1]` before association and inverts exactly afterwards, so ByteTrack sees them
and no artifact ever records a rescaled score. A test asserts a tracked box still reports 0.42 after
being fed through a 0.15 floor.

**Method**

Staged sweep over `artifacts/colab/variant-yolov10e/detections/` (10 clips, detector-only,
confidence 0.05, produced on an A100 — so nothing here is a CPU-inference result). New
`panaf-phase1 track-sweep`; **0.30 s per arm** for 10 clips with `-j 8`, so this cost minutes.

Ranked by a new metric, **identity coverage**: the share of an ape's frames held by its *single
best* track. Plain coverage rewards chopping one ape into several tracks; counting switches alone
rewards merging two apes into one. Identity coverage is pushed down by both.

`track_purity` and `id_merges` were added first, deliberately, **before** any step that joins tracks
together. Every other metric here improves when two apes are merged into a single track — the
switches vanish, fragmentation reaches a perfect 1.00, coverage does not move. Without purity the
numbers would have endorsed exactly that.

**Result** — shipped settings vs `configs/tracking-candidate.yaml`, same 10 clips:

| | shipped | candidate |
| --- | --- | --- |
| Identity coverage | 0.6449 | **0.7561** |
| Coverage | 0.7284 | 0.7639 |
| ID switches | 46 | **4** |
| Fragmentation | 2.57 | **1.13** |
| Track purity | 0.9963 | 0.9975 |
| Tracks holding 2+ apes | 3 | 2 |
| Jitter | 0.0230 | **0.0042** |
| Mostly tracked | 13/23 | 16/23 |
| Detection precision | 0.9530 | 0.9556 |
| Detection recall | 0.7246 | 0.7639 |

(46 rather than 36 switches is the same shipped settings re-tracked over the 0.05 cache instead of
the 0.20 one — the substrate every arm here shares. The 0.20 baseline is 36.)

**What actually mattered, in order**

1. **Activation threshold 0.20 → 0.40**, the single largest effect. Because supervision needs
   `activation + 0.1` to *start* a track, this means only a 0.50 detection opens a new identity
   while anything down to the score floor can still extend one. Spurious track creation was the main
   source of fragmentation.
2. **Detect permissively, let the tracker filter.** Raw precision at 0.05 is 0.64 — but the
   *tracked* output is 0.9556, better than the 0.9308 the 0.20 pipeline reports, at higher recall.
   This inverts the assumption that the detector threshold should be tuned for precision.
3. **Interpolation is what broke the coverage ceiling**, 0.7246 → 0.7639, for 195 synthesised boxes
   and a precision cost of 0.0002. Every one is flagged `interpolated` so Phase 2 can exclude boxes
   no detector saw.
4. **Smoothing cut jitter by 82% and slightly *raised* both precision and recall** — steadier boxes
   clear IoU 0.5 more often. I expected it to be cosmetic only.

**Dead ends and corrections**

- **Stitching turned out to be redundant.** Every `stitch_max_gap` from 0 to 96 gave byte-identical
  results, which looked like a broken function. It is not: on the untuned baseline it cuts tracks
  59 → 46 and fragmentation 2.57 → 2.00. It does nothing at the tuned operating point because
  `lost_track_buffer: 120` already reconnects the same fragments, with a motion model inside the
  tracker rather than a repair afterwards. Kept, defaulted off, and documented as a result.
- **`detection_floor` does nothing**, 0.00 and 0.10 differing in the fourth decimal — because
  supervision discards ≤0.10 itself. Axis dropped after one sweep.
- **`minimum_consecutive_frames` looked like the answer early on** (switches 46 → 18 on its own,
  from a parameter that was never passed to the library at all) and then stopped helping once
  activation was raised. It costs coverage, and at activation 0.40 there is little flicker left for
  it to remove. Ended at 1.
- **My first smoothing implementation was wrong and a test caught it.** The window shrank
  *asymmetrically* at the ends of a track, so a box moving at constant velocity came out at 5.0
  where it should have been 0.0 — every track's first and last boxes dragged inward. Now symmetric.

**Interpretation**

Tracking is no longer the bottleneck; the detector is, again. Coverage 0.7639 against detection
recall 0.8221 on this cache means association plus interpolation now recovers most of what the
detector finds, and 4 switches across 23 individuals is close to the floor for this footage.

**This is not adopted, and must not be.** Every number above was tuned *and* measured on the same
10 clips, and those 10 were purposively chosen to be hard, so they are not a random sample either.
`configs/tracking-candidate.yaml` records the settings and stays out of `base.yaml`.

**Next action**
Run `configs/colab-full500.yaml` — all 500 PanAf500 clips, detector-only at 0.05, ~100 minutes on
an A100 by the 121 s/10 clips measured on 2026-07-27, plus ~1.1 GB of download. Then re-sweep on the
dataset's own `train` split, confirm on `validation`, and touch `test` exactly once. Adopt only if
it holds there, and report full-500 and hard-10 numbers separately so the easier clips do not
flatter the result.

PanAf20K cannot extend this: only the 500-clip subset carries per-frame `ape_id`, so identity
metrics are undefined on the rest, not merely expensive.

---

## 2026-07-28 — The tracking candidate validated on all 500 clips, and one regression it hid

**Goal**
Decide whether the tracker settings tuned on 10 clips are real, by measuring them on clips they
were never tuned on. Colab A100, all 500 PanAf500 clips, detector-only at confidence 0.05, then
both trackers over the same cache at no GPU cost.

**Result** — 500 clips, **874 annotated individuals**:

| | shipped | candidate |
| --- | --- | --- |
| Identity coverage | 0.7436 | **0.8197** |
| Coverage | 0.8621 | 0.8709 |
| ID switches | 1910 | **409** |
| Fragmentation | 2.65 | **1.30** |
| Track purity | 0.9970 | 0.9913 |
| Tracks holding 2+ apes | 59 | **79** |
| Jitter | 0.0161 | **0.0037** |
| Mostly tracked | 640 | 670 |
| Mostly lost | 59 | 65 |

**It holds, and it was oversold.** The 10-clip margin was identity coverage +11.1pp
(0.6449 → 0.7561). Over 500 clips it is **+7.6pp** (0.7436 → 0.8197). So roughly a third of the
apparent gain was fitting the tuning set, and two thirds is real. ID switches fall 79% here against
91% on the 10 clips. Both arms also score higher in absolute terms than on the 10-clip sample,
which is the expected direction: those 10 were purposively selected to be hard, so they are not a
random sample of anything.

**The regression that only appeared at scale.** Tracks holding two or more apes went **59 → 79**
and purity 0.9970 → 0.9913. As a rate that is worse than it reads: 59 of ~2316 shipped tracks
(2.5%) against 79 of ~1136 candidate tracks (7.0%). On the 10-clip sample merges had *improved*,
3 → 2, so nothing before this run hinted at it.

This is the one metric that must not be traded away, because merging two animals into a single
track is the failure identity coverage can partly *reward* -- both apes then count the merged track
as their dominant one. The selection rule fixed before any of this was measured was "maximise
identity coverage, subject to merges not exceeding baseline". **By that rule the candidate fails**,
and the rule does not get dropped because the other nine numbers are good.

**The sweep was inconclusive, and said so.** 108 arms, 2654.8 s (24.58 s per arm over 500 clips).
The command reported *"the configured settings are not among the arms"* -- correctly: the grid
swept activation over 0.10-0.30 while the candidate uses 0.40, so the candidate was never an arm
and there was no reference point. One thing is still established: **no arm in that range beat it**,
best 0.8018 against the candidate's 0.8197. Activation 0.40 is better than anything below it.

**Changes made in response**

- `track-sweep --max-merges N` rejects arms above a merge ceiling instead of merely printing the
  column. Enforcing the stated rule by eye is how it gets quietly dropped.
- `configs/sweeps/around-candidate.yaml` brackets the candidate rather than re-testing a range
  already ruled out: activation 0.35-0.50, buffer 30/60/120, match 0.7-0.9, minimum track length
  8/16 -- with the candidate itself as an arm, so it competes against its neighbours.

**Interpretation**

Tracking is no longer the bottleneck. 409 switches across 874 individuals, fragmentation 1.30, and
coverage 0.8709 against detection recall on this cache -- the tracker now recovers most of what the
detector finds, and the smoothing is measurable rather than cosmetic (jitter down 77%).

What remains is a purity question, not a coverage one: whether some setting near the candidate
keeps most of the +7.6pp while holding merged tracks at or below the shipped 59. That is a CPU
sweep over the cache already on disk, not another GPU run.

**Next action**
Run `configs/sweeps/around-candidate.yaml` against the 500-clip cache with `--max-merges 59`
(~30 min, no GPU). If an arm clears the ceiling and keeps most of the gain, adopt that into
`base.yaml` and `colab.yaml`. If none does, adopt the candidate anyway and **report the merge
regression in the write-up as a cost**, rather than choosing settings that hide it.
