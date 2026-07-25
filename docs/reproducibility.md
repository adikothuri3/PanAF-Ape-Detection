# Reproducibility

The contract: **another researcher should be able to clone this repository, obtain the permitted
sample data, follow the README, and produce an equivalent annotated clip.**

"Equivalent", not "bit-identical" — see [Honest limits](#honest-limits-of-this-contract) below.

## The nine commitments

### 1. The environment is locked

`uv.lock` is committed and resolves the full dependency graph, including the `inference` extra.
`.python-version` pins Python 3.11. CI installs from the lockfile, so a dependency that breaks the
build breaks CI rather than someone's afternoon.

`requirements-colab.txt` is **generated from the lockfile**, never hand-maintained:

```bash
uv export --extra inference --no-hashes --no-dev --format requirements-txt -o requirements-colab.txt
```

`scripts/verify_repository.py` fails if it stops looking like a `uv export`, and if `uv.lock` drifts
from `pyproject.toml`.

### 2. Raw inputs are immutable

Nothing in this codebase writes to `data/raw/`. Files land there once and are never renamed,
re-encoded, trimmed, or cleaned in place. Derived data goes to `data/interim/`, `data/processed/`
or `artifacts/`.

This is what makes checksums meaningful: a digest over a directory that code writes into proves
nothing.

### 3. Clip selection is a checksummed manifest

`data/sample_manifest.csv` records every clip with a SHA-256 for its video and annotation file, plus
a `selected_reason`. Runs will verify digests before processing and refuse to proceed on a mismatch.

A checksum mismatch means your inputs changed. Every result derived from them is suspect until you
find out why.

### 4. Configuration is versioned

Experiment settings live in `configs/*.yaml`, committed, strictly validated, with unknown keys
rejected. Changing a threshold produces a diff.

The narrow environment-variable overrides (`PANAF_DEVICE`, `PANAF_ARTIFACTS_DIR`, ...) exist for
things that legitimately differ per machine. Anything scientifically meaningful belongs in YAML,
where review can see it. The full resolved config is captured in run metadata regardless.

### 5. Every run records its git commit — and whether the tree was dirty

`RunMetadata.git_commit` and `RunMetadata.git_dirty`. The dirty flag is the important one: a commit
SHA from a modified working tree is an actively misleading record, because the SHA no longer
describes the code that ran. Results produced from a dirty tree should be treated as provisional
and re-run from a clean commit before being reported.

### 6. Seeds are set where they have an effect

`project.seed` is applied to Python, NumPy and PyTorch RNGs. Pretrained detection inference is
largely deterministic, so this matters less in Phase 1 than it will once sampling, augmentation or
stochastic tracking enter — setting it now costs nothing and avoids retrofitting.

Setting a seed is not the same as achieving determinism (see below).

### 7. Every run writes metadata

Schema: `RunMetadata` in [`../src/panaf_ape_detection/types.py`](../src/panaf_ape_detection/types.py).

> **The producer exists; no pipeline calls it yet.**
> `panaf_ape_detection.provenance.build_run_metadata()` assembles the record and
> `write_run_metadata()` persists it, both tested. `scripts/smoke_detect.py` produces a real one.
> What does *not* exist is a pipeline stage that runs inference over clips and writes it — so
> `artifacts/metadata/` is still empty in normal use.

| Field | Why it is there |
| --- | --- |
| `experiment_name` | Ties outputs to a log entry |
| `started_at_utc` | Timezone-aware UTC; local timestamps are ambiguous across machines |
| `git_commit`, `git_dirty` | Which code ran, and whether that claim is trustworthy |
| `config_snapshot` | The fully resolved config, after env overrides |
| `python_version`, `dependency_versions` | The environment as installed, not as locked |
| `platform` | OS and version |
| `device` | The device the weights are **verified** to be on — not the one requested. PyTorch-Wildlife ignores `device=` (see [model.md](model.md)), so the requested value is not evidence. Use `runtime.module_device()`. |
| `model_name`, `model_variant` | **Which model.** A result without this cannot be reproduced |
| `confidence_threshold` | A detection count is meaningless without it |
| `seed` | The seed actually used |
| `inputs` | Filename, SHA-256 and size per input file |
| `output_paths` | What was produced, so results trace back to their run |
| `elapsed_seconds` | Cost, for planning and for spotting a run that silently did nothing |

Note `inputs` stores **filenames only**, not absolute paths — metadata should not leak local
directory structure or reveal more of the dataset layout than necessary.

### 8. Hardware is documented

Device, OS and machine architecture go in run metadata; `panaf-phase1 doctor` reports the same for
the current environment. Timing numbers are meaningless without it, and some numerical differences
are explained only by it.

### 9. Generated results are never hand-edited

`artifacts/` is entirely git-ignored and entirely disposable. If a number is wrong, fix the code and
re-run — never touch the output file.

A hand-corrected result is indistinguishable from a fabricated one after a week. This applies to
detection records, metrics and figures alike. Figures in `reports/figures/` should be regenerable
from a recorded run.

## Honest limits of this contract

**GPU operations are not always deterministic.** Even with seeds fixed, cuDNN algorithm selection,
non-deterministic reduction ordering in floating-point kernels, and TF32 on Ampere-and-later GPUs
can all produce slightly different results between runs, and reliably different results between
different GPUs, driver versions, or CPU-vs-GPU execution. Confidence scores may differ in the third
decimal place; a detection sitting exactly on the threshold can therefore appear in one run and not
another.

`torch.use_deterministic_algorithms(True)` reduces this, at a real speed cost, and does not
eliminate cross-hardware variation.

So the contract is **explicability and repeatability, not bit-identity**:

- Same code, same config, same inputs, same machine → results that agree to within numerical noise.
- Different machine → results that agree qualitatively; a box on the same animal, a score that is
  close, possibly a different count at a borderline threshold.
- Any difference is *explicable* from the recorded metadata, rather than mysterious.

If a result depends on the third decimal place of a confidence score, the result is too fragile to
report. Choose thresholds that are robust to it.

## Other honest limits

- **Model weights are downloaded from Zenodo at runtime** and are not vendored. If that record
  becomes unavailable, the environment is reproducible but the model is not. Record the variant and
  the file name (see [`model.md`](model.md)) so the weights can at least be identified.
- **The dataset cannot be redistributed**, so reproduction requires the reader to obtain it
  independently under its own licence. The manifest checksums are what let them confirm they got
  the same bytes.
- **Colab runtimes drift.** Preinstalled package versions change without notice, which is exactly
  why the notebook installs from the exported lockfile rather than relying on what Colab ships.
- **A small purposive sample does not generalise.** Reproducing the run reproduces the observation,
  not a property of PanAf500.

## Checklist before reporting a result

- [ ] Working tree is clean and the commit is pushed.
- [ ] The config used is committed.
- [ ] Manifest checksums verify against the files on disk.
- [ ] Run metadata was written alongside the outputs.
- [ ] The model variant and confidence threshold appear in whatever you are writing.
- [ ] Hardware and device are recorded.
- [ ] The figure or number came from a recorded run and was not edited afterwards.
- [ ] An experiment-log entry exists, including anything that failed.
