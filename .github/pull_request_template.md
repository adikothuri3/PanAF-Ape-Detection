## What and why

<!-- What does this change do, and what problem does it solve? Link the issue or
     the experiment-log entry that motivated it. -->

## Type of change

- [ ] Pipeline implementation (a stage that did not exist before)
- [ ] Bug fix
- [ ] Configuration or environment change
- [ ] Documentation
- [ ] Tests or tooling
- [ ] Experiment log / findings write-up

## Checks

- [ ] `make quality` passes (lint, format, mypy, tests)
- [ ] `make verify` passes
- [ ] New or changed behaviour has tests
- [ ] Tests still pass **without** the `inference` extra and **without** a GPU
- [ ] Public functions, classes and modules have docstrings
- [ ] Documentation updated where behaviour changed

## Data, weights and secrets

- [ ] No dataset files, video, frames or annotations are committed
- [ ] No model weights (`*.pt`, `*.pth`, `*.onnx`, ...) are committed
- [ ] No credentials, tokens or `.env` file are committed
- [ ] No files added under `artifacts/`
- [ ] Notebooks committed with outputs cleared
- [ ] `git diff --stat` reviewed for anything unexpectedly large

## Honesty

<!-- The single most important section. -->

- [ ] No fabricated results, metrics, example detections or placeholder numbers
- [ ] Anything unimplemented is described as unimplemented, in code and in docs
- [ ] Results quoted here come from a recorded run, with the model variant and
      confidence threshold stated
- [ ] No fine-tuning or training code was added (out of scope this phase)

## If this changes dependencies

- [ ] `pyproject.toml` updated with a comment explaining *why*
- [ ] `make lock` run and both `uv.lock` and `requirements-colab.txt` committed
- [ ] `import PytorchWildlife` verified if the `inference` extra changed
      (resolution succeeding is **not** the same as import succeeding)

## If this ran inference

- Model / variant: <!-- e.g. MegaDetectorV6 / MDV6-yolov9-c -->
- Confidence threshold:
- Clips (manifest ids):
- Device and hardware:
- Run metadata: <!-- path under artifacts/metadata/ -->
- [ ] `experiments/experiment_log.md` entry added, including failures

## Notes for the reviewer

<!-- Anything you are unsure about, deliberately deferred, or want scrutinised. -->
