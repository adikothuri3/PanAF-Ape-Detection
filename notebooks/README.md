# Notebooks

## Contents

- [`phase1_colab.ipynb`](phase1_colab.ipynb) — Google Colab scaffold for the Phase 1 pipeline.
  Environment setup, verification and configuration loading work; the pipeline sections are
  explicit placeholders that raise `NotImplementedError`.

## The rule: no business logic in notebooks

Reusable logic belongs in `src/panaf_ape_detection/`, with types and a test. The notebook
orchestrates and displays; it does not define.

This is not style preference. Code that exists only in a notebook cannot be imported, cannot be
unit-tested, cannot be type-checked, does not appear meaningfully in a diff, and cannot be reused
from the CLI. Every one of those matters for work that has to be reproducible.

A practical rule: **if a cell has grown past a few lines of glue, it is a module you have not
written yet.** Move it to `src/`, add a test, and import it back.

## Committing notebooks

**Clear all outputs before committing** — *Edit → Clear all outputs* in Colab, or
*Kernel → Restart & Clear Output* in Jupyter.

`scripts/verify_repository.py` fails CI if a committed notebook contains stored output or execution
counts. Three reasons:

1. **Licensing.** Output cells can embed dataset frames as base64 images. That is redistribution of
   PanAf footage through this repository, which the dataset licence does not permit.
2. **Honesty.** Committed output makes a notebook look like it ran and produced results. In a
   scaffold with unimplemented stages, that is actively misleading.
3. **Diff noise.** Embedded images make notebook diffs unreviewable and inflate repository size.

## Using the Colab notebook

1. Open it in Colab.
2. **Runtime → Change runtime type → T4 GPU**, *before* installing anything — changing the runtime
   restarts the session and discards installs.
3. Set `REPO_URL` in the clone cell.
4. Run sections 1–3 (clone, install, verify). Colab may ask for a session restart after the install;
   accept it, then resume at section 3.
5. Section 4 (Google Drive) is optional and separable — skip it if you are not persisting anything.
6. Section 5 creates data folders. **It downloads nothing**; obtain PanAf500 yourself under its own
   licence, per [`../data/README.md`](../data/README.md).
7. Section 6 demonstrates configuration loading, and works today.
8. Sections 7–12 are placeholders and will raise `NotImplementedError` until the corresponding
   modules exist.

The notebook installs from `requirements-colab.txt`, which is generated from `uv.lock` rather than
maintained by hand — that is what keeps the Colab environment aligned with the locked local one
instead of drifting with whatever Colab preinstalls.

## Adding a notebook

Reasonable additions: exploratory data inspection, visual review of results, figure generation for
the write-up.

Not reasonable: a second implementation of a pipeline stage. If a notebook and `src/` disagree about
how detection works, the results are untrustworthy and it will not be obvious which one produced any
given figure.

Name new notebooks `<phase>_<purpose>.ipynb` and add a line to this README.
