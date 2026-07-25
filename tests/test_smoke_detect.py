"""Opt-in test that loads real MegaDetector weights.

**Deselected by default.** `pyproject.toml` sets `addopts = [..., "-m", "not
weights"]`, so `make test` and CI never reach this. Run it deliberately:

    uv run pytest -m weights
    make smoke-detect          # the same path, as a script

It needs the `inference` extra, network access on first run (roughly a gigabyte
of weights, cached afterwards) and a few minutes. It is the gate to clear before
touching real PanAf clips: if it passes, a Phase 1c failure is in the new
pipeline code rather than the stack underneath it.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import pytest

from panaf_ape_detection.paths import repository_root

pytestmark = pytest.mark.weights


def _load_smoke_detect() -> ModuleType:
    """Import `scripts/smoke_detect.py` as a module."""
    path = repository_root() / "scripts" / "smoke_detect.py"
    spec = importlib.util.spec_from_file_location("_smoke_detect_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_inference_extra_is_installed():
    """Fail with a useful message rather than an ImportError deep in the run."""
    if importlib.util.find_spec("PytorchWildlife") is None:
        pytest.fail(
            "the `inference` extra is not installed. Run `uv sync --extra inference`, "
            "or use `uv run --extra inference pytest -m weights`."
        )


def test_real_weights_load_and_inference_runs():
    """End-to-end: config -> device -> weights -> inference -> run metadata.

    Asserts only that the chain completes and the device was verified. It makes
    **no claim about detection quality** -- the frame is synthetic noise, where
    any detection would be a false positive.
    """
    smoke_detect = _load_smoke_detect()

    assert smoke_detect.main([]) == 0
