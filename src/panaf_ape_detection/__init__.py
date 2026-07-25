"""panaf_ape_detection: Phase 1 ("See") great-ape detection scaffold.

This package deliberately keeps its top-level namespace free of heavyweight
machine-learning imports. Importing :mod:`panaf_ape_detection` must stay cheap
and must succeed in an environment installed *without* the ``inference`` extra,
so that the CLI, configuration loading and the test suite remain usable by any
contributor who has only run ``uv sync``.

Inference, tracking and visualisation modules will be added in later phases and
will import their heavy dependencies lazily, inside functions.
"""

from __future__ import annotations

from panaf_ape_detection.config import Config, load_config
from panaf_ape_detection.manifest import MANIFEST_COLUMNS, ManifestRow
from panaf_ape_detection.paths import RepositoryPaths, repository_root
from panaf_ape_detection.provenance import build_run_metadata, file_sha256
from panaf_ape_detection.runtime import module_device, resolve_device, set_seeds

__all__ = [
    "MANIFEST_COLUMNS",
    "Config",
    "ManifestRow",
    "RepositoryPaths",
    "__version__",
    "build_run_metadata",
    "file_sha256",
    "load_config",
    "module_device",
    "repository_root",
    "resolve_device",
    "set_seeds",
]

__version__ = "0.1.0"
