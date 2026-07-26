"""Orchestration.

The only layer that knows the order of stages. Stage modules stay unaware of
each other so that any one of them can be replaced without the rest noticing.
"""

from __future__ import annotations

from panaf_ape_detection.pipeline.runner import ClipResult, load_manifest, run_clip, run_manifest

__all__ = ["ClipResult", "load_manifest", "run_clip", "run_manifest"]
