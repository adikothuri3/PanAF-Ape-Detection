"""Execute the Colab notebook's analysis cells against a synthetic artifacts tree.

**This is the test that was missing.** Three notebook defects reached the user in
a row, the last of them a ``KeyError`` from reading ``artifacts/`` with the wrong
assumption about its layout. Static checks cannot catch that class: it needs real
files with real shapes.

So this builds a miniature ``artifacts/`` directory -- detection metrics, track
metrics, saved detections, run metadata, in the layout a real run produces -- and
runs every notebook cell that only reads from disk, in order, in one shared
namespace, exactly as *Run all* would.

Cells needing Colab, a GPU, the network or ffmpeg are skipped by marker; what
remains is precisely the code that broke.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from panaf_ape_detection.paths import repository_root
from panaf_ape_detection.reporting import TRACK_METRICS_SUBDIR

pytest.importorskip("pandas", reason="the notebook's tables use pandas")

# Markers for cells that cannot run here: they install packages, clone the repo,
# talk to Colab, need a GPU, or shell out to a real binary.
_UNRUNNABLE = (
    "REPO_URL",
    "pip install",
    "google.colab",
    "nvidia-smi",
    "tomllib",
    "smoke_inference",
    "fetch_panaf500",
    # Any CLI invocation: needs weights, a GPU and the dataset. Matched on the
    # module path, which no formatter will reflow -- an earlier version matched
    # on `cli", "detect"` and silently stopped matching when ruff reformatted
    # the cell, so the test tried to run real inference.
    "panaf_ape_detection.cli",
    "importlib.invalidate_caches",
    "import torch",
    "ffmpeg",
)


def _counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    total = tp + fp
    annotated = tp + fn
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / total if total else 0.0,
        "recall": tp / annotated if annotated else 0.0,
        "f1": (2 * tp / (total + annotated)) if (total + annotated) else 0.0,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def artifacts_tree(tmp_path: Path) -> Path:
    """A realistic run: two clips, tracking on, detections and metadata saved."""
    root = tmp_path / "artifacts"
    for clip in ("clip-a", "clip-b"):
        _write(
            root / "metrics" / f"{clip}.json",
            {
                "clip_id": clip,
                "iou_threshold": 0.5,
                "confidence_threshold": 0.2,
                "frames_evaluated": 360,
                "empty_frames": 3,
                "false_positives_on_empty_frames": 1,
                "mean_iou": 0.83,
                "overall": _counts(100, 20, 60),
                "by_behaviour": {"walking": _counts(60, 0, 30), "hanging": _counts(2, 0, 30)},
                "by_size": {"small": _counts(20, 0, 40), "large": _counts(80, 0, 20)},
            },
        )
        _write(
            root / "metrics" / TRACK_METRICS_SUBDIR / f"{clip}.json",
            {
                "clip_id": clip,
                "iou_threshold": 0.5,
                "frames_evaluated": 360,
                "annotated_individuals": 3,
                "predicted_tracks": 4,
                "total_id_switches": 2,
                "mean_fragmentation": 1.33,
                "mean_coverage": 0.42,
                "mostly_tracked": 1,
                "mostly_lost": 1,
                "individuals": [],
            },
        )
        _write(
            root / "detections" / f"{clip}.json",
            {
                "clip_id": clip,
                "model": {"confidence_threshold": 0.05},
                "video": {"width": 720, "height": 404, "fps": 24.0, "frame_count": 2},
                "frames": [
                    {"frame_index": 0, "detections": [{"confidence": c} for c in (0.06, 0.2, 0.9)]},
                    {"frame_index": 1, "detections": [{"confidence": 0.12}]},
                ],
            },
        )
    _write(
        root / "metadata" / "phase1-colab_1.json",
        {
            "experiment_name": "phase1-colab",
            "git_commit": "abc123",
            "git_dirty": False,
            "device": "cuda",
            "model_variant": "MDV6-yolov9-c",
            "confidence_threshold": 0.2,
            "seed": 42,
            "elapsed_seconds": 12.5,
            "inputs": [{"path": "x", "sha256": "y"}],
        },
    )
    # The manifest cell reads this.
    manifest = tmp_path / "data" / "sample_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "clip_id,split,species,selected_reason\n"
        "clip-a,train,chimpanzee,hard behaviours: hanging\n"
        "clip-b,test,gorilla,small subjects\n",
        encoding="utf-8",
    )
    return tmp_path


def _analysis_cells() -> list[tuple[int, str]]:
    notebook = json.loads(
        (repository_root() / "notebooks" / "phase1_colab.ipynb").read_text(encoding="utf-8")
    )
    cells = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if any(marker in source for marker in _UNRUNNABLE):
            continue
        cells.append((index, source))
    return cells


def test_there_are_analysis_cells_to_check():
    """If the markers ever exclude everything, the test below would pass vacuously."""
    assert len(_analysis_cells()) >= 4


def test_notebook_analysis_cells_run_against_a_real_artifacts_tree(
    artifacts_tree: Path, monkeypatch: pytest.MonkeyPatch
):
    """Run all, minus the parts that need Colab. Any raise here is a shipped bug.

    The regression this pins: `artifacts/metrics/` holds detection metrics *and*
    a `tracking/` subdirectory, so a cell that reads the directory naively and
    expects one schema raises `KeyError: 'overall'`.
    """
    monkeypatch.chdir(artifacts_tree)
    shown: list[object] = []

    def display(*objects: object, **_kwargs: object) -> None:
        """Stand in for IPython's display, recording what the cells rendered."""
        shown.extend(objects)

    namespace: dict[str, Any] = {"__name__": "__main__", "display": display}

    for index, source in _analysis_cells():
        stripped = "\n".join(
            "pass" if line.lstrip().startswith(("!", "%")) else line for line in source.splitlines()
        )
        try:
            exec(compile(stripped, f"<notebook cell {index}>", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - the failure message is the point
            pytest.fail(f"notebook cell {index} raised {type(exc).__name__}: {exc}")

    # The cells really did compute something, rather than silently no-opping.
    assert len(namespace["metrics"]) == 2
    assert namespace["pooled"].true_positives == 200
    assert shown, "no cell rendered a table; the analysis path did not run"
