"""Command-line interface for the Phase 1 scaffold.

The setup-oriented commands, none of which touch a model:

``doctor``
    Report the environment a run would execute in.
``validate-config``
    Load and validate a YAML configuration file.
``show-paths``
    Print the resolved repository layout.

``fetch-clips`` selects and downloads a PanAf500 sample; ``detect`` runs
MegaDetector over it and writes annotated video; ``evaluate`` recomputes accuracy
from saved detections, and ``track`` re-tracks them and measures identity
quality. Both read ``artifacts/detections/``, so neither needs a GPU.

This module must import cleanly without the ``inference`` extra installed; every
machine-learning import happens lazily inside :func:`_probe_optional_dependency`.
"""

from __future__ import annotations

import importlib.util
import logging
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from panaf_ape_detection import __version__
from panaf_ape_detection.config import Config, ConfigError, load_config
from panaf_ape_detection.evaluation.detection import MatchCounts
from panaf_ape_detection.evaluation.tracking import ClipTrackEvaluation
from panaf_ape_detection.manifest import ManifestRow
from panaf_ape_detection.paths import RepositoryPaths, repository_root

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Iterator

    from panaf_ape_detection.inference.base import Detector
    from panaf_ape_detection.pipeline.runner import ClipResult

__all__ = ["app", "main"]

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="panaf-phase1",
    help=("Phase 1 tooling for pretrained MegaDetector V6 inference over PanAf500 clips."),
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
_error_console = Console(stderr=True)

_OK = "[green]yes[/green]"
_NO = "[yellow]no[/yellow]"

_INFERENCE_MODULES: tuple[tuple[str, str], ...] = (
    ("PytorchWildlife", "PyTorch-Wildlife"),
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("cv2", "OpenCV (headless)"),
    ("supervision", "supervision"),
    ("numpy", "NumPy"),
    ("pandas", "pandas"),
)


@dataclass(frozen=True, slots=True)
class _Probe:
    """Result of checking whether an optional dependency is importable.

    Attributes:
        installed: Whether the module was found on the import path.
        version: Reported version string, when the module exposes one.
    """

    installed: bool
    version: str | None = None


def _probe_optional_dependency(module_name: str) -> _Probe:
    """Check for *module_name* without importing heavyweight packages eagerly.

    The module is only actually imported when a version string is worth having,
    and any import failure is treated as "not installed" rather than propagated.

    Args:
        module_name: Importable module name, e.g. ``"torch"``.

    Returns:
        A :class:`_Probe` describing availability.
    """
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):  # pragma: no cover - broken install only
        return _Probe(installed=False)
    if spec is None:
        return _Probe(installed=False)

    try:
        from importlib.metadata import PackageNotFoundError, version

        distribution = {"cv2": "opencv-python-headless", "PytorchWildlife": "pytorchwildlife"}.get(
            module_name, module_name
        )
        try:
            return _Probe(installed=True, version=version(distribution))
        except PackageNotFoundError:
            return _Probe(installed=True)
    except Exception:  # pragma: no cover - defensive
        logger.debug("version lookup failed for %s", module_name, exc_info=True)
        return _Probe(installed=True)


def _torch_accelerators() -> tuple[str, str]:
    """Return ``(cuda_status, mps_status)`` markup, importing torch only if present.

    Returns:
        Rich-markup strings suitable for direct display. When PyTorch is absent
        both entries report that the check was skipped.
    """
    if not _probe_optional_dependency("torch").installed:
        return ("[dim]n/a (torch not installed)[/dim]", "[dim]n/a (torch not installed)[/dim]")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - broken torch install
        message = f"[red]import failed: {exc}[/red]"
        return (message, message)

    cuda = _OK if torch.cuda.is_available() else _NO
    try:
        mps = _OK if torch.backends.mps.is_available() else _NO
    except AttributeError:  # pragma: no cover - very old torch
        mps = _NO
    return (cuda, mps)


def _version_callback(value: bool) -> None:
    """Print the package version and exit."""
    if value:
        console.print(f"panaf-phase1 {__version__}")
        raise typer.Exit


@app.callback()
def _main_callback(
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the package version and exit.",
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug-level logging.")
    ] = False,
) -> None:
    """Configure logging shared by every subcommand."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


@app.command()
def doctor() -> None:
    """Report the environment that a pipeline run would execute in.

    Works without the ``inference`` extra: missing optional packages are
    reported, never raised.
    """
    paths = RepositoryPaths.from_root(repository_root())

    environment = Table(title="Environment", show_header=True, header_style="bold")
    environment.add_column("Check")
    environment.add_column("Value", overflow="fold")

    environment.add_row("panaf-phase1 version", __version__)
    environment.add_row("Python", sys.version.split()[0])
    environment.add_row("Python executable", sys.executable)
    environment.add_row("Operating system", f"{platform.system()} {platform.release()}")
    environment.add_row("Machine", platform.machine())
    environment.add_row("Repository root", str(paths.root))

    ffmpeg = shutil.which("ffmpeg")
    environment.add_row("FFmpeg on PATH", f"{_OK} ({ffmpeg})" if ffmpeg else _NO)

    cuda_status, mps_status = _torch_accelerators()
    environment.add_row("CUDA available", cuda_status)
    environment.add_row("Apple MPS available", mps_status)
    console.print(environment)

    # Inference is refused on CPU (runtime.require_accelerator), so say so here
    # rather than letting the first `detect` be where that is discovered.
    if _OK not in (cuda_status, mps_status):
        console.print(
            "[yellow]No accelerator here.[/yellow] Inference will be refused: this project "
            "never runs models on CPU.\nRun on a Colab GPU instead — "
            "[bold]notebooks/phase1_colab.ipynb[/bold] (Runtime → Change runtime type → T4 GPU)."
        )

    optional = Table(title="Optional inference stack", show_header=True, header_style="bold")
    optional.add_column("Package")
    optional.add_column("Installed")
    optional.add_column("Version")

    probes = {name: _probe_optional_dependency(name) for name, _ in _INFERENCE_MODULES}
    for module_name, label in _INFERENCE_MODULES:
        probe = probes[module_name]
        optional.add_row(label, _OK if probe.installed else _NO, probe.version or "-")
    console.print(optional)

    extra_ready = all(probe.installed for probe in probes.values())
    if extra_ready:
        console.print("[green]inference extra: present[/green]")
    else:
        console.print(
            "[yellow]inference extra: not fully installed[/yellow] "
            "(run [bold]uv sync --extra inference[/bold] when you are ready to run detection)"
        )

    directories = Table(title="Expected directories", show_header=True, header_style="bold")
    directories.add_column("Name")
    directories.add_column("Path", overflow="fold")
    directories.add_column("Exists")
    for name, path in paths.as_mapping().items():
        directories.add_row(name, str(path), _OK if path.is_dir() else _NO)
    console.print(directories)

    missing = paths.missing_directories()
    if missing:
        console.print(
            "[yellow]Missing directories:[/yellow] "
            + ", ".join(str(path.relative_to(paths.root)) for path in missing)
        )
    if not ffmpeg:
        console.print(
            "[yellow]FFmpeg was not found on PATH.[/yellow] Video decoding and export will "
            "need it; install it with your system package manager (macOS: "
            "[bold]brew install ffmpeg[/bold])."
        )

    console.print(
        "\n[dim]Note: this command only inspects the environment; it loads no model "
        "and reads no data.[/dim]"
    )


@app.command("validate-config")
def validate_config(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to a YAML config file, relative to the repository root.",
        ),
    ] = Path("configs/base.yaml"),
    show: Annotated[
        bool, typer.Option("--show/--no-show", help="Print the resolved configuration.")
    ] = True,
) -> None:
    """Load a YAML configuration file and report validation problems.

    Exits with status 1 and an actionable message when the file is missing,
    malformed, contains unknown keys, or violates a field constraint.
    """
    try:
        loaded = load_config(config)
    except ConfigError as exc:
        _error_console.print(f"[red]Configuration invalid[/red]\n{exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]OK[/green] {config} is valid.")

    if not loaded.model.variant_is_recognised:
        console.print(
            f"[yellow]Note:[/yellow] model.variant {loaded.model.variant!r} is not in the "
            "set documented in docs/obsidian/05 Technical/model.md. Confirm your installed "
            "PyTorch-Wildlife version supports it before relying on it."
        )

    if show:
        table = Table(title="Resolved configuration", show_header=True, header_style="bold")
        table.add_column("Key")
        table.add_column("Value", overflow="fold")
        for key, value in loaded.describe().items():
            table.add_row(key, value)
        console.print(table)

    manifest = loaded.data.manifest_path
    if not manifest.is_file():
        console.print(
            f"[yellow]Manifest not present yet:[/yellow] {manifest}\n"
            "  Create it from data/sample_manifest.example.csv once you have selected clips "
            "(see data/README.md)."
        )


@app.command("show-paths")
def show_paths(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help=(
                "Resolve paths through this config, so the output reflects what a run would "
                "actually use. Omit to show the plain checkout layout."
            ),
        ),
    ] = None,
) -> None:
    """Print the resolved repository layout and expected artifact directories.

    With ``--config`` the ``paths`` section (and any ``PANAF_*`` override) is
    applied, which is what a pipeline run will see. Without it, the layout of a
    fresh checkout is shown.
    """
    if config is None:
        paths = RepositoryPaths.from_root(repository_root())
        console.print(
            "[dim]Showing the default checkout layout. "
            "Pass --config to see the paths a run would use.[/dim]"
        )
    else:
        try:
            loaded = load_config(config)
        except ConfigError as exc:
            _error_console.print(f"[red]Configuration invalid[/red]\n{exc}")
            raise typer.Exit(code=1) from exc
        paths = loaded.repository_paths()
        console.print(f"[dim]Paths resolved through {config}.[/dim]")

    layout = Table(title="Repository layout", show_header=True, header_style="bold")
    layout.add_column("Name")
    layout.add_column("Path", overflow="fold")
    layout.add_column("Exists")
    for name, path in paths.as_mapping().items():
        layout.add_row(name, str(path), _OK if path.is_dir() else _NO)
    console.print(layout)

    artifacts = Table(
        title="Artifact subdirectories (git-ignored, created on demand)",
        show_header=True,
        header_style="bold",
    )
    artifacts.add_column("Name")
    artifacts.add_column("Path", overflow="fold")
    artifacts.add_column("Exists")
    for name, path in paths.artifact_subdirectories().items():
        artifacts.add_row(name, str(path), _OK if path.is_dir() else _NO)
    console.print(artifacts)


_CONFIG_OPTION = typer.Option(
    "--config", "-c", help="YAML config, relative to the repository root."
)


def _load_or_exit(config: Path) -> Config:
    """Load a config, exiting with an actionable message on failure."""
    try:
        return load_config(config)
    except ConfigError as exc:
        _error_console.print(f"[red]Configuration invalid[/red]\n{exc}")
        raise typer.Exit(code=1) from exc


def _build_detector(config: Config) -> Detector:
    """Construct the configured detector, or exit with guidance.

    Imported here rather than at module scope so the rest of the CLI keeps
    working without the ``inference`` extra.
    """
    try:
        from panaf_ape_detection.inference.megadetector import MegaDetectorV6Runner
    except ImportError as exc:
        _error_console.print(
            "[red]The inference extra is not installed.[/red]\n"
            "Run [bold]uv sync --extra inference[/bold] first."
        )
        raise typer.Exit(code=1) from exc

    with console.status(f"Loading {config.model.variant} (first run downloads weights)..."):
        try:
            return MegaDetectorV6Runner(
                config.model.variant,
                device=config.model.device,
                confidence_threshold=config.model.confidence_threshold,
            )
        except Exception as exc:
            _error_console.print(f"[red]Could not load the detector:[/red] {exc}")
            raise typer.Exit(code=1) from exc


def _report(results: list[ClipResult]) -> None:
    """Print the per-clip and overall accuracy tables."""
    from panaf_ape_detection.pipeline.runner import summarise

    table = Table(title="Per clip", show_header=True, header_style="bold")
    for column in ("Clip", "Frames", "Kept", "P", "R", "F1", "mIoU", "FP on empty"):
        table.add_column(column)
    for result in results:
        evaluation = result.evaluation
        if evaluation is None:
            table.add_row(result.clip_id, str(result.frames_processed), *(["-"] * 6))
            continue
        counts = evaluation.overall
        table.add_row(
            result.clip_id,
            str(result.frames_processed),
            str(result.detections_kept),
            f"{counts.precision:.3f}",
            f"{counts.recall:.3f}",
            f"{counts.f1:.3f}",
            f"{evaluation.mean_iou:.3f}",
            str(evaluation.false_positives_on_empty_frames),
        )
    console.print(table)

    summary = summarise(results)
    overall = summary["overall"]
    assert isinstance(overall, dict)
    console.print(
        f"\n[bold]Overall[/bold]  precision {overall['precision']} · "
        f"recall {overall['recall']} · F1 {overall['f1']} · "
        f"mean IoU {summary['mean_iou']}"
    )
    console.print(
        "[dim]MegaDetector emits a generic 'animal' class, so a detection counts as correct when "
        "it localises an annotated ape. No claim about species is made.[/dim]"
    )

    tracked = [result for result in results if result.track_evaluation is not None]
    if tracked:
        console.print()
        console.print(_track_table(result.track_evaluation for result in tracked))
        console.print(_TRACK_LEGEND)


def _track_table(evaluations: Iterable[ClipTrackEvaluation | None]) -> Table:
    """Build the per-clip track-quality table shared by ``detect`` and ``track``."""
    table = Table(title="Track quality", show_header=True, header_style="bold")
    for column in ("Clip", "Frames", "Apes", "Tracks", "ID sw.", "Frag.", "Coverage", "MT", "ML"):
        table.add_column(column)
    for evaluation in evaluations:
        if evaluation is None:  # pragma: no cover - callers filter these out
            continue
        table.add_row(
            evaluation.clip_id,
            str(evaluation.frames_evaluated),
            str(len(evaluation.individuals)),
            str(evaluation.predicted_tracks),
            str(evaluation.total_id_switches),
            f"{evaluation.mean_fragmentation:.2f}",
            f"{evaluation.mean_coverage:.3f}",
            str(evaluation.mostly_tracked),
            str(evaluation.mostly_lost),
        )
    return table


_TRACK_LEGEND = (
    "[dim]Frag. is predicted tracks per annotated ape (1.00 is ideal, 0 means never tracked). "
    "MT/ML count individuals covered in >=80% / <=20% of their frames. Fragmentation is bounded "
    "by detection recall: a box that was never detected cannot be associated.[/dim]"
)


@app.command("fetch-clips")
def fetch_clips(
    count: Annotated[int, typer.Option("--count", help="Clips to download (5-10).")] = 10,
    pool: Annotated[int, typer.Option("--pool", help="Candidates to profile first.")] = 150,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Select only; download nothing.")
    ] = False,
) -> None:
    """Select and download a purposive PanAf500 sample, writing the manifest.

    Downloads dataset files under the deposit's non-commercial licence. Nothing
    it writes may be committed.
    """
    import subprocess

    script = repository_root() / "scripts" / "fetch_panaf500.py"
    command = [sys.executable, str(script), "--count", str(count), "--pool", str(pool)]
    if dry_run:
        command.append("--dry-run")
    raise typer.Exit(code=subprocess.run(command, check=False).returncode)


@app.command()
def detect(
    config: Annotated[Path, _CONFIG_OPTION] = Path("configs/base.yaml"),
    clips: Annotated[int | None, typer.Option("--clips", help="Cap clips processed.")] = None,
    frames: Annotated[int | None, typer.Option("--frames", help="Cap frames per clip.")] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Re-run existing outputs.")
    ] = False,
) -> None:
    """Run MegaDetector over the manifest's clips and write annotated video.

    Produces per-frame detections, an annotated MP4 per clip, accuracy against
    ground truth, and a run-metadata record.
    """
    from panaf_ape_detection.pipeline.runner import run_manifest
    from panaf_ape_detection.runtime import set_seeds

    loaded = _load_or_exit(config)
    set_seeds(loaded.project.seed)
    detector = _build_detector(loaded)

    verified = getattr(detector, "verified_device", None)
    console.print(
        f"[green]{detector.name} {detector.variant}[/green] on "
        f"[bold]{verified or detector.device.value}[/bold]"
    )

    try:
        results = run_manifest(
            loaded,
            detector,
            max_clips=clips,
            overwrite=overwrite,
            limit_frames=frames,
        )
    except (FileNotFoundError, ValueError) as exc:
        _error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    _report(results)
    artifacts = loaded.repository_paths().artifacts_dir
    console.print(f"\nAnnotated video: {artifacts / 'videos'}")
    console.print(f"Detections:      {artifacts / 'detections'}")
    console.print(f"Metrics:         {artifacts / 'metrics'}")


def _saved_detection_documents(
    loaded: Config,
) -> Iterator[tuple[ManifestRow, str, dict[str, Any]]]:
    """Yield ``(row, annotation filename, document)`` per clip with saved detections.

    Rows with no stored detections, or no annotation file to compare against,
    are skipped -- both are ordinary when only part of the manifest has been
    run locally.

    Raises:
        typer.Exit: If the manifest cannot be read.
    """
    import json as _json

    from panaf_ape_detection.pipeline.runner import load_manifest

    paths = loaded.repository_paths()
    try:
        rows = load_manifest(loaded.data.manifest_path)
    except FileNotFoundError as exc:
        _error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    for row in rows:
        stored = paths.artifacts_dir / "detections" / f"{row.clip_id}.json"
        annotation = row.annotation_filename
        if not stored.is_file() or not annotation:
            continue
        yield row, annotation, _json.loads(stored.read_text(encoding="utf-8"))


@app.command()
def evaluate(
    config: Annotated[Path, _CONFIG_OPTION] = Path("configs/base.yaml"),
    confidence: Annotated[
        float | None,
        typer.Option(
            "--confidence",
            min=0.0,
            max=1.0,
            help="Re-score at this threshold instead of the one used at detection time.",
        ),
    ] = None,
) -> None:
    """Recompute accuracy from saved detections, without re-running inference.

    Reads ``artifacts/detections/`` and compares against ground truth, so a
    threshold can be reconsidered without spending GPU time again.

    ``--confidence`` can only be raised, never lowered: the saved detections
    were already filtered at detection time, so boxes below that threshold were
    never written and cannot be recovered here. Lowering it requires re-running
    ``detect`` with a lower ``model.confidence_threshold``.
    """
    from panaf_ape_detection.data.annotations import load_ground_truth
    from panaf_ape_detection.evaluation.detection import evaluate_clip
    from panaf_ape_detection.inference.filtering import filter_by_confidence
    from panaf_ape_detection.types import Detection as _Detection

    loaded = _load_or_exit(config)
    raw_root = loaded.repository_paths().raw_data_dir / "panaf500"

    title = "Accuracy from saved detections"
    if confidence is not None:
        title += f" (re-scored at {confidence:.2f})"
    table = Table(title=title, show_header=True, header_style="bold")
    for column in ("Clip", "Kept", "P", "R", "F1", "mIoU"):
        table.add_column(column)

    found = 0
    totals = [0, 0, 0]  # true positives, false positives, false negatives
    for row, annotation, document in _saved_detection_documents(loaded):
        stored_threshold = float(document["model"]["confidence_threshold"])
        threshold = stored_threshold if confidence is None else confidence
        if threshold < stored_threshold:
            _error_console.print(
                f"[red]--confidence {threshold:.2f} is below the {stored_threshold:.2f} "
                f"used when {row.clip_id} was detected.[/red]\n"
                "Boxes under that score were never saved, so the result would look like a "
                "genuine measurement while silently missing detections. Re-run "
                "[bold]detect[/bold] with a lower model.confidence_threshold instead."
            )
            raise typer.Exit(code=1)

        # `evaluate_clip` records the threshold but does not apply it, so the
        # filtering has to happen here for --confidence to mean anything.
        predictions = {
            frame["frame_index"]: filter_by_confidence(
                [_Detection(**d) for d in frame["detections"]], threshold
            )
            for frame in document["frames"]
        }
        truth = load_ground_truth(
            raw_root / "annotations" / annotation,
            frame_width=document["video"]["width"],
            frame_height=document["video"]["height"],
            clip_id=row.clip_id,
        )
        evaluation = evaluate_clip(row.clip_id, predictions, truth, confidence_threshold=threshold)
        counts = evaluation.overall
        totals[0] += counts.true_positives
        totals[1] += counts.false_positives
        totals[2] += counts.false_negatives
        table.add_row(
            row.clip_id,
            str(counts.true_positives + counts.false_positives),
            f"{counts.precision:.3f}",
            f"{counts.recall:.3f}",
            f"{counts.f1:.3f}",
            f"{evaluation.mean_iou:.3f}",
        )
        found += 1

    if not found:
        _error_console.print(
            "[yellow]No saved detections found.[/yellow] "
            "Run [bold]panaf-phase1 detect[/bold] first."
        )
        raise typer.Exit(code=1)
    console.print(table)

    pooled = MatchCounts(
        true_positives=totals[0], false_positives=totals[1], false_negatives=totals[2]
    )
    console.print(
        f"\n[bold]Pooled over {found} clip(s)[/bold]  precision {pooled.precision:.4f} · "
        f"recall {pooled.recall:.4f} · F1 {pooled.f1:.4f}"
    )


@app.command()
def track(
    config: Annotated[Path, _CONFIG_OPTION] = Path("configs/base.yaml"),
    minimum_track_length: Annotated[
        int | None,
        typer.Option("--min-track-length", min=1, help="Override tracking.minimum_track_length."),
    ] = None,
) -> None:
    """Track saved detections and measure identity quality against ``ape_id``.

    Runs the configured tracker over ``artifacts/detections/`` rather than over
    video, so tracker settings can be explored on a laptop in seconds without
    re-running the detector. Track ids already stored in those files are
    discarded and recomputed, so the numbers always match the settings printed.

    Reports ID switches, fragmentation and coverage per clip, and writes
    ``artifacts/metrics/<clip>_tracking.json``.
    """
    import json as _json

    from panaf_ape_detection.data.annotations import load_ground_truth
    from panaf_ape_detection.evaluation.tracking import evaluate_tracking
    from panaf_ape_detection.pipeline.runner import build_tracker
    from panaf_ape_detection.tracking.bytetrack import drop_short_tracks
    from panaf_ape_detection.types import Detection as _Detection

    loaded = _load_or_exit(config)
    if not loaded.tracking.enabled:
        _error_console.print(
            "[red]tracking.enabled is false in this config.[/red] Enable it to run tracking."
        )
        raise typer.Exit(code=1)

    paths = loaded.repository_paths()
    raw_root = paths.raw_data_dir / "panaf500"
    minimum = minimum_track_length or loaded.tracking.minimum_track_length

    evaluations: list[ClipTrackEvaluation] = []
    for row, annotation, document in _saved_detection_documents(loaded):
        video = document["video"]
        try:
            # A fresh tracker per clip, sized and timed from that clip.
            tracker = build_tracker(
                loaded,
                frame_rate=float(video["fps"]),
                frame_width=int(video["width"]),
                frame_height=int(video["height"]),
            )
        except ValueError as exc:
            _error_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        except ImportError as exc:
            _error_console.print(
                "[red]The inference extra is not installed.[/red]\n"
                "Tracking needs [bold]supervision[/bold]; run "
                "[bold]uv sync --extra inference[/bold]."
            )
            raise typer.Exit(code=1) from exc
        assert tracker is not None  # tracking.enabled was checked above

        # Frame order matters: a tracker fed frames out of order produces
        # meaningless motion estimates, and JSON preserves file order only.
        frames = sorted(document["frames"], key=lambda frame: int(frame["frame_index"]))
        tracked: dict[int, list[Any]] = {}
        for frame in frames:
            # Rebuild as plain Detections so any stored track_id is ignored.
            detections = [
                _Detection(
                    box=d["box"],
                    confidence=d["confidence"],
                    category_id=d["category_id"],
                    category_name=d["category_name"],
                )
                for d in frame["detections"]
            ]
            tracked[int(frame["frame_index"])] = tracker.update(detections)

        tracked = dict(drop_short_tracks(tracked, minimum))
        truth = load_ground_truth(
            raw_root / "annotations" / annotation,
            frame_width=video["width"],
            frame_height=video["height"],
            clip_id=row.clip_id,
        )
        evaluation = evaluate_tracking(row.clip_id, tracked, truth)

        destination = paths.artifacts_dir / "metrics" / f"{row.clip_id}_tracking.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            _json.dumps(
                {**evaluation.as_dict(), "minimum_track_length": minimum}, indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )

        evaluations.append(evaluation)

    if not evaluations:
        _error_console.print(
            "[yellow]No saved detections found.[/yellow] "
            "Run [bold]panaf-phase1 detect[/bold] first."
        )
        raise typer.Exit(code=1)

    console.print(_track_table(evaluations))
    console.print(_TRACK_LEGEND)
    console.print(
        f"\n[dim]tracker {loaded.tracking.backend.value}, minimum track length {minimum} "
        f"frame(s), confidence {loaded.model.confidence_threshold:.2f}[/dim]"
    )
    console.print(f"Track metrics: {paths.artifacts_dir / 'metrics'}")


def main() -> None:
    """Entry point used by the ``panaf-phase1`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
