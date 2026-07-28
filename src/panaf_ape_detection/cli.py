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
import time
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
from panaf_ape_detection.reporting import TRACK_METRICS_SUBDIR

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

_SWEEP_METRICS_SUBDIR = "tracking-sweep"
"""Where sweep records live, relative to ``artifacts/metrics/``."""

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
    for column in (
        "Clip",
        "Apes",
        "Tracks",
        "ID sw.",
        "Frag.",
        "Coverage",
        "ID cov.",
        "Purity",
        "Merges",
        "Jitter",
        "MT",
        "ML",
    ):
        table.add_column(column)
    for evaluation in evaluations:
        if evaluation is None:  # pragma: no cover - callers filter these out
            continue
        table.add_row(
            evaluation.clip_id,
            str(len(evaluation.individuals)),
            str(evaluation.predicted_tracks),
            str(evaluation.total_id_switches),
            f"{evaluation.mean_fragmentation:.2f}",
            f"{evaluation.mean_coverage:.3f}",
            f"{evaluation.mean_identity_coverage:.3f}",
            f"{evaluation.mean_track_purity:.3f}",
            str(evaluation.id_merges),
            f"{evaluation.mean_jitter:.4f}",
            str(evaluation.mostly_tracked),
            str(evaluation.mostly_lost),
        )
    return table


_TRACK_LEGEND = (
    "[dim]Frag. is predicted tracks per annotated ape (1.00 is ideal, 0 means never tracked). "
    "ID cov. is the share of an ape's frames held by its single best track -- unlike Coverage it "
    "falls when one ape is split across several tracks, so it is the number to tune against. "
    "Purity is the share of a track's frames belonging to one ape, and Merges counts tracks that "
    "covered more than one: together they catch the failure every other column here rewards, "
    "joining two apes into a single track. Jitter is normalised box shake; lower is smoother. "
    "MT/ML count individuals covered in >=80% / <=20% of their frames. Coverage is bounded by "
    "detection recall: a box that was never detected cannot be associated.[/dim]"
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
    loaded: Config, *, detections_dir: Path | None = None
) -> Iterator[tuple[ManifestRow, str, dict[str, Any]]]:
    """Yield ``(row, annotation filename, document)`` per clip with saved detections.

    Rows with no stored detections, or no annotation file to compare against,
    are skipped -- both are ordinary when only part of the manifest has been
    run locally.

    Args:
        loaded: The resolved configuration.
        detections_dir: Read from here instead of ``<artifacts>/detections``,
            so an experiment can run against another run's output without
            copying it or pointing the whole config elsewhere.

    Raises:
        typer.Exit: If the manifest cannot be read.
    """
    import json as _json

    from panaf_ape_detection.pipeline.runner import load_manifest

    paths = loaded.repository_paths()
    directory = detections_dir if detections_dir is not None else paths.artifacts_dir / "detections"
    try:
        rows = load_manifest(loaded.data.manifest_path)
    except FileNotFoundError as exc:
        _error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    for row in rows:
        stored = directory / f"{row.clip_id}.json"
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
    detections_dir: Annotated[
        Path | None,
        typer.Option("--detections-dir", help="Read detections from here instead of artifacts/."),
    ] = None,
    annotations_dir: Annotated[
        Path | None,
        typer.Option("--annotations-dir", help="Read ground truth from here instead of data/raw."),
    ] = None,
    per_clip: Annotated[
        bool,
        typer.Option("--per-clip/--pooled-only", help="Show a row per clip, not just the total."),
    ] = True,
) -> None:
    """Recompute accuracy from saved detections, without re-running inference.

    Reads ``artifacts/detections/`` and compares against ground truth, so a
    threshold can be reconsidered without spending GPU time again.

    ``--confidence`` can only be raised, never lowered: the saved detections
    were already filtered at detection time, so boxes below that threshold were
    never written and cannot be recovered here. Lowering it requires re-running
    ``detect`` with a lower ``model.confidence_threshold``.
    """
    from panaf_ape_detection.evaluation.detection import evaluate_clip
    from panaf_ape_detection.inference.filtering import filter_by_confidence
    from panaf_ape_detection.pipeline.retrack import load_clip
    from panaf_ape_detection.reporting import detection_fields
    from panaf_ape_detection.types import Detection as _Detection

    loaded = _load_or_exit(config)

    title = "Accuracy from saved detections"
    if confidence is not None:
        title += f" (re-scored at {confidence:.2f})"
    table = Table(title=title, show_header=True, header_style="bold")
    for column in ("Clip", "Kept", "P", "R", "F1", "mIoU"):
        table.add_column(column)

    found = 0
    totals = [0, 0, 0]  # true positives, false positives, false negatives
    for source in _clip_sources(loaded, detections_dir, annotations_dir):
        document, truth = load_clip(source)
        stored_threshold = float(document["model"]["confidence_threshold"])
        threshold = stored_threshold if confidence is None else confidence
        if threshold < stored_threshold:
            _error_console.print(
                f"[red]--confidence {threshold:.2f} is below the {stored_threshold:.2f} "
                f"used when {source.clip_id} was detected.[/red]\n"
                "Boxes under that score were never saved, so the result would look like a "
                "genuine measurement while silently missing detections. Re-run "
                "[bold]detect[/bold] with a lower model.confidence_threshold instead."
            )
            raise typer.Exit(code=1)

        # `evaluate_clip` records the threshold but does not apply it, so the
        # filtering has to happen here for --confidence to mean anything.
        #
        # `detection_fields` rather than `_Detection(**d)`: a document written
        # with tracking on carries track_id, which `Detection` forbids.
        predictions = {
            frame["frame_index"]: filter_by_confidence(
                [_Detection(**detection_fields(d)) for d in frame["detections"]], threshold
            )
            for frame in document["frames"]
        }
        evaluation = evaluate_clip(
            source.clip_id,
            predictions,
            truth,
            confidence_threshold=threshold,
            model_variant=str(document["model"].get("variant", "")),
        )
        counts = evaluation.overall
        totals[0] += counts.true_positives
        totals[1] += counts.false_positives
        totals[2] += counts.false_negatives
        if per_clip:
            table.add_row(
                source.clip_id,
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
    if per_clip:
        console.print(table)

    pooled = MatchCounts(
        true_positives=totals[0], false_positives=totals[1], false_negatives=totals[2]
    )
    console.print(
        f"\n[bold]Pooled over {found} clip(s)[/bold]  precision {pooled.precision:.4f} · "
        f"recall {pooled.recall:.4f} · F1 {pooled.f1:.4f}"
    )


def _clip_sources(
    loaded: Config, detections_dir: Path | None, annotations_dir: Path | None = None
) -> list[Any]:
    """Collect the clips a re-tracking command can work over.

    Two ways of deciding *which* clips, and the second exists because the first
    was a redundancy that blocked real work:

    * **From the manifest** (default). The manifest is the record of a curated
      sample, so it is the right source when re-tracking that sample.
    * **From the detections directory**, when ``--detections-dir`` is given and
      the manifest does not cover what is in it. ``--detections-dir`` already
      names the clips; requiring a manifest that lists the same ones again means
      a cache produced elsewhere -- another machine, a Colab session -- cannot be
      re-tracked without first reconstructing a manifest for it, checksums and
      all, for files that may not even be present.

    Args:
        loaded: The resolved configuration.
        detections_dir: Read detections from here instead of the configured
            artifacts directory.
        annotations_dir: Read ground truth from here instead of
            ``<raw_data_dir>/panaf500/annotations``.

    Returns:
        A list of :class:`~panaf_ape_detection.pipeline.retrack.ClipSource`,
        ordered by clip id, and only for clips whose annotation exists.
    """
    from panaf_ape_detection.pipeline.retrack import ClipSource

    paths = loaded.repository_paths()
    annotations = (
        Path(annotations_dir)
        if annotations_dir is not None
        else paths.raw_data_dir / "panaf500" / "annotations"
    )
    override = Path(detections_dir) if detections_dir is not None else None

    sources: list[Any] = []
    seen: set[str] = set()
    for row, annotation, _ in _saved_detection_documents(loaded, detections_dir=override):
        stored = (override or paths.artifacts_dir / "detections") / f"{row.clip_id}.json"
        seen.add(row.clip_id)
        sources.append(
            ClipSource(
                clip_id=row.clip_id,
                detections_path=stored,
                annotation_path=annotations / annotation,
            )
        )

    if override is not None:
        # Anything in the directory the manifest did not account for. Skipped
        # silently when no annotation exists, since a clip with no ground truth
        # cannot be scored either way.
        for stored in sorted(override.glob("*.json")):
            clip_id = stored.stem
            annotation_path = annotations / f"{clip_id}.json"
            if clip_id in seen or not annotation_path.is_file():
                continue
            sources.append(
                ClipSource(
                    clip_id=clip_id,
                    detections_path=stored,
                    annotation_path=annotation_path,
                )
            )

    return sorted(sources, key=lambda source: source.clip_id)


def _write_tracked_detections(
    destination: Path,
    *,
    clip_id: str,
    document: dict[str, Any],
    tracked: dict[int, Any],
    settings: Any,
) -> None:
    """Write the tracker's refined boxes back out as a detections document.

    The output of tracking is not the same set of boxes as its input: short
    tracks have been dropped, gaps interpolated, positions smoothed. Only by
    writing it out can that output be measured against ground truth, or consumed
    by a later stage.

    The ``model`` block is copied from the source document rather than
    reconstructed. These boxes originate from that detector run, and saying so is
    what lets the file be read on its own later.

    Args:
        destination: Where to write.
        clip_id: Manifest identifier.
        document: The source detections document, for provenance and geometry.
        tracked: ``{frame_index: tracked detections}``.
        settings: The :class:`RetrackSettings` used, recorded alongside.
    """
    from panaf_ape_detection.pipeline.runner import write_detections_document
    from panaf_ape_detection.types import TrackedFrameDetections

    video = document["video"]
    fps = float(video["fps"]) or 24.0
    records = [
        # TrackedFrameDetections, not FrameDetections: pydantic serialises to the
        # *declared* field type, so the wrong container here silently drops every
        # track_id -- the bug the type exists to prevent.
        TrackedFrameDetections(
            clip_id=clip_id,
            frame_index=index,
            frame_width=int(video["width"]),
            frame_height=int(video["height"]),
            timestamp_seconds=index / fps,
            detections=tracked[index],
        )
        for index in sorted(tracked)
    ]

    write_detections_document(
        destination,
        clip_id=clip_id,
        model=document.get("model", {}),
        tracking={"enabled": True, "backend": "bytetrack", **settings.as_dict()},
        video=video,
        records=records,
    )


def _exit_without_the_inference_extra() -> typer.Exit:
    """Explain a missing ``supervision`` rather than showing an ImportError."""
    _error_console.print(
        "[red]The inference extra is not installed.[/red]\n"
        "Tracking needs [bold]supervision[/bold]; run "
        "[bold]uv sync --extra inference[/bold]."
    )
    return typer.Exit(code=1)


@app.command()
def track(
    config: Annotated[Path, _CONFIG_OPTION] = Path("configs/base.yaml"),
    minimum_track_length: Annotated[
        int | None,
        typer.Option("--min-track-length", min=1, help="Override tracking.minimum_track_length."),
    ] = None,
    activation_threshold: Annotated[
        float | None,
        typer.Option(
            "--activation", min=0.0, max=1.0, help="Override tracking.activation_threshold."
        ),
    ] = None,
    lost_track_buffer: Annotated[
        int | None,
        typer.Option("--buffer", min=1, help="Override tracking.lost_track_buffer."),
    ] = None,
    minimum_matching_threshold: Annotated[
        float | None,
        typer.Option("--match", min=0.0, max=1.0, help="Override minimum_matching_threshold."),
    ] = None,
    minimum_consecutive_frames: Annotated[
        int | None,
        typer.Option("--min-consecutive", min=1, help="Override minimum_consecutive_frames."),
    ] = None,
    detection_floor: Annotated[
        float,
        typer.Option(
            "--detection-floor",
            min=0.0,
            max=1.0,
            help="Drop saved detections below this score before tracking.",
        ),
    ] = 0.0,
    detections_dir: Annotated[
        Path | None,
        typer.Option("--detections-dir", help="Read detections from here instead of artifacts/."),
    ] = None,
    annotations_dir: Annotated[
        Path | None,
        typer.Option("--annotations-dir", help="Read ground truth from here instead of data/raw."),
    ] = None,
    metrics_dir: Annotated[
        Path | None,
        typer.Option("--metrics-dir", help="Write track metrics here instead of artifacts/."),
    ] = None,
    write_detections: Annotated[
        bool,
        typer.Option(
            "--write-detections",
            help="Also write the tracker's refined boxes as a detections document.",
        ),
    ] = False,
) -> None:
    """Track saved detections and measure identity quality against ``ape_id``.

    Runs the tracker over ``artifacts/detections/`` rather than over video, so
    settings can be explored on a laptop in seconds without re-running the
    detector. Track ids already stored in those files are discarded and
    recomputed, so the numbers always match the settings printed.

    Every override defaults to the configured value, so plain ``track`` measures
    exactly what ``detect`` would produce. ``--detections-dir`` and
    ``--metrics-dir`` keep an experiment from overwriting a baseline in place.

    ``--write-detections`` also writes the tracker's *output* boxes, which are
    not its input: short tracks have been dropped, gaps interpolated, positions
    smoothed. Running ``evaluate`` over that directory measures the accuracy of
    the tracked result rather than the detector's, and it is the artifact a
    downstream stage such as pose estimation should consume.
    """
    import json as _json

    from panaf_ape_detection.pipeline.retrack import RetrackSettings, load_clip, track_clip

    loaded = _load_or_exit(config)
    if not loaded.tracking.enabled:
        _error_console.print(
            "[red]tracking.enabled is false in this config.[/red] Enable it to run tracking."
        )
        raise typer.Exit(code=1)

    overrides: dict[str, float | int] = {"detection_floor": detection_floor}
    for name, value in (
        ("minimum_track_length", minimum_track_length),
        ("activation_threshold", activation_threshold),
        ("lost_track_buffer", lost_track_buffer),
        ("minimum_matching_threshold", minimum_matching_threshold),
        ("minimum_consecutive_frames", minimum_consecutive_frames),
    ):
        if value is not None:
            overrides[name] = value
    settings = RetrackSettings.from_config(loaded).with_values(overrides)

    sources = _clip_sources(loaded, detections_dir, annotations_dir)
    if not sources:
        _error_console.print(
            "[yellow]No saved detections found.[/yellow] "
            "Run [bold]panaf-phase1 detect[/bold] first."
        )
        raise typer.Exit(code=1)

    destination_root = Path(metrics_dir) if metrics_dir else loaded.repository_paths().artifacts_dir
    evaluations: list[ClipTrackEvaluation] = []

    try:
        for source in sources:
            document, truth = load_clip(source)
            evaluation, tracked = track_clip(source.clip_id, document, truth, settings)
            evaluations.append(evaluation)

            destination = (
                destination_root / "metrics" / TRACK_METRICS_SUBDIR / f"{source.clip_id}.json"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                _json.dumps(
                    {**evaluation.as_dict(), **settings.as_dict()}, indent=2, sort_keys=True
                ),
                encoding="utf-8",
            )

            if write_detections:
                _write_tracked_detections(
                    destination_root / "detections" / f"{source.clip_id}.json",
                    clip_id=source.clip_id,
                    document=document,
                    tracked=tracked,
                    settings=settings,
                )
    except ImportError as exc:
        raise _exit_without_the_inference_extra() from exc

    console.print(_track_table(evaluations))
    console.print(_TRACK_LEGEND)
    console.print(f"\n[dim]tracker {loaded.tracking.backend.value} · {settings.as_dict()}[/dim]")
    console.print(f"Track metrics: {destination_root / 'metrics' / TRACK_METRICS_SUBDIR}")
    if write_detections:
        console.print(f"Tracked boxes: {destination_root / 'detections'}")
        console.print(
            "[dim]`evaluate` over that directory measures the accuracy of the *tracked* "
            "output, which is not the same as the detector's.[/dim]"
        )


@app.command("track-sweep")
def track_sweep(
    grid: Annotated[
        Path, typer.Option("--grid", help="YAML sweep definition: a name and its axes.")
    ],
    config: Annotated[Path, _CONFIG_OPTION] = Path("configs/base.yaml"),
    detections_dir: Annotated[
        Path | None,
        typer.Option("--detections-dir", help="Read detections from here instead of artifacts/."),
    ] = None,
    annotations_dir: Annotated[
        Path | None,
        typer.Option("--annotations-dir", help="Read ground truth from here instead of data/raw."),
    ] = None,
    jobs: Annotated[
        int, typer.Option("--jobs", "-j", min=1, help="Worker processes to sweep arms across.")
    ] = 1,
    top: Annotated[int, typer.Option("--top", min=1, help="Arms to display.")] = 15,
    max_merges: Annotated[
        int | None,
        typer.Option(
            "--max-merges",
            min=0,
            help="Reject arms with more than this many tracks holding 2+ apes.",
        ),
    ] = None,
) -> None:
    """Try many tracker settings over saved detections and rank them.

    Costs no GPU: detection is already done, and this only re-runs association.
    The configured settings are the baseline every arm is compared against, so
    an arm that fails to beat it is a real answer rather than a broken run.

    Arms are ranked by **identity coverage** -- the share of each ape's frames
    held by its single best track. Coverage alone would reward splitting one
    animal across several tracks, and ID switches alone would reward merging two
    animals into one; identity coverage is pushed down by both. ``purity`` and
    ``merges`` are printed beside it so a merge-driven "win" is visible.

    Visible is not the same as prevented. Identity coverage can still be *raised*
    by merging two apes, because both then count the merged track as their
    dominant one. ``--max-merges`` makes the ceiling a rule the tool enforces
    rather than something to notice afterwards, which is worth doing: validating
    on 500 clips produced an arm that improved every other number while merged
    tracks went from 59 to 79.

    Writes every arm to ``artifacts/metrics/tracking-sweep/<name>.json``, which
    is its own directory and schema because this payload is neither detection
    nor per-clip track metrics.
    """
    import json as _json

    import yaml

    from panaf_ape_detection.pipeline.retrack import (
        TRACKING_SWEEP_SCHEMA,
        RetrackSettings,
        expand_grid,
        sweep,
    )
    from panaf_ape_detection.reporting import pooled_track_metrics

    loaded = _load_or_exit(config)
    try:
        definition = yaml.safe_load(Path(grid).read_text(encoding="utf-8")) or {}
        name = str(definition.get("name") or Path(grid).stem)
        axes = definition.get("axes") or {}
        baseline = RetrackSettings.from_config(loaded)
        arms = expand_grid(baseline, axes)
    except FileNotFoundError as exc:
        _error_console.print(f"[red]no sweep grid at {grid}[/red]")
        raise typer.Exit(code=1) from exc
    except (ValueError, yaml.YAMLError) as exc:
        _error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    sources = _clip_sources(loaded, detections_dir, annotations_dir)
    if not sources:
        _error_console.print(
            "[yellow]No saved detections found.[/yellow] "
            "Run [bold]panaf-phase1 detect[/bold] first."
        )
        raise typer.Exit(code=1)

    console.print(
        f"[bold]{name}[/bold]: {len(arms)} arm(s) over {len(sources)} clip(s), {jobs} job(s)"
    )

    started = time.perf_counter()
    try:
        results = list(sweep(sources, arms, jobs=jobs))
    except ImportError as exc:
        raise _exit_without_the_inference_extra() from exc
    elapsed = time.perf_counter() - started

    scored = [
        (pooled_track_metrics([e.as_dict() for e in arm.evaluations]), arm) for arm in results
    ]
    # The configured settings, so "better" always means better than the pipeline.
    reference = next((p for p, arm in scored if arm.settings == baseline), None)
    scored.sort(key=lambda pair: pair[0].identity_coverage, reverse=True)
    ranked = scored

    # Enforce the merge ceiling in the tool rather than by eye. Identity coverage
    # can be *raised* by merging two apes into one track -- both then count the
    # merged track as their dominant one -- so ranking on it alone will happily
    # promote an arm that is more wrong. Validating on 500 clips showed exactly
    # that: merges rose 59 -> 79 while every other number improved.
    if max_merges is not None:
        within = [pair for pair in scored if pair[0].id_merges <= max_merges]
        rejected = len(scored) - len(within)
        if not within:
            _error_console.print(
                f"[red]No arm keeps tracks holding 2+ apes at or below {max_merges}.[/red]\n"
                f"The best any arm managed was {min(p.id_merges for p, _ in scored)}. "
                "Either the ceiling is too strict, or these settings genuinely cannot "
                "separate the animals in this footage."
            )
            raise typer.Exit(code=1)
        console.print(
            f"[dim]{rejected} arm(s) rejected for exceeding {max_merges} merged track(s).[/dim]"
        )
        # Only the *ranking* is filtered. The record below keeps every arm,
        # including the rejected ones -- the trade-off between identity coverage
        # and merged tracks is the finding here, and a file that holds only the
        # arms which passed cannot show it.
        ranked = within

    table = Table(title=f"{name} — ranked by identity coverage", header_style="bold")
    for column in ("ID cov.", "Cov.", "Sw.", "Frag.", "Purity", "Merges", "Jitter", "Settings"):
        table.add_column(column)
    for pooled, arm in ranked[:top]:
        changed = {
            key: value
            for key, value in arm.settings.as_dict().items()
            if value != baseline.as_dict()[key]
        }
        table.add_row(
            f"{pooled.identity_coverage:.4f}",
            f"{pooled.coverage:.4f}",
            str(pooled.id_switches),
            f"{pooled.fragmentation:.2f}",
            f"{pooled.track_purity:.4f}",
            str(pooled.id_merges),
            f"{pooled.jitter:.4f}",
            "baseline"
            if not changed
            else ", ".join(f"{k}={v}" for k, v in sorted(changed.items())),
        )
    console.print(table)
    if reference is not None:
        console.print(
            f"[dim]baseline (the configured settings): identity coverage "
            f"{reference.identity_coverage:.4f}, {reference.id_switches} switches, "
            f"purity {reference.track_purity:.4f}[/dim]"
        )
    else:
        console.print(
            "[yellow]The configured settings are not among the arms,[/yellow] so there is "
            "nothing here to say whether the best arm beats the pipeline. Include them in the grid."
        )

    destination = (
        loaded.repository_paths().artifacts_dir / "metrics" / _SWEEP_METRICS_SUBDIR / f"{name}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _json.dumps(
            {
                "schema": TRACKING_SWEEP_SCHEMA,
                "name": name,
                "clips": [source.clip_id for source in sources],
                "baseline": baseline.as_dict(),
                "max_merges": max_merges,
                "elapsed_seconds": round(elapsed, 2),
                "arms": [
                    {
                        **arm.as_dict(),
                        "pooled": {
                            "identity_coverage": round(pooled.identity_coverage, 4),
                            "coverage": round(pooled.coverage, 4),
                            "id_switches": pooled.id_switches,
                            "fragmentation": round(pooled.fragmentation, 3),
                            "track_purity": round(pooled.track_purity, 4),
                            "id_merges": pooled.id_merges,
                            "jitter": round(pooled.jitter, 5),
                            "mostly_tracked": pooled.mostly_tracked,
                            "mostly_lost": pooled.mostly_lost,
                        },
                        "within_merge_ceiling": (
                            max_merges is None or pooled.id_merges <= max_merges
                        ),
                    }
                    for pooled, arm in scored
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(
        f"\n{len(results)} arm(s) in {elapsed:.1f}s "
        f"({elapsed / max(len(results), 1):.2f}s per arm)\nSweep: {destination}"
    )


def main() -> None:
    """Entry point used by the ``panaf-phase1`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
