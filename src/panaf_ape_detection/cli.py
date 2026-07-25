"""Command-line interface for the Phase 1 scaffold.

Only honest, setup-oriented commands exist right now:

``doctor``
    Report the environment a run would execute in.
``validate-config``
    Load and validate a YAML configuration file.
``show-paths``
    Print the resolved repository layout.

The pipeline commands (``extract-frames``, ``detect``, ``track``, ``annotate``)
are intentionally absent rather than stubbed, so that ``--help`` never advertises
functionality that does not exist. See ``README.md`` for the planned workflow.

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
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from panaf_ape_detection import __version__
from panaf_ape_detection.config import ConfigError, load_config
from panaf_ape_detection.paths import RepositoryPaths, repository_root

__all__ = ["app", "main"]

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="panaf-phase1",
    help=(
        "Phase 1 tooling for pretrained MegaDetector V6 inference over PanAf500 clips. "
        "Setup commands only; the detection pipeline is not implemented yet."
    ),
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
        "\n[dim]Note: detection, tracking and annotation are not implemented yet. "
        "This command only inspects the environment.[/dim]"
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
            "set documented in 05 Technical/model.md. Confirm your installed "
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


def main() -> None:
    """Entry point used by the ``panaf-phase1`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
