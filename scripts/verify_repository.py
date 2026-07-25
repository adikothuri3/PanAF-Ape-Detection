#!/usr/bin/env python3
"""Verify the repository's structural invariants.

This script is run in CI and by ``make verify``. It is a guard against the ways
a research repository quietly rots: a dataset file committed by accident, a
config that no longer validates, a notebook that stops being valid JSON, a
pipeline command that got stubbed out with a fake implementation.

It performs **no** network access, downloads no weights and requires no GPU.

Exit status is 0 when every check passes and 1 otherwise; each failure is
printed with enough context to act on.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files that must exist for the repository to be usable by a new contributor.
REQUIRED_FILES: tuple[str, ...] = (
    ".env.example",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".python-version",
    "CITATION.cff",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "Makefile",
    "README.md",
    "pyproject.toml",
    "references.bib",
    "requirements-colab.txt",
    "uv.lock",
    ".github/dependabot.yml",
    ".github/pull_request_template.md",
    ".github/workflows/ci.yml",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/experiment.yml",
    "configs/base.yaml",
    "configs/colab.yaml",
    "data/README.md",
    "data/sample_manifest.example.csv",
    "docs/architecture.md",
    "docs/dataset.md",
    "docs/licensing.md",
    "docs/model.md",
    "docs/reproducibility.md",
    "experiments/README.md",
    "experiments/experiment_log.md",
    "notebooks/README.md",
    "notebooks/phase1_colab.ipynb",
    "reports/phase1_writeup_template.md",
    "scripts/check_environment.py",
    "scripts/verify_repository.py",
    "src/panaf_ape_detection/__init__.py",
    "src/panaf_ape_detection/cli.py",
    "src/panaf_ape_detection/config.py",
    "src/panaf_ape_detection/paths.py",
    "src/panaf_ape_detection/py.typed",
    "src/panaf_ape_detection/types.py",
    "tests/test_cli.py",
    "tests/test_config.py",
    "tests/test_paths.py",
)

REQUIRED_DIRECTORIES: tuple[str, ...] = (
    "configs",
    "data/interim",
    "data/processed",
    "data/raw",
    "docs",
    "experiments",
    "notebooks",
    "reports/figures",
    "scripts",
    "src/panaf_ape_detection",
    "tests",
)

# Extensions that must never be tracked by git, whatever the directory.
FORBIDDEN_TRACKED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".avi",
        ".bin",
        ".ckpt",
        ".h5",
        ".hdf5",
        ".mkv",
        ".mov",
        ".mp4",
        ".npy",
        ".npz",
        ".onnx",
        ".pb",
        ".pt",
        ".pth",
        ".safetensors",
        ".webm",
        ".weights",
    }
)

# Paths that must never be tracked, by prefix.
FORBIDDEN_TRACKED_PREFIXES: tuple[str, ...] = ("artifacts/", "data/raw/", "data/interim/")

# Tracked files that are allowed to be large, with a justification.
LARGE_FILE_ALLOWLIST: frozenset[str] = frozenset({"uv.lock", "requirements-colab.txt"})

MAX_TRACKED_FILE_BYTES = 1_000_000

# Commands the CLI is allowed to expose while the pipeline is unimplemented.
EXPECTED_CLI_COMMANDS: frozenset[str] = frozenset({"doctor", "validate-config", "show-paths"})

_failures: list[str] = []


def fail(message: str) -> None:
    """Record a verification failure."""
    _failures.append(message)


def _git(*args: str) -> list[str]:
    """Run a git command in the repository and return its stdout lines."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def _in_git_repository() -> bool:
    """Return whether the working directory is inside a git repository."""
    return bool(_git("rev-parse", "--is-inside-work-tree"))


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_required_paths() -> None:
    """Every required file and directory exists."""
    for relative in REQUIRED_FILES:
        if not (REPO_ROOT / relative).is_file():
            fail(f"missing required file: {relative}")
    for relative in REQUIRED_DIRECTORIES:
        if not (REPO_ROOT / relative).is_dir():
            fail(f"missing required directory: {relative}")


def check_license_documentation_is_consistent() -> None:
    """The presence of a LICENSE file must agree with docs/licensing.md.

    Adding a LICENSE without updating the documentation (or vice versa) leaves
    the repository making two contradictory claims about reuse rights.
    """
    license_present = any(
        (REPO_ROOT / name).is_file() for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")
    )
    licensing_doc = (REPO_ROOT / "docs" / "licensing.md").read_text(encoding="utf-8").lower()
    declares_unselected = (
        "no code licence has been selected" in licensing_doc
        or "no code license has been selected" in licensing_doc
    )

    if not license_present and not declares_unselected:
        fail(
            "no LICENSE file exists, but docs/licensing.md no longer states that the code "
            "licence is unselected. Keep the two consistent."
        )
    if license_present and declares_unselected:
        fail(
            "a LICENSE file exists, but docs/licensing.md still says no code licence has been "
            "selected. Update docs/licensing.md, CITATION.cff and the README licensing table."
        )


def check_configs_validate() -> None:
    """Every YAML file in configs/ loads against the typed model."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from panaf_ape_detection.config import ConfigError, load_config
    except ImportError as exc:
        fail(f"cannot import panaf_ape_detection.config: {exc}")
        return

    configs = sorted((REPO_ROOT / "configs").glob("*.yaml"))
    if not configs:
        fail("configs/ contains no YAML files")
    for config_path in configs:
        try:
            load_config(config_path, use_env_overrides=False)
        except ConfigError as exc:
            fail(f"{config_path.relative_to(REPO_ROOT)} does not validate:\n{exc}")


def check_notebooks_are_valid_json() -> None:
    """Every notebook parses as JSON and has the expected nbformat keys."""
    notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    if not notebooks:
        fail("notebooks/ contains no .ipynb files")
    for notebook_path in notebooks:
        relative = notebook_path.relative_to(REPO_ROOT)
        try:
            document = json.loads(notebook_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"{relative} is not valid JSON: {exc}")
            continue
        for key in ("cells", "metadata", "nbformat", "nbformat_minor"):
            if key not in document:
                fail(f"{relative} is missing the required notebook key {key!r}")


def check_notebooks_have_no_stored_output() -> None:
    """Notebooks are committed without executed output.

    Stored output is how dataset frames and fabricated-looking results leak into
    version control.
    """
    for notebook_path in sorted((REPO_ROOT / "notebooks").glob("*.ipynb")):
        try:
            document = json.loads(notebook_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue  # already reported
        relative = notebook_path.relative_to(REPO_ROOT)
        for index, cell in enumerate(document.get("cells", [])):
            if cell.get("outputs"):
                fail(f"{relative} cell {index} has stored output; clear it before committing")
            if cell.get("execution_count") is not None:
                fail(f"{relative} cell {index} has an execution count; clear it before committing")


def check_cli_exposes_only_implemented_commands() -> None:
    """The CLI must not advertise unimplemented pipeline stages."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from panaf_ape_detection.cli import app
    except ImportError as exc:
        fail(f"cannot import panaf_ape_detection.cli: {exc}")
        return

    registered = {
        command.name or (command.callback.__name__ if command.callback else "")
        for command in app.registered_commands
    }
    unexpected = registered - EXPECTED_CLI_COMMANDS
    if unexpected:
        fail(
            f"CLI exposes unexpected command(s): {sorted(unexpected)}. "
            "Pipeline commands must be implemented and tested before being registered, "
            "and this allowlist updated in the same change."
        )
    missing = EXPECTED_CLI_COMMANDS - registered
    if missing:
        fail(f"CLI is missing expected command(s): {sorted(missing)}")


def check_package_imports_without_inference_extra() -> None:
    """Importing the package must not require the heavy ML stack."""
    code = (
        "import sys; "
        "blocked = {'torch', 'torchvision', 'PytorchWildlife', 'cv2', 'supervision'}; "
        "import panaf_ape_detection, panaf_ape_detection.cli; "
        "leaked = sorted(blocked & set(sys.modules)); "
        "sys.exit('eagerly imported: ' + ', '.join(leaked) if leaked else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    if result.returncode != 0:
        fail(
            "importing panaf_ape_detection pulled in heavyweight ML modules or failed:\n"
            f"{result.stdout.strip()}\n{result.stderr.strip()}"
        )


def check_manifest_example_has_no_real_data() -> None:
    """The example manifest must contain headers and placeholders only."""
    path = REPO_ROOT / "data" / "sample_manifest.example.csv"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        fail("data/sample_manifest.example.csv is empty")
        return

    header = lines[0].split(",")
    expected = [
        "clip_id",
        "split",
        "video_filename",
        "annotation_filename",
        "species",
        "site",
        "selected_reason",
        "video_sha256",
        "annotation_sha256",
        "notes",
    ]
    if header != expected:
        fail(f"data/sample_manifest.example.csv header is {header}, expected {expected}")

    for row in lines[1:]:
        if "PLACEHOLDER" not in row.upper():
            fail(
                "data/sample_manifest.example.csv contains a row that is not marked as a "
                f"placeholder: {row[:80]!r}. Real clip ids and checksums must not be committed."
            )


def check_no_forbidden_files_tracked() -> None:
    """Git must not track data, weights, media or generated artifacts."""
    if not _in_git_repository():
        print("  (skipped: not a git repository yet)")
        return

    tracked = _git("ls-files")
    for relative in tracked:
        suffix = Path(relative).suffix.lower()
        if suffix in FORBIDDEN_TRACKED_SUFFIXES:
            fail(f"forbidden file type tracked by git: {relative}")
        if relative.startswith(FORBIDDEN_TRACKED_PREFIXES) and not relative.endswith(
            (".gitkeep", "README.md")
        ):
            fail(f"file tracked from an ignored data/artifact directory: {relative}")
        if relative == "data/sample_manifest.csv":
            fail("data/sample_manifest.csv is tracked; only the .example.csv template may be")
        if relative == ".env":
            fail(".env is tracked; secrets must never be committed")


def check_no_large_tracked_files() -> None:
    """No tracked file exceeds the size limit without being allowlisted."""
    if not _in_git_repository():
        print("  (skipped: not a git repository yet)")
        return

    for relative in _git("ls-files"):
        if relative in LARGE_FILE_ALLOWLIST:
            continue
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_TRACKED_FILE_BYTES:
            fail(f"{relative} is {size:,} bytes, over the {MAX_TRACKED_FILE_BYTES:,} byte limit")


def check_lockfile_is_current() -> None:
    """uv.lock must be in sync with pyproject.toml."""
    result = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        fail(
            "uv.lock is out of date with pyproject.toml; run `uv lock` and commit the result.\n"
            f"{result.stderr.strip()}"
        )


def check_colab_requirements_derived_from_lock() -> None:
    """requirements-colab.txt must look like a uv export, not a hand-written list."""
    path = REPO_ROOT / "requirements-colab.txt"
    text = path.read_text(encoding="utf-8")
    if "uv export" not in text:
        fail(
            "requirements-colab.txt does not carry the `uv export` provenance header; "
            "regenerate it with `make colab-requirements` rather than editing it by hand"
        )
    if "pytorchwildlife==" not in text.lower():
        fail("requirements-colab.txt does not pin pytorchwildlife; regenerate it from uv.lock")


def check_no_fabricated_results() -> None:
    """Report templates must not have been pre-filled with invented findings."""
    template = REPO_ROOT / "reports" / "phase1_writeup_template.md"
    text = template.read_text(encoding="utf-8")
    if "TODO" not in text and "_TBD_" not in text:
        fail(
            "reports/phase1_writeup_template.md has no remaining placeholders; a template "
            "should not contain conclusions. Copy it to a dated write-up before filling it in."
        )


# (group, description, callable). The group lets a caller -- notably the
# pre-commit hook -- run a focused subset without duplicating the implementation.
CHECKS: tuple[tuple[str, str, Callable[[], None]], ...] = (
    ("structure", "required files and directories exist", check_required_paths),
    ("licensing", "licence documentation is consistent", check_license_documentation_is_consistent),
    ("config", "configs validate against the typed model", check_configs_validate),
    ("notebooks", "notebooks are valid JSON", check_notebooks_are_valid_json),
    ("notebooks", "notebooks carry no stored output", check_notebooks_have_no_stored_output),
    ("cli", "CLI exposes only implemented commands", check_cli_exposes_only_implemented_commands),
    (
        "cli",
        "package imports without the inference extra",
        check_package_imports_without_inference_extra,
    ),
    ("data", "example manifest contains no real data", check_manifest_example_has_no_real_data),
    ("data", "no data, weights or artifacts are tracked", check_no_forbidden_files_tracked),
    ("data", "no oversized files are tracked", check_no_large_tracked_files),
    ("deps", "uv.lock is in sync with pyproject.toml", check_lockfile_is_current),
    (
        "deps",
        "requirements-colab.txt is derived from uv.lock",
        check_colab_requirements_derived_from_lock,
    ),
    ("honesty", "no fabricated results in templates", check_no_fabricated_results),
)

GROUPS: tuple[str, ...] = tuple(dict.fromkeys(group for group, _, _ in CHECKS))


def main(argv: list[str] | None = None) -> int:
    """Run the selected checks and report the outcome.

    Args:
        argv: Command-line arguments. Defaults to :data:`sys.argv`.

    Returns:
        ``0`` when all selected checks pass, ``1`` otherwise.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--only",
        choices=GROUPS,
        metavar="GROUP",
        help=f"Run only one group of checks. One of: {', '.join(GROUPS)}.",
    )
    # pre-commit passes matched filenames; the checks scan the repository
    # themselves, so the list is accepted and ignored.
    parser.add_argument("files", nargs="*", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)

    selected = [
        (description, check)
        for group, description, check in CHECKS
        if arguments.only is None or group == arguments.only
    ]

    scope = f" [{arguments.only}]" if arguments.only else ""
    print(f"Verifying repository at {REPO_ROOT}{scope}\n")

    for description, check in selected:
        before = len(_failures)
        print(f"- {description}")
        try:
            check()
        except Exception as exc:
            fail(f"check {description!r} raised {type(exc).__name__}: {exc}")
        if len(_failures) == before:
            print("  ok")

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} problem(s) across {len(selected)} check(s)\n")
        for problem in _failures:
            print(f"  * {problem}")
        return 1

    print(f"PASSED: {len(selected)} check(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
