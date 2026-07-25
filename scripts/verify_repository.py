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
import re
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
    "scripts/smoke_detect.py",
    "scripts/smoke_inference.py",
    "scripts/verify_repository.py",
    "src/panaf_ape_detection/__init__.py",
    "src/panaf_ape_detection/cli.py",
    "src/panaf_ape_detection/config.py",
    "src/panaf_ape_detection/manifest.py",
    "src/panaf_ape_detection/paths.py",
    "src/panaf_ape_detection/provenance.py",
    "src/panaf_ape_detection/py.typed",
    "src/panaf_ape_detection/runtime.py",
    "src/panaf_ape_detection/types.py",
    "tests/test_cli.py",
    "tests/test_config.py",
    "tests/test_manifest.py",
    "tests/test_paths.py",
    "tests/test_provenance.py",
    "tests/test_runtime.py",
    "tests/test_smoke_detect.py",
    "tests/test_types.py",
    "tests/test_verify_repository.py",
    # Claude Code integration
    ".claude/settings.json",
    ".claude/hooks/ruff-check.sh",
    ".claude/skills/start-session/SKILL.md",
    ".claude/skills/log-session/SKILL.md",
    ".claude/skills/finish-phase/SKILL.md",
    # Obsidian vault -- the repository root doubles as the vault
    ".obsidian/app.json",
    ".obsidian/core-plugins.json",
    "00 Start Here/PanAf Command Center.md",
    "01 Onboarding/Geologic Dome Context.md",
    "01 Onboarding/Four Phase Arc.md",
    "01 Onboarding/Phase 1 Task Spec.md",
    "01 Onboarding/How We Work.md",
    "02 Reading/Reading List.md",
    "03 Check-ins/Check-in Template.md",
    "04 Reference/PanAf500 Action Labels.md",
    "04 Reference/MegaDetector Variants.md",
    "04 Reference/Glossary.md",
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
    "00 Start Here",
    "01 Onboarding",
    "02 Reading",
    "03 Check-ins",
    "04 Reference",
)

# Numbered note folders that make up the added vault layer. `docs/`, `experiments/`
# and `reports/` are part of the same vault but predate it and are checked elsewhere.
VAULT_DIRECTORIES: tuple[str, ...] = (
    "00 Start Here",
    "01 Onboarding",
    "02 Reading",
    "03 Check-ins",
    "04 Reference",
)

# Allowed `status:` values for a note in `02 Reading/`.
READING_STATUSES: frozenset[str] = frozenset({"not-started", "in-progress", "done"})

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

    # Single source of truth: the package owns the column order, so the template,
    # data/README.md and this check cannot drift apart.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from panaf_ape_detection.manifest import MANIFEST_COLUMNS
    except ImportError as exc:
        fail(f"cannot import panaf_ape_detection.manifest: {exc}")
        return

    header = lines[0].split(",")
    expected = list(MANIFEST_COLUMNS)
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


def _markdown_notes() -> list[Path]:
    """Return every Markdown file Obsidian treats as a note in this vault."""
    notes: list[Path] = []
    for directory in (*VAULT_DIRECTORIES, "docs", "experiments", "reports"):
        notes.extend(sorted((REPO_ROOT / directory).rglob("*.md")))
    notes.extend(
        REPO_ROOT / name
        for name in ("README.md", "CLAUDE.md", "CONTRIBUTING.md")
        if (REPO_ROOT / name).is_file()
    )
    return notes


def check_wikilinks_resolve() -> None:
    """Every ``[[wikilink]]`` points at a note that exists.

    Broken links are how a vault rots: Obsidian shows them as a slightly different
    colour and nothing else complains, so they accumulate silently. This is the
    highest-value vault check.
    """
    note_names = {path.stem for path in _markdown_notes()}
    # Obsidian also resolves links to any Markdown file in the vault, not just the
    # curated folders above -- include everything outside ignored directories.
    for path in REPO_ROOT.rglob("*.md"):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative.startswith((".venv/", "node_modules/", "artifacts/", ".claude/")):
            continue
        note_names.add(path.stem)

    pattern = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
    for note in _markdown_notes():
        relative = note.relative_to(REPO_ROOT).as_posix()
        text = note.read_text(encoding="utf-8")
        # Skip fenced code blocks (templates embed example wikilinks not meant to
        # resolve until copied) and inline code spans (prose *about* wikilinks,
        # e.g. "fails on a broken `[[wikilink]]`", is not a link).
        prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        prose = re.sub(r"`[^`\n]*`", "", prose)
        for match in pattern.finditer(prose):
            target = match.group(1).strip()
            if not target:
                continue
            # A link may name a path (`docs/model`) or just the note stem.
            stem = Path(target).stem
            if stem not in note_names:
                fail(f"{relative}: wikilink [[{target}]] does not resolve to any note")


def check_reading_notes_have_status() -> None:
    """Every note in `02 Reading/` declares a known `status:` in its frontmatter."""
    reading_dir = REPO_ROOT / "02 Reading"
    notes = [p for p in sorted(reading_dir.glob("*.md")) if p.name != "Reading List.md"]
    if not notes:
        fail("02 Reading/ contains no reading notes")

    for note in notes:
        relative = note.relative_to(REPO_ROOT).as_posix()
        text = note.read_text(encoding="utf-8")
        if not text.startswith("---"):
            fail(f"{relative}: missing YAML frontmatter")
            continue
        match = re.search(r"^status:\s*(\S+)\s*$", text, re.MULTILINE)
        if match is None:
            fail(f"{relative}: frontmatter has no `status:` field")
            continue
        status = match.group(1).strip().strip("\"'")
        if status not in READING_STATUSES:
            valid = ", ".join(sorted(READING_STATUSES))
            fail(f"{relative}: status {status!r} is not one of: {valid}")


def check_vault_local_state_not_tracked() -> None:
    """Per-machine Obsidian UI state must not be committed."""
    if not _in_git_repository():
        print("  (skipped: not a git repository yet)")
        return

    tracked = set(_git("ls-files"))
    for local_only in (".obsidian/workspace.json", ".obsidian/workspace-mobile.json"):
        if local_only in tracked:
            fail(f"{local_only} is tracked; it is per-machine UI state and should be git-ignored")


def check_no_second_research_log() -> None:
    """There is exactly one running research log.

    The onboarding asks for *one* running doc. A second log in the vault would
    split the record and neither half would be trustworthy.
    """
    canonical = REPO_ROOT / "experiments" / "experiment_log.md"
    if not canonical.is_file():
        fail("experiments/experiment_log.md is missing; it is the single running research log")
        return

    for directory in VAULT_DIRECTORIES:
        for note in (REPO_ROOT / directory).rglob("*.md"):
            stem = note.stem.lower()
            if "experiment log" in stem or "experiment_log" in stem or stem.endswith("session log"):
                relative = note.relative_to(REPO_ROOT).as_posix()
                fail(
                    f"{relative} looks like a second research log. There is exactly one, at "
                    "experiments/experiment_log.md -- link to it instead."
                )


def check_report_figures_are_trackable() -> None:
    """Write-up figures must not be silently git-ignored.

    ``reports/phase1_writeup_template.md`` asks for a figure in
    ``reports/figures/``, but the blanket ``*.png`` rule in ``.gitignore``
    previously swallowed it — with no error, so the deliverable would simply be
    missing a figure nobody noticed was dropped.
    """
    if not _in_git_repository():
        print("  (skipped: not a git repository yet)")
        return

    for name in ("audit_probe.png", "audit_probe.jpg", "audit_probe.gif", "audit_probe.svg"):
        candidate = f"reports/figures/{name}"
        result = subprocess.run(
            ["git", "check-ignore", "-q", candidate],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        # check-ignore exits 0 when the path IS ignored.
        if result.returncode == 0:
            fail(
                f"{candidate} would be git-ignored, but the write-up template asks for a figure "
                "there. Add a negation for reports/figures/ in .gitignore."
            )


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
    ("data", "write-up figures are not git-ignored", check_report_figures_are_trackable),
    ("vault", "wikilinks resolve to real notes", check_wikilinks_resolve),
    ("vault", "reading notes declare a known status", check_reading_notes_have_status),
    ("vault", "Obsidian local UI state is not tracked", check_vault_local_state_not_tracked),
    ("vault", "there is exactly one research log", check_no_second_research_log),
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
