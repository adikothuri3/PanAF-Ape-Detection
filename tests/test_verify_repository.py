"""Tests for the repository verifier.

`scripts/verify_repository.py` is the guard the whole repository leans on, and
until now nothing guarded it. A check with a bug does not report a failure — it
silently *passes*, which is precisely what happened when `runtime.module_device`
returned `None` and the smoke test read that as success.

Each test here points the verifier at a deliberately broken tree and asserts the
relevant check **fails**. A guard that has never been observed to fail is not a
guard.

The checks read the module-global `REPO_ROOT` and append to the module-global
`_failures`, so both are redirected per test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from panaf_ape_detection.manifest import MANIFEST_COLUMNS
from panaf_ape_detection.paths import repository_root


def _load_verifier() -> ModuleType:
    """Import `scripts/verify_repository.py` as a module."""
    path = repository_root() / "scripts" / "verify_repository.py"
    spec = importlib.util.spec_from_file_location("_verify_repository_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


@pytest.fixture
def broken_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the verifier at an empty tmp tree with its failure list cleared."""
    monkeypatch.setattr(verifier, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(verifier, "_failures", [])
    for name in ("00 Start Here", "01 Onboarding", "02 Reading", "03 Check-ins", "04 Reference"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    for name in ("docs", "experiments", "reports"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    yield tmp_path


def failures() -> list[str]:
    return verifier._failures


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# The verifier passes on the real repository. If this fails, everything below
# is measuring the wrong thing.
# --------------------------------------------------------------------------- #


def test_real_repository_passes_every_check():
    assert verifier.main([]) == 0


def test_only_flag_runs_a_subset():
    assert verifier.main(["--only", "vault"]) == 0


def test_unknown_group_is_rejected():
    with pytest.raises(SystemExit):
        verifier.main(["--only", "not-a-group"])


# --------------------------------------------------------------------------- #
# Vault checks
# --------------------------------------------------------------------------- #


def test_broken_wikilink_is_caught(broken_repo: Path):
    write(broken_repo / "01 Onboarding" / "Note.md", "See [[No Such Note]] for detail.\n")

    verifier.check_wikilinks_resolve()

    assert any("No Such Note" in f for f in failures())


def test_resolving_wikilink_is_accepted(broken_repo: Path):
    write(broken_repo / "01 Onboarding" / "Target.md", "# Target\n")
    write(broken_repo / "01 Onboarding" / "Source.md", "See [[Target]].\n")

    verifier.check_wikilinks_resolve()

    assert failures() == []


def test_wikilink_inside_a_code_fence_is_ignored(broken_repo: Path):
    write(broken_repo / "01 Onboarding" / "Note.md", "```\n[[Template Placeholder]]\n```\n")

    verifier.check_wikilinks_resolve()

    assert failures() == []


def test_wikilink_inside_inline_code_is_ignored(broken_repo: Path):
    """Prose *about* wikilinks is not a link, e.g. a sentence quoting `[[wikilink]]` inline."""
    write(broken_repo / "01 Onboarding" / "Note.md", "fails on a broken `[[wikilink]]`\n")

    verifier.check_wikilinks_resolve()

    assert failures() == []


def test_wikilink_with_an_alias_resolves_on_the_target(broken_repo: Path):
    write(broken_repo / "01 Onboarding" / "Target.md", "# Target\n")
    write(broken_repo / "01 Onboarding" / "Source.md", "See [[Target|friendly name]].\n")

    verifier.check_wikilinks_resolve()

    assert failures() == []


def test_reading_note_without_status_is_caught(broken_repo: Path):
    write(broken_repo / "02 Reading" / "Reading List.md", "# index\n")
    write(broken_repo / "02 Reading" / "Item.md", "---\ntags: [reading]\n---\n\n# Item\n")

    verifier.check_reading_notes_have_status()

    assert any("no `status:` field" in f for f in failures())


def test_reading_note_with_an_unknown_status_is_caught(broken_repo: Path):
    write(broken_repo / "02 Reading" / "Reading List.md", "# index\n")
    write(broken_repo / "02 Reading" / "Item.md", "---\nstatus: nearly-done\n---\n")

    verifier.check_reading_notes_have_status()

    assert any("nearly-done" in f for f in failures())


def test_reading_note_with_a_known_status_passes(broken_repo: Path):
    write(broken_repo / "02 Reading" / "Reading List.md", "# index\n")
    write(broken_repo / "02 Reading" / "Item.md", "---\nstatus: in-progress\n---\n")

    verifier.check_reading_notes_have_status()

    assert failures() == []


def test_empty_reading_directory_is_caught(broken_repo: Path):
    verifier.check_reading_notes_have_status()

    assert any("no reading notes" in f for f in failures())


def test_a_second_research_log_is_caught(broken_repo: Path):
    write(broken_repo / "experiments" / "experiment_log.md", "# log\n")
    write(broken_repo / "03 Check-ins" / "Experiment Log.md", "# a rival log\n")

    verifier.check_no_second_research_log()

    assert any("second research log" in f for f in failures())


def test_missing_canonical_log_is_caught(broken_repo: Path):
    verifier.check_no_second_research_log()

    assert any("experiment_log.md is missing" in f for f in failures())


# --------------------------------------------------------------------------- #
# Notebook checks
# --------------------------------------------------------------------------- #


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def test_invalid_notebook_json_is_caught(broken_repo: Path):
    write(broken_repo / "notebooks" / "bad.ipynb", "{not json")

    verifier.check_notebooks_are_valid_json()

    assert any("not valid JSON" in f for f in failures())


def test_notebook_missing_required_keys_is_caught(broken_repo: Path):
    write(broken_repo / "notebooks" / "bad.ipynb", json.dumps({"cells": []}))

    verifier.check_notebooks_are_valid_json()

    assert any("nbformat" in f for f in failures())


def test_stored_notebook_output_is_caught(broken_repo: Path):
    """Output cells can embed dataset frames -- a licensing problem, not just noise."""
    cell = {"cell_type": "code", "source": [], "outputs": [{"text": "hi"}], "execution_count": 1}
    write(broken_repo / "notebooks" / "run.ipynb", json.dumps(notebook([cell])))

    verifier.check_notebooks_have_no_stored_output()

    assert any("stored output" in f for f in failures())


def test_execution_count_alone_is_caught(broken_repo: Path):
    cell = {"cell_type": "code", "source": [], "outputs": [], "execution_count": 3}
    write(broken_repo / "notebooks" / "run.ipynb", json.dumps(notebook([cell])))

    verifier.check_notebooks_have_no_stored_output()

    assert any("execution count" in f for f in failures())


def test_clean_notebook_passes(broken_repo: Path):
    cell = {"cell_type": "code", "source": [], "outputs": [], "execution_count": None}
    write(broken_repo / "notebooks" / "run.ipynb", json.dumps(notebook([cell])))

    verifier.check_notebooks_have_no_stored_output()

    assert failures() == []


# --------------------------------------------------------------------------- #
# Data and honesty checks
# --------------------------------------------------------------------------- #


def test_manifest_row_without_a_placeholder_marker_is_caught(broken_repo: Path):
    """Real clip ids and checksums must never reach the tracked template."""
    header = ",".join(MANIFEST_COLUMNS)
    write(
        broken_repo / "data" / "sample_manifest.example.csv",
        f"{header}\nclip-042,train,real.mp4,real.json,chimpanzee,site-a,because,{'a' * 64},"
        f"{'b' * 64},\n",
    )

    verifier.check_manifest_example_has_no_real_data()

    assert any("not marked as a placeholder" in f for f in failures())


def test_manifest_with_a_wrong_header_is_caught(broken_repo: Path):
    write(broken_repo / "data" / "sample_manifest.example.csv", "clip_id,notes\nPLACEHOLDER,x\n")

    verifier.check_manifest_example_has_no_real_data()

    assert any("header is" in f for f in failures())


def test_prefilled_writeup_template_is_caught(broken_repo: Path):
    """A template with no placeholders left is a template someone filled in."""
    write(
        broken_repo / "reports" / "phase1_writeup_template.md",
        "# Findings\n\nDetection worked well and we measured 87% recall.\n",
    )

    verifier.check_no_fabricated_results()

    assert any("no remaining placeholders" in f for f in failures())


def test_template_with_placeholders_passes(broken_repo: Path):
    write(broken_repo / "reports" / "phase1_writeup_template.md", "# Findings\n\nTODO\n")

    verifier.check_no_fabricated_results()

    assert failures() == []


def test_licence_inconsistency_is_caught(broken_repo: Path):
    """A LICENSE file while the docs still say none was chosen is a contradiction."""
    write(broken_repo / "LICENSE", "MIT License\n")
    write(broken_repo / "docs" / "licensing.md", "No code licence has been selected.\n")

    verifier.check_license_documentation_is_consistent()

    assert any("still says no code licence" in f for f in failures())


def test_missing_licence_declaration_is_caught(broken_repo: Path):
    write(broken_repo / "docs" / "licensing.md", "Everything is MIT.\n")

    verifier.check_license_documentation_is_consistent()

    assert any("no longer states" in f for f in failures())


def test_missing_required_files_are_caught(broken_repo: Path):
    verifier.check_required_paths()

    assert any("missing required file" in f for f in failures())
