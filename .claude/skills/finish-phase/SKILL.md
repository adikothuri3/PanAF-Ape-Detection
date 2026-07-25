---
name: finish-phase
description: Use when closing out a Phase 1 sub-phase (1a-1f) on the PanAf ape detection project. Runs the full quality gate, checks the deliverable checklist honestly, creates a dated write-up from the template if the phase is complete, drafts a weekly check-in, and logs the session.
user-invocable: true
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
---

# Finish Phase

Close out a sub-phase properly. Heavier than `/log-session` — this runs the gates and produces the
artefacts a phase is supposed to leave behind.

## Steps

1. **Run the full gate.** All of it, and report actual output:
   ```bash
   make quality        # ruff · ruff format --check · mypy · pytest
   make verify         # scripts/verify_repository.py
   ```
   If anything fails, **stop and fix it**. Do not proceed to the write-up with a red gate, and do
   not describe a failing gate as passing.

2. **Confirm the tree and lockfile.** `git status`; if dependencies changed, `make lock` and commit
   both `uv.lock` and `requirements-colab.txt`. If the `inference` extra changed, verify the import
   actually works — `uv run --extra inference python -c "import PytorchWildlife"` — because a
   successful resolve is not a successful import.

3. **Walk the deliverable checklist** in `docs/obsidian/00 Start Here/PanAf Command Center.md` against reality.
   Tick only what exists. If 2–3 annotated clips do not exist in `artifacts/videos/`, that box stays
   unticked no matter how close the code is.

4. **Write-up — only when Phase 1 is genuinely complete.** Copy
   `reports/phase1_writeup_template.md` to `reports/phase1_findings_YYYY-MM-DD.md` and fill it from
   recorded runs. Leave `TODO` wherever there is no measurement. Never edit the template itself —
   `verify_repository.py` fails if its placeholders disappear.

5. **Draft a check-in.** New dated note in `docs/obsidian/03 Check-ins/` from `Check-in Template.md`. Include what
   is stuck, with the three required fields: what you tried, the exact error, what you expected.

6. **Log the session** — follow `/log-session` and append to `experiments/experiment_log.md`.

7. **Update the Command Center** — Current Status, Current Phase, Current Active Task, Next
   Recommended Task, `phase:` and `updated:` frontmatter.

8. **Report.** Verbatim gate results, what was produced, what remains.

## Rules

- **Report outcomes faithfully.** If tests fail, say so and show the output. If a step was skipped,
  say which and why. Never call a phase done because most of it is.
- **No fabricated findings.** No invented metrics, no illustrative detections, no pre-filled
  conclusions. Every number in a write-up traces to a recorded run.
- **State the model variant and confidence threshold** anywhere a result appears.
- **Do not commit or push unless asked.** Prepare the work; the user decides when it ships.
- A phase with a failing gate is not finished. Say that plainly rather than softening it.

## Output

Gate results (verbatim) · deliverable checklist with real state · files created · what is still
outstanding · the recommended next sub-phase.

## Related

`docs/obsidian/00 Start Here/PanAf Command Center.md` · `docs/obsidian/01 Onboarding/Phase 1 Task Spec.md` · `reports/phase1_writeup_template.md` · `/log-session`
