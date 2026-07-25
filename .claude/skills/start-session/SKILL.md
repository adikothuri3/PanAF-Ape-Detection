---
name: start-session
description: Use at the beginning of a working session on the PanAf ape detection project. Reads the Command Center, reports the current phase, active task, next recommended task and reading progress, checks the environment with `panaf-phase1 doctor`, and confirms the working tree is clean.
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash
---

# Start Session

Orient before touching anything. This is read-only — it changes no files.

## Steps

1. **Read the Command Center.** `docs/obsidian/00 Start Here/PanAf Command Center.md` is the single entry point.
   Note Current Status, Current Phase, Current Active Task, Next Recommended Task and Known gaps.
2. **Check the environment.** Run `uv run panaf-phase1 doctor`. Flag anything the session will
   need and does not have — FFmpeg and the `inference` extra are the usual gaps.
3. **Check the tree.** `git status --short --branch`. Report uncommitted changes and whether the
   branch is behind `origin/main`. A dirty tree at the start of a session is usually unfinished
   work from the last one — say so rather than building on top of it silently.
4. **Check reading progress.** Read `docs/obsidian/02 Reading/Reading List.md` and report which of this week's
   four items are still outstanding.
5. **Report and propose.** Summarise the above in a few lines, then propose the single next action
   and wait for confirmation.

## Rules

- **Report, do not fix.** If the tree is dirty or a check fails, say so and let the user decide.
  Do not commit, stash, or install anything.
- **Do not invent progress.** If the Command Center says no detection has been run, that is the
  truth regardless of how much code exists.
- If the Command Center's `updated:` date is well behind the last commit, mention that it may be
  stale rather than trusting it blindly.

## Output

Current phase · active task · next recommended task · environment gaps · tree state · outstanding
reading · the one action you propose starting with.

## Related

`docs/obsidian/00 Start Here/PanAf Command Center.md` · `docs/obsidian/01 Onboarding/How We Work.md` · `/log-session` · `/finish-phase`
