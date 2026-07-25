---
name: log-session
description: Use at the end of a working session on the PanAf ape detection project to record what happened. Appends an entry to experiments/experiment_log.md using the template in that file, and updates the Command Center and reading-note statuses only where they actually changed.
user-invocable: true
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
---

# Log Session

Capture what happened into the running research log. The onboarding asks for "one running doc of
what you tried each session, **dead ends included**" — this writes it.

## Where the log lives

**`experiments/experiment_log.md`, and nowhere else.** There is exactly one running log in this
project. Do not create a session note in the vault; do not start a parallel log. The entry template
is already at the top of that file — use it as-is, newest entry appended at the **bottom**.

## Steps

1. **Summarise the session.** What was attempted, what happened, what was decided or discovered.
2. **Append a log entry** to `experiments/experiment_log.md` using that file's own template.
   Every field, in order. Fields that do not apply get an explicit "not applicable" or
   "not measured" — not a deleted heading.
3. **Command Center** — update `00 Start Here/PanAf Command Center.md` only if Current Status,
   Current Phase, Current Active Task, Next Recommended Task, the deliverable checklist or Known
   gaps actually changed. Bump `updated:` when you change anything.
4. **Reading notes** — if a reading item was started or finished, update its `status:` frontmatter
   in `02 Reading/` **and** the matching row in `02 Reading/Reading List.md`. Keep the two in sync;
   `verify_repository.py --only vault` checks the frontmatter.
5. **Check-in** — if something blocking came up, note it under Standing questions in
   `03 Check-ins/Check-in Template.md` or draft a dated check-in.
6. **Verify** — run `uv run python scripts/verify_repository.py --only vault` to confirm no
   wikilink was broken.

## Rules

- **Only update what actually changed.** Do not manufacture entries. An honest short log beats a
  padded one.
- **Never invent a number.** If it was not measured, write "not measured". A plausible-looking
  figure will be quoted later by someone who assumes it was real.
- **Record failures and dead ends.** An entry with no failures is usually one written from memory.
- **Paste exact errors**, verbatim, including the traceback. Never paraphrase.
- **Record the model variant and confidence threshold** for any entry involving inference. A
  detection count without them cannot be compared to anything.
- **Record whether the working tree was dirty.** A commit SHA from a modified tree describes code
  that did not run.
- Does **not** run quality checks or close a phase — that is `/finish-phase`.

## Output

What was logged and where, as links. What you deliberately did **not** update, and why.

## Related

`experiments/experiment_log.md` · `01 Onboarding/How We Work.md` · `00 Start Here/PanAf Command Center.md` · `/finish-phase`
