# Claude Code hooks

## `ruff-check.sh`

Runs after every `Edit`, `Write` or `MultiEdit`.

**What it does:** reads the tool payload from stdin, extracts the edited file path, and — only if
that path ends in `.py` — runs `uv run ruff check --fix` and `uv run ruff format` on **that one
file**. Everything else exits immediately.

**What it deliberately does not do:** run mypy, pytest, or `verify_repository.py`. Those take
several seconds and belong in `make quality` / `make verify`, which the session skills call. A gate
that runs after every keystroke gets disabled within a day.

**It never blocks.** The script always exits 0. A lint failure it cannot auto-fix will surface in
`make quality` before a commit; it should not interrupt an edit.

**It is a no-op when** stdin is empty, the payload is not JSON, the file is not Python, the file no
longer exists, or `uv` is not on `PATH`.

### Disabling it

Delete the `hooks` block from `.claude/settings.json`, or delete this directory. Nothing else
depends on it — the repository's actual quality contract is `make quality` and `make verify`, both
of which CI enforces independently.

### Why the PATH line

`uv` installs to `~/.local/bin`, which is not always on a hook's `PATH` depending on how the shell
was initialised. The script prepends it rather than assuming.
