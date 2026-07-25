#!/usr/bin/env bash
# PostToolUse hook: lint and format the single Python file that was just edited.
#
# Deliberately light. The full gate (mypy, pytest, verify_repository) is slow enough that running
# it after every edit would be a tax on every keystroke, so it stays in `make quality` and the
# session skills. This hook only keeps individual files clean as they are written.
#
# Always exits 0 — advisory, never blocks. A lint error should surface in `make quality`, not
# derail an edit mid-thought.
set -uo pipefail

# Claude passes the tool payload as JSON on stdin. `jq` is not guaranteed to be installed;
# python3 is, since this is a Python project.
PAYLOAD="$(cat 2>/dev/null || true)"
[ -z "$PAYLOAD" ] && exit 0

FILE_PATH="$(
  printf '%s' "$PAYLOAD" | python3 -c '
import json, sys
try:
    payload = json.load(sys.stdin)
except (json.JSONDecodeError, ValueError):
    sys.exit(0)
tool_input = payload.get("tool_input") or {}
print(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
' 2>/dev/null
)"

# Nothing to do unless a Python file was touched.
case "$FILE_PATH" in
  *.py) ;;
  *) exit 0 ;;
esac

[ -f "$FILE_PATH" ] || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# uv lives in ~/.local/bin, which is not always on a hook's PATH.
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || exit 0

echo "ruff: $(basename "$FILE_PATH")"
uv run --quiet ruff check --fix "$FILE_PATH" || true
uv run --quiet ruff format "$FILE_PATH" || true

exit 0
