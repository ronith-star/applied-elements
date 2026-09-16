#!/usr/bin/env bash
# Commit with the test count MEASURED, never typed.
#
# Two commit messages in this repository (619b368, and da9d6ea which was pushed
# with a failing test) recorded a count that contradicted pytest's own summary
# line while claiming to have been read from it. A number typed into a heredoc
# before the run is a guess with a provenance annotation attached, which is
# worse than an unannotated guess.
#
# This script runs the suite, REFUSES to commit on any failure, and appends the
# measured count to the message body itself.
#
# Usage:  scripts/commit_with_count.sh path/to/message.txt
set -euo pipefail
msg_file="${1:?usage: commit_with_count.sh <message-file>}"
[ -f "$msg_file" ] || { echo "no such message file: $msg_file" >&2; exit 2; }

cd "$(dirname "$0")/.."

# The pytest exit status is captured EXPLICITLY rather than left to set -e.
#
# The first version of this script read
#   out="$(python -m pytest ... | tail -3)"
# under `set -euo pipefail`, so a failing suite aborted the script AT THE
# ASSIGNMENT and the refusal message below was unreachable. Refusal still
# happened, by dying silently with a non-zero status, but the operator saw no
# diagnostic and the branch that was supposed to explain the failure had never
# once executed. A guard whose message never prints is not a guard the operator
# can trust; disabling errexit around the run is what makes the branch live.
set +e
out="$(python -m pytest tests/ src/ 2>&1 | tail -20)"
pytest_status=$?
set -e

summary="$(printf '%s\n' "$out" | grep -Eo '[0-9]+ (passed|failed|error[s]?)[^ ]*' \
           | tr '\n' ' ')"

if [ "$pytest_status" -ne 0 ]; then
  echo "REFUSING TO COMMIT: pytest exited $pytest_status." >&2
  echo "--- last lines of the run ---" >&2
  printf '%s\n' "$out" >&2
  exit 1
fi

# Belt and braces: a zero exit with failure text would also be refused.
if printf '%s' "$out" | grep -qE '[0-9]+ (failed|error)'; then
  echo "REFUSING TO COMMIT: pytest exited 0 but reported failures." >&2
  printf '%s\n' "$out" >&2
  exit 1
fi

passed="$(printf '%s' "$out" | grep -Eo '[0-9]+ passed' | grep -Eo '[0-9]+')"
[ -n "$passed" ] || { echo "could not parse a pass count from pytest" >&2; exit 3; }

tmp="$(mktemp)"
cat "$msg_file" > "$tmp"
printf '\n%s tests pass (count appended by scripts/commit_with_count.sh from\npytest summary: %s).\n' \
  "$passed" "$summary" >> "$tmp"
git add -A
git commit -q -F "$tmp"
rm -f "$tmp"
git log --oneline -1
