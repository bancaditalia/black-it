#!/usr/bin/env bash
#
# Compare the set of files touched between two git revisions. By default, only
# modified and new files are listed, while deleted files are not shown.
set -eu

read -r -d '' USAGE <<'EOF' || true
USAGE:
    ./get-changed-files.sh REVISION_1 REVISION_2 [filter-pattern]

    The filter pattern must be quoted, and it must follow the regex syntax (it
    can't be a bash glob).

EXAMPLES:
    ./get-changed-files.sh HEAD HEAD~
    ./get-changed-files.sh HEAD HEAD~ "\.py$"
EOF

if [ $# -lt 2 ] || [ $# -gt 3 ]
then
  echo "${USAGE}"
  exit 1
fi

revision_1="$1"
revision_2="$2"

# list files added and modified between revision_1 and revision_2. Do not list
# files that were deleted in revision_2.
changed_files="$(git diff --name-only --diff-filter=d "$revision_1" "$revision_2")"

if [ $# -eq 2 ]; then
  filtered_changed_files="$changed_files"
else
  filter_pattern="$3"
  # grep exits with 1 if there are no matches and 2 in case of errors. We
  # accept anything instead.
  # TODO: we should really only accept 0 and 1 and reject anything else.
  filtered_changed_files="$(grep --extended-regexp "$filter_pattern" <<< "$changed_files" || true)"
fi

if [ ! -z "${filtered_changed_files}" ];
then
  echo "${filtered_changed_files}"
fi
