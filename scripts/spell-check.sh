#!/bin/bash
# This script requires `mdspell`:
#
#    https://www.npmjs.com/package/markdown-spellcheck
#
# Run this script from the root directory.
# Usage:
#   ./scripts/spell-check.sh
#

set -eu
shopt -s inherit_errexit

read -r -d '' USAGE <<'EOF' || true
USAGE:
    $(basename "$0") [-h|--help]
    $(basename "$0") [-i|--interactive]
    $(basename "$0") [-r|--report]

    Run the mdspell tool against the documentation files of the repository.

    The script is a wrapper of the mdspell tool. It is required that the mdspell tool is
    installed and accessible at the system path in order for this script to work.

    With the '-i' or '--interactive' arguments, the script runs in interactive mode:
    at each error, the mdspell tool asks the user to fix it.

    With the '-r' or '--report' arguments, the mdspell tool is run in 'report' mode; the tool
    only reports errors and it does not ask the user how to fix them. If an error is
    found, it returns with error code 1.

EXAMPLES:
    ./spell-check.sh -i  # runs in interactive mode
    ./spell-check.sh -r  # runs in report mode
    ./spell-check.sh -h  # prints the usage
EOF

MDSPELL_PATH="$(which mdspell || true)"
if [ -z "${MDSPELL_PATH}" ]; then
  echo "Cannot find executable 'mdspell'. Please install it to run this script: npm i markdown-spellcheck -g"
  exit 127
fi

echo "Found 'mdspell' executable at ${MDSPELL_PATH}"

mdspell_ignore_strings_args=('!docs/losses.md' '!docs/calibrator.md' '!docs/samplers.md' '!docs/search_space.md' '!HISTORY.md' '!docs/schedulers.md')
if [ $# == 0 ] || [ $# -gt 1 ] || { [ $# == 1 ] && { [ "$1" == "-h" ] || [ "$1" == "--help" ]; }; }; then
  echo "${USAGE}"
  exit 1
elif [  $# == 1 ] && { [ "$1" == "-i" ] || [ "$1" == "--interactive" ]; } then
  mdspell -n -a --en-gb '**/*.md' "${mdspell_ignore_strings_args[@]}"
elif [  $# == 1 ] && { [ "$1" == "-r" ] || [ "$1" == "--report" ]; } then
  mdspell -n -a --en-gb '**/*.md' "${mdspell_ignore_strings_args[@]}" --report
else
  echo "${USAGE}"
  exit 1
fi
