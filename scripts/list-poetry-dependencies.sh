#!/usr/bin/env bash

set -eu
shopt -s inherit_errexit

__usage="USAGE:
    $(basename "$0") [-h|--help]
    $(basename "$0") --all
    $(basename "$0") <dep-name> [...]

With the --all argument, prints to stdout the contents of the whole poetry.lock
file in pip's requirements.txt format (equivalent of
'poetry export --without-hashes --dev').

With args, considers each argument as a search pattern to select requirements.

EXAMPLES:
  ./$(basename "$0") -h           # prints the usage
  ./$(basename "$0") --all        # exports all the packages from poetry.lock
  ./$(basename "$0") flake8 numpy # exports the packages needed by flake8 and numpy
"

if [ $# == 0 ] || ([ $# == 1 ] && ([ "$1" == "-h" ] || [ "$1" == "--help" ]));
then
    echo "${__usage}"
    exit 1
fi

all_frozen_dependencies="$(poetry export --without-hashes --dev)"
if [ "$1" == "--all" ]; then
    dependencies_to_install="${all_frozen_dependencies}"
else
    pattern="$(echo "$@" |tr " " "|")"
    dependencies_to_install="$(echo "${all_frozen_dependencies}" | grep --extended-regexp "${pattern}")"
fi

echo "${dependencies_to_install}"
