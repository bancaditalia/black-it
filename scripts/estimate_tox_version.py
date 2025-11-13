#!/usr/bin/env python3
# Black-box ABM Calibration Kit (Black-it)
# Copyright (C) 2021-2024 Banca d'Italia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
r"""Read the output of `poetry export --only=dev --without-hashes` and return the tox version.

The export format is similar to:
    tox==4.18.1 ; python_version >= "3.9" and python_version < "3.13"
    ....

The following two commands should be equivalent:
    poetry export --only=dev --without-hashes | ./estimate_tox_version.py
    poetry export --only=dev --without-hashes | grep tox | head -n 1 | sed --regexp-extended 's/^tox==(.*) *;.*$/\1/g'
"""

from __future__ import annotations

import logging
import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


logger = logging.getLogger(__name__)


def main(in_stream: Iterable[str]) -> str | None:
    """Read through the input iterable, looking for a tox specifier, and return its version."""
    version = None
    count_matches = 0
    for line in in_stream:
        match = re.match(
            r"^tox==(?P<version>[0-9\.]+)\s*;.*$",
            line,
        )
        if match is None:
            continue
        if count_matches > 0:
            msg = "More than one match found. This is not acceptable."
            raise ValueError(msg)
        count_matches += 1
        version = match.group("version")
        if version is None:
            msg = "version is None: this should never happen"
            raise ValueError(msg)
    if version is None:
        msg = "could not find tox version"
        raise ValueError(msg)
    return version


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    msg = (
        f"this program is intended to be invoked like this: `poetry export --only=dev --without-hashes | {sys.argv[0]}`"
    )
    logger.info(msg)
    logger.info("reading from stdin...")
    try:
        print(main(sys.stdin))
    except ValueError as e:
        logger.error(e)  # noqa: TRY400
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        sys.exit(2)
