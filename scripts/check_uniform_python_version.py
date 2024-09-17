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

"""Ensure that the python versions indicated in setup.cfg and .ruff.toml are the same."""

import configparser
import logging
import pathlib
import platform
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

try:
    import tomllib  # type: ignore[import-not-found]

    logger.warning(
        "tomllib found in the stdlib, using that. Since you are running python %s "
        "maybe you could get rid of the tomli shim in this file and in pyproject.toml?",
        platform.python_version(),
    )
except ImportError:
    # please remove this and the whole tomli directory once the project starts
    # supporting a minimal python version of 3.11.
    import tomli as tomllib

    logger.info(
        "Using tomli as shim for tomllib because you are using python %s.",
        platform.python_version(),
    )


MY_DIR = pathlib.Path(__file__).parent.resolve()


def get_setup_cfg_py_version() -> str:
    """Read setup.cfg. Get the value of the mypy.python_version variable.

    The value is expected to be of the form: "3.9", "3.11", etc.
    """
    setup_cfg_path = (MY_DIR / ".." / "setup.cfg").resolve(strict=True)
    setup_cfg = configparser.ConfigParser()
    setup_cfg.read(setup_cfg_path)
    # expected to be of the form: "3.9"
    return setup_cfg["mypy"]["python_version"]


def get_ruff_toml_py_version() -> str:
    """Read .ruff.toml. Get the value of the target-version variable.

    The value is expected to be of the form: "py39", "py311", etc.
    """
    ruff_toml_path = (MY_DIR / ".." / ".ruff.toml").resolve(strict=True)
    with ruff_toml_path.open("rb") as f:
        ruff_toml = tomllib.load(f)
    # expected to be of the form: "py39"
    return ruff_toml["target-version"]


def main() -> int:
    """Main routine."""
    setup_cfg_python_version = get_setup_cfg_py_version()
    ruff_toml_target_version = get_ruff_toml_py_version()
    logger.debug("ruff_toml: %s", ruff_toml_target_version)  # 'py39'
    logger.debug("setup_cfg: %s", setup_cfg_python_version)  # '3.9'
    setup_cfg_converted = f"py{setup_cfg_python_version.replace('.', '')}"
    if setup_cfg_converted != ruff_toml_target_version:
        logger.error(
            "ERROR: the detected version in setup.cfg (%s) is different than the one in ruff.toml (%s)",
            setup_cfg_python_version,
            ruff_toml_target_version,
        )
        return 1
    logger.info(
        "OK: the versions are the same. setup_cfg: %s, ruff.toml: %s",
        setup_cfg_python_version,
        ruff_toml_target_version,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
