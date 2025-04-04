#!/usr/bin/env python

import configparser
import pathlib
import platform
import sys

try:
    import tomllib

    print(
        f"tomllib found in the stdlib, using that. Since you are running "
        f"python {platform.python_version()} maybe you could get rid of the "
        f"tomli shim in this file?",
    )
except ImportError:
    # please remove this and the whole tomli directory once the project starts
    # supporting a minimal python version of 3.11.
    import tomli as tomllib

    print(
        f"Using tomli as shim for tomllib because you are using python "
        f"{platform.python_version()}.",
    )

MY_DIR = pathlib.Path(__file__).parent.resolve()


def get_setup_cfg_py_version() -> str:
    setup_cfg_path = (MY_DIR / ".." / "setup.cfg").resolve()
    setup_cfg = configparser.ConfigParser()
    setup_cfg.read(setup_cfg_path)
    # expected to be of the form: "3.9"
    return setup_cfg["mypy"]["python_version"]


def get_ruff_toml_py_version() -> str:
    ruff_toml_path = (MY_DIR / ".." / ".ruff.toml").resolve()
    with ruff_toml_path.open("rb") as f:
        ruff_toml = tomllib.load(f)
    # expected to be of the form: "py39"
    return ruff_toml["target-version"]


setup_cfg_python_version = get_setup_cfg_py_version()
ruff_toml_target_version = get_ruff_toml_py_version()

print(f"{ruff_toml_target_version=}")  # 'py39'
print(f"{setup_cfg_python_version=}")  # '3.9'

converted = f"py{setup_cfg_python_version.replace('.', '')}"
if converted != ruff_toml_target_version:
    print(f"ERROR {converted=}")
    sys.exit(1)

print("OK: the versions are the same")
