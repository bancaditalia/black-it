import sys
import platform
import configparser

try:
    import tomllib
except ImportError:
    print(f"Could not find tommlib, and this is Python {platform.python_version()}. Please run with python >= 3.11")
    sys.exit(1)

setup_cfg = configparser.ConfigParser()
setup_cfg.read("setup.cfg")
setup_cfg_python_version = setup_cfg["mypy"]["python_version"]
with open(".ruff.toml", "rb") as f:
    ruff_toml = tomllib.load(f)
ruff_toml_target_version = ruff_toml["target-version"]
print(f"{ruff_toml_target_version=}") # 'py39'
print(f"{setup_cfg_python_version=}") # '3.9'

converted = f"py{setup_cfg_python_version.replace('.', '')}"
if converted != ruff_toml_target_version:
    print(f"ERROR {converted=}")
    exit(1)

print("OK: the versions are the same")
