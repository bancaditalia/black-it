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

"""Conftest file."""
import inspect
from pathlib import Path

import numpy as np
import pytest

CUR_PATH = Path(inspect.getfile(inspect.currentframe())).parent  # type: ignore[arg-type]
ROOT_DIR = Path(CUR_PATH, "..").resolve().absolute()
DOCS_DIR = ROOT_DIR / "docs"
TEST_DIR = ROOT_DIR / "tests"
FIXTURES_DIR = TEST_DIR / "fixtures"
PLOT_DIR = FIXTURES_DIR / "plots"
EXAMPLE_SAVING_FOLDER = ROOT_DIR / "examples" / "saving_folder"

DEFAULT_SUBPROCESS_TIMEOUT = 100.0


@pytest.fixture
def rng() -> np.random.Generator:
    """Return random number generator."""
    return np.random.default_rng(seed=11)
