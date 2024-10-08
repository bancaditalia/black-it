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

"""Test the main example."""
from __future__ import annotations

import sys

from tests.conftest import DEFAULT_SUBPROCESS_TIMEOUT, ROOT_DIR
from tests.test_examples.base import BaseMainExampleTestClass

EXAMPLE_MAIN_SCRIPT_PATH = ROOT_DIR / "examples" / "main.py"


TRUE_PARAMETERS_STR = "True parameters:       [0.2, 0.2, 0.75]"
BEST_PARAMETERS_STR = "Best parameters found: [0.19 0.21 0.68]"


class TestMainExample(BaseMainExampleTestClass):
    """Test that example main can be run successfully."""

    script_path = EXAMPLE_MAIN_SCRIPT_PATH
    timeout = DEFAULT_SUBPROCESS_TIMEOUT
    nb_batches = 5
    if sys.platform == "darwin":
        # 2024-09-10: MethodOfMomentsLoss started giving slightly different
        # values on MacOS when statsmodels was updated from 0.13 to 0.14. This
        # captures the change, so that when a fix arrives upstream we can update
        # the test again.
        BEST_PARAMETERS_STR = "Best parameters found: [0.21 0.19 0.76]"

    expected_lines: tuple[str, ...] = (
        "PARAMS SAMPLED: 0",
        "METHOD: HaltonSampler",
        "METHOD: RandomForestSampler",
        "METHOD: BestBatchSampler",
        *[f"BATCH NUMBER:   {i}" for i in range(1, nb_batches + 1)],
        TRUE_PARAMETERS_STR,
        BEST_PARAMETERS_STR,
    )
