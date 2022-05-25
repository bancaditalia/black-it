# Black-box ABM Calibration Kit (Black-it)
# Copyright (C) 2021-2022 Banca d'Italia
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

"""This test module contains tests for the plot_descriptive_statistics.py module."""
from typing import Any, List

import numpy as np

from black_it.plot.plot_descriptive_statistics import ts_stats
from tests.conftest import PLOT_DIR
from tests.test_plot.base import BasePlotTest


class TestTsStats(BasePlotTest):
    """Test 'ts_stats' plotting function."""

    plotting_function = ts_stats
    args: List[Any] = []
    expected_image = PLOT_DIR / "ts_stats-expected.png"

    def setup(self) -> None:
        """Set up the test."""
        np.random.seed(42)
        data = np.random.rand(100)
        self.args = [data]
