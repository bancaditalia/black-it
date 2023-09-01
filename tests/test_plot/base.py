# Black-box ABM Calibration Kit (Black-it)
# Copyright (C) 2021-2023 Banca d'Italia
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

"""This module contains utilities for the test_plot package."""
from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images


class BasePlotTest:
    """Base test class for plotting functions."""

    plotting_function: Callable
    expected_image: Path
    tolerance: float | None = 0.000
    args: Sequence[Any] = ()

    def test_run(self) -> None:
        """Run the test."""
        self.run()

    def run(self) -> None:
        """Run the plotting and the image comparison."""
        logging.getLogger("matplotlib.font_manager").disabled = True
        plt.close()
        self.plotting_function.__func__(*self.args)  # type: ignore[attr-defined]
        with TemporaryDirectory() as tmpdir:
            actual_figure_path = Path(tmpdir) / "actual.png"
            plt.savefig(actual_figure_path)

            if self.tolerance is None:
                logging.warning("Test run with tolerance=None, skipping the test")
                return

            comparison_result = compare_images(
                str(self.expected_image),
                str(actual_figure_path),
                self.tolerance,
            )
            if comparison_result is not None:
                logging.warning("%s", comparison_result)


class BasePlotResultsTest(BasePlotTest):
    """Base test class for 'plot_results' tests."""

    saving_folder: Path

    @classmethod
    def setup(cls) -> None:
        """Set up the test."""
        cls.args = [cls.saving_folder, *cls.args]
