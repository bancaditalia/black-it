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

"""This test module contains tests for the plot_results module."""
from typing import Any
from unittest.mock import patch

import pytest

from black_it.plot.plot_results import (
    plot_convergence,
    plot_losses,
    plot_losses_interact,
    plot_losses_method_num,
    plot_sampling,
    plot_sampling_batch_nums,
    plot_sampling_interact,
)
from tests.conftest import EXAMPLE_SAVING_FOLDER, PLOT_DIR
from tests.test_plot.base import BasePlotResultsTest
from tests.utils.base import skip_on_windows


@skip_on_windows()
class TestPlotConvergence(
    BasePlotResultsTest
):  # pylint: disable=too-few-public-methods
    """Test the 'plot_convergence' function."""

    plotting_function = plot_convergence
    saving_folder = EXAMPLE_SAVING_FOLDER
    expected_image = PLOT_DIR / "plot_convergence-expected.png"
    tolerance = None


@skip_on_windows()
class TestPlotLosses(BasePlotResultsTest):  # pylint: disable=too-few-public-methods
    """Test the 'plot_losses' function."""

    plotting_function = plot_losses
    saving_folder = EXAMPLE_SAVING_FOLDER
    expected_image = PLOT_DIR / "plot_losses-expected.png"


@skip_on_windows()
class TestPlotSampling(BasePlotResultsTest):  # pylint: disable=too-few-public-methods
    """Test the 'plot_sampling' function."""

    plotting_function = plot_sampling
    saving_folder = EXAMPLE_SAVING_FOLDER
    expected_image = PLOT_DIR / "plot_sampling-expected.png"


@skip_on_windows()
class TestPlotLossesMethodNum(
    BasePlotResultsTest
):  # pylint: disable=too-few-public-methods
    """Test the 'plot_losses_method_num' function."""

    plotting_function = plot_losses_method_num
    saving_folder = EXAMPLE_SAVING_FOLDER

    @pytest.mark.skip(
        "for this test case, we use the parametrized method 'test_run_by_method_num'"
    )
    def test_run(self) -> None:
        """Run the test."""

    @pytest.mark.parametrize("method_num", list(range(5)))
    def test_run_by_method_num(
        self, method_num: int
    ) -> None:  # pylint: disable=arguments-differ
        """Run the test for all method numbers."""
        self.expected_image = (
            PLOT_DIR / f"plot_losses_method_num_{method_num}-expected.png"
        )
        self.args = [self.saving_folder, method_num]
        super().run()

    def teardown(self) -> None:
        """Tear down the test."""
        # restore default attributes
        delattr(self, "expected_image")
        self.args = []


@skip_on_windows()
class TestPlotBatchNums(BasePlotResultsTest):  # pylint: disable=too-few-public-methods
    """Test the 'plot_sampling_batch_nums' function."""

    plotting_function = plot_sampling_batch_nums
    saving_folder = EXAMPLE_SAVING_FOLDER

    @pytest.mark.skip(
        "for this test case, we use the parametrized method 'test_run_by_batch_num'"
    )
    def test_run(self) -> None:
        """Run the test."""

    @pytest.mark.parametrize("batch_num", list(range(0, 13, 3)))
    def test_run_by_batch_num(
        self, batch_num: int
    ) -> None:  # pylint: disable=arguments-differ
        """Run the test for all method numbers."""
        self.expected_image = (
            PLOT_DIR / f"plot_sampling_batch_nums_{batch_num:03d}-expected.png"
        )
        # gather batches up to batch_num
        self.args = [self.saving_folder, list(range(batch_num))]
        super().run()

    def teardown(self) -> None:
        """Tear down the test."""
        # restore default attributes
        delattr(self, "expected_image")
        self.args = []


@patch("pandas.read_csv", return_value={"method_samp": set()})
def test_plot_losses_method_num_raises_error_if_method_num_not_known(
    *_mocks: Any,
) -> None:
    """Test that 'plot_losses_method_num' raises error if the method num is not known."""
    method_num = 1
    with pytest.raises(
        ValueError, match=f"Samplers with method_num = {method_num} was never used"
    ):
        plot_losses_method_num("dummy_folder", method_num)


@patch("ipywidgets.widgets.interaction.interact")
def test_plot_losses_interact(*_mocks: Any) -> None:
    """
    Test 'plot_losses_interact' function.

    Note that this function does not test the interaction with the plots,
    as the 'ipywidgets.widgets.interaction.interact' function is mocked.

    Args:
        *_mocks: mocked items.
    """
    plot_losses_interact(EXAMPLE_SAVING_FOLDER)


@patch("ipywidgets.widgets.interaction.interact")
def test_plot_sampling_interact(*_mocks: Any) -> None:  # noqa
    """
    Test 'plot_sampling_interact' function.

    Note that this function does not test the interaction with the plots,
    as the 'ipywidgets.widgets.interaction.interact' function is mocked.

    Args:
        *_mocks: mocked items.
    """
    plot_sampling_interact(EXAMPLE_SAVING_FOLDER)
