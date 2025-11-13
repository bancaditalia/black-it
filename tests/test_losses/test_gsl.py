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

"""This module contains tests for the GSL-div loss."""

from __future__ import annotations

import numpy as np
from hypothesis import given

from black_it.loss_functions.gsl_div import GslDivLoss
from tests.utils.strategies import discretize_args, get_words_args


def is_sorted(array: np.ndarray, *, reverse: bool = False) -> bool:
    """Test that a unidimensional array is sorted.

    Args:
        array: the array
        reverse: if False, check never-decreasing; if True, check never-increasing.

    Returns:
        True if it is sorted, False otherwise.
    """
    assert len(array.shape) == 1

    def fail_check(left: float, right: float) -> bool:
        return left > right if not reverse else left < right

    return all(not fail_check(array[i], array[i + 1]) for i in range(array.size - 1))


class TestDiscretize:
    """Test the 'discretize' function."""

    @given(discretize_args())
    def test_discretize_time_series_any_args(  # noqa: PLR6301
        self,
        args: tuple,
    ) -> None:
        """Test the case with randomly generated args."""
        GslDivLoss.discretize(*args)

    @given(discretize_args())
    def test_discretize_time_series_partition(  # noqa: PLR6301
        self,
        args: tuple,
    ) -> None:
        """Test that discretize computes the right number of partitions."""
        time_series, nb_values, start_index, stop_index = args
        actual = GslDivLoss.discretize(time_series, nb_values, start_index, stop_index)
        max_nb_values = nb_values + 2
        assert len(np.unique(actual)) <= max_nb_values
        assert (actual >= 0).all()
        assert (actual <= max_nb_values).all()

    @given(discretize_args())
    def test_ordering_is_preserved(self, args: tuple) -> None:  # noqa: PLR6301
        """Test that the time series ordering is preserved when discretized."""
        time_series, nb_values, start_index, stop_index = args
        increasing_time_series = np.sort(time_series)
        increasing_actual = GslDivLoss.discretize(
            increasing_time_series,
            nb_values,
            start_index,
            stop_index,
        )
        assert is_sorted(increasing_actual, reverse=False)

        decreasing_time_series = increasing_time_series[::-1]
        decreasing_actual = GslDivLoss.discretize(
            decreasing_time_series,
            nb_values,
            start_index,
            stop_index,
        )
        assert is_sorted(decreasing_actual, reverse=True)


class TestGetWords:
    """Test the 'get_words' function."""

    @given(get_words_args())
    def test_get_words(self, args: tuple) -> None:  # noqa: PLR6301
        """Test the case with randomly generated args."""
        GslDivLoss.get_words(*args)


def test_gsl_default(rng: np.random.Generator) -> None:
    """Test the Gsl-div loss function."""
    expected_loss = 0.3972285978726733

    series_sim = rng.normal(0, 1, (100, 3))
    series_real = rng.normal(0, 1, (100, 3))

    loss_func = GslDivLoss()
    loss = loss_func.compute_loss(series_sim[None, :, :], series_real)

    assert np.isclose(expected_loss, loss)


def test_gsl_with_nb_values(rng: np.random.Generator) -> None:
    """Test the Gsl-div loss function with nb_values set."""
    expected_loss = 0.4354049587629579

    series_sim = rng.normal(0, 1, (2, 100, 3))
    series_real = rng.normal(0, 1, (100, 3))

    loss_func = GslDivLoss(nb_values=10)
    loss = loss_func.compute_loss(series_sim, series_real)

    assert np.isclose(expected_loss, loss)


def test_gsl_with_nb_word_lengths(rng: np.random.Generator) -> None:
    """Test the Gsl-div loss function with nb_word_lengths set."""
    expected_loss = 0.7177347914787273

    series_sim = rng.normal(0, 1, (100, 3))
    series_real = rng.normal(0, 1, (100, 3))

    loss_func = GslDivLoss(nb_word_lengths=10)
    loss = loss_func.compute_loss(series_sim[None, :, :], series_real)

    assert np.isclose(expected_loss, loss)
