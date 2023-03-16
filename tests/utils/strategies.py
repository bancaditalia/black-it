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

"""Hypothesis strategies for testing."""
from typing import Callable, Tuple  # pylint: disable=unused-import

import hypothesis.extra.numpy
import hypothesis.strategies as st
from hypothesis import assume
from hypothesis.extra.numpy import floating_dtypes
from hypothesis.strategies import composite, floats, integers

MAX_TIME_SERIES_LENGTH = 100
MAX_NB_VALUES = 10
MAX_WORD_LENGTH = 5


def time_series(max_length: int = MAX_TIME_SERIES_LENGTH) -> st.SearchStrategy:
    """
    Return a strategy to generate time_series.

    Args:
        max_length: the max length of the time series.

    Returns:
        the time series strategy
    """
    return hypothesis.extra.numpy.arrays(
        dtype=floating_dtypes(),
        shape=integers(min_value=0, max_value=max_length),
    )


@composite
def discretize_args(
    draw: Callable,
    max_length: int = MAX_TIME_SERIES_LENGTH,
    max_nb_values: int = MAX_NB_VALUES,
) -> Tuple:
    """
    Return the strategies for the arguments of the 'discretize' function.

    Args:
        draw: the Hypothesis draw function
        max_length: the maximum length
        max_nb_values: the maximum number of values

    Returns:
        a tuple of Hypothesis.strategies
    """
    time_series_ = draw(time_series(max_length))
    nb_values = draw(integers(min_value=0, max_value=max_nb_values))
    start_index = draw(floats())
    stop_index = draw(floats())
    return time_series_, nb_values, start_index, stop_index


@composite
def get_words_args(
    draw: Callable,
    max_length: int = MAX_TIME_SERIES_LENGTH,
    max_word_length: int = MAX_WORD_LENGTH,
) -> Tuple:
    """
    Return the strategies for the arguments of the 'discretize' function.

    Args:
        draw: the Hypothesis draw function
        max_length: the maximum length
        max_word_length: the maximum word length

    Returns:
        a pair (time series, word length)
    """
    time_series_ = draw(time_series(max_length))
    word_length = draw(integers(min_value=0, max_value=max_word_length))
    assume(len(time_series_) + 1 > word_length)
    return time_series_, word_length
