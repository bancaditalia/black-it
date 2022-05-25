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

"""This module contains tests for the black_it.SearchSpace class."""

from typing import List

import numpy as np
import pytest

from black_it.search_space import (
    BadPrecisionLengthError,
    BoundsNotOfSizeTwoError,
    BoundsOfDifferentLengthError,
    LowerBoundGreaterThanUpperBoundError,
    PrecisionGreaterThanBoundsRangeError,
    PrecisionZeroError,
    SameLowerAndUpperBoundError,
    SearchSpace,
)


@pytest.mark.parametrize(
    "lower,upper,precision,expected_num_params,expected_cardinality",
    [
        pytest.param([0], [1], [0.5], 1, 3, id="single_param"),
        pytest.param([0], [100], [1], 1, 101, id="show_that_size_is_off_by_one"),
        pytest.param([0, 0, 0], [10, 20, 30], [0.1, 1, 5], 3, 14847, id="three_params"),
    ],
)
def test_search_space_successful(
    lower: List[float],
    upper: List[float],
    precision: List[float],
    expected_num_params: int,
    expected_cardinality: int,
) -> None:
    """Test successful construction of the SearchSpace object."""
    search_space = SearchSpace(np.array([lower, upper]), np.array(precision), True)

    assert search_space.dims == expected_num_params
    assert search_space.space_size == expected_cardinality


def test_search_space_fails_bounds_not_of_size_two() -> None:
    """Failed construction of SearchSpace due to BoundsNotOfSizeTwoError."""
    bounds = [[], [], [3]]
    precision = [0.1, 1, 0.1]

    with pytest.raises(BoundsNotOfSizeTwoError) as exc_info:
        _ = SearchSpace(np.array(bounds, dtype=object), np.array(precision), True)
    assert exc_info.value.count_bounds_subarrays == len(bounds)


def test_search_space_fails_bounds_of_different_length() -> None:
    """Failed construction of SearchSpace due to BoundsOfDifferentLengthError."""
    lower = [0, 0]
    upper = [10]
    precision = [0.1, 1]

    with pytest.raises(BoundsOfDifferentLengthError) as exc_info:
        _ = SearchSpace(
            np.array([lower, upper], dtype=object), np.array(precision), True
        )
    assert exc_info.value.lower_bounds_length == len(lower)
    assert exc_info.value.upper_bounds_length == len(upper)


def test_search_space_fails_bad_precision_length() -> None:
    """Failed construction of SearchSpace due to BadPrecisionLengthError."""
    lower = [0, 0]
    upper = [10, 0]
    precision = [0.1]

    with pytest.raises(BadPrecisionLengthError) as exc_info:
        _ = SearchSpace(np.array([lower, upper]), np.array(precision), True)
    assert exc_info.value.bounds_length == len(lower)
    assert exc_info.value.precisions_length == len(precision)


def test_search_space_fails_same_lower_and_upper_bound() -> None:
    """Failed construction of SearchSpace due to SameLowerAndUpperBoundError."""
    lower = [0, 5]
    upper = [10, 5]
    precision = [0.1, 0.1]

    with pytest.raises(SameLowerAndUpperBoundError) as exc_info:
        _ = SearchSpace(np.array([lower, upper]), np.array(precision), True)
    assert exc_info.value.param_index == 1
    assert exc_info.value.bound_value == lower[1]


def test_search_space_fails_lower_bound_greater_than_upper_bound() -> None:
    """Failed construction of SearchSpace due to LowerBoundGreaterThanUpperBoundError."""
    lower = [0, 15]
    upper = [10, 2]
    precision = [0.1, 0.1]

    with pytest.raises(LowerBoundGreaterThanUpperBoundError) as exc_info:
        _ = SearchSpace(np.array([lower, upper]), np.array(precision), True)
    assert exc_info.value.param_index == 1
    assert exc_info.value.lower_bound == lower[1]
    assert exc_info.value.upper_bound == upper[1]


def test_search_space_fails_precision_cant_be_zero() -> None:
    """Failed construction of SearchSpace due to PrecisionZeroError."""
    lower = [0, 0]
    upper = [10, 10]
    precision = [1, 0]

    with pytest.raises(PrecisionZeroError) as exc_info:
        _ = SearchSpace(np.array([lower, upper]), np.array(precision), True)
    assert exc_info.value.param_index == 1


def test_search_space_fails_precision_greater_than_bounds_range() -> None:
    """Failed construction of SearchSpace due to PrecisionGreaterThanBoundsRangeError."""
    lower = [0]
    upper = [10]
    precision = [10.1]

    with pytest.raises(PrecisionGreaterThanBoundsRangeError) as exc_info:
        _ = SearchSpace(np.array([lower, upper]), np.array(precision), True)
    assert exc_info.value.param_index == 0
    assert exc_info.value.lower_bound == lower[0]
    assert exc_info.value.upper_bound == upper[0]
    assert exc_info.value.precision == precision[0]
