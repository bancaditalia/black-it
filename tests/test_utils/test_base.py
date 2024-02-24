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
"""This module contains tests for the utils.base module."""
import json

import hypothesis.extra.numpy
import numpy as np
import pytest
from hypothesis import given
from numpy.typing import NDArray

from black_it.utils.base import NumpyArrayEncoder, _assert, digitize_data, is_symmetric


def test_assert_default() -> None:
    """Test the 'assert_' function, default exception."""
    message = "this is a message error"
    with pytest.raises(Exception, match=message):
        _assert(False, message)


def test_assert_custom_exception() -> None:
    """Test the 'assert_' function, with a specific exception."""
    message = "this is a message error"
    with pytest.raises(ValueError, match=message):
        _assert(False, message, exception_class=ValueError)

    class CustomError(Exception):
        """Custom exception."""

    with pytest.raises(CustomError, match=message):
        _assert(False, message, exception_class=CustomError)


def test_digitize_data() -> None:
    """Test the digitize_data function."""
    # a grid specifying 4 dimensions with integers from 1 to 3
    param_grid = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
    ]

    # an array containing 2 parameters of 4 dimensions
    input_array = np.array([[1.1, 2.0, 3.04, 4.5], [0.9, 1.9, 2.8, 3.5]])

    output_array = digitize_data(input_array, param_grid)

    expected_output = np.array([[1.0, 2.0, 3.0, 3.0], [1.0, 2.0, 3.0, 3.0]])

    assert output_array == pytest.approx(expected_output)


def test_is_symmetric(rng: np.random.Generator) -> None:
    """Test the 'is_symmetric' function."""
    assert is_symmetric(np.zeros((0, 0)))
    assert is_symmetric(np.zeros((4, 4)))
    assert is_symmetric(np.ones((4, 4)))

    assert not is_symmetric(rng.random(size=(4, 4)))


@given(
    hypothesis.extra.numpy.arrays(
        np.float64,
        hypothesis.extra.numpy.array_shapes(max_dims=3, max_side=3),
    ),
)
def test_numpy_array_json_encoder_for_any_numpy_array(array: NDArray) -> None:
    """Test JSONEncoder 'NumpyArrayEncoder' for any NumPy array."""
    serialized = json.dumps(array, cls=NumpyArrayEncoder)
    deserialized = json.loads(serialized)
    assert np.array_equal(array, deserialized, equal_nan=True)
