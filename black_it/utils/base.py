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

"""This module contains generic utility functions for the library."""
import os
from json import JSONEncoder
from typing import Any, List, Type, Union

import numpy as np
from numpy.typing import NDArray

PathLike = Union[str, os.PathLike]


def assert_(
    condition: bool, message: str, exc_cls: Type[Exception] = Exception
) -> None:
    """
    Enforce a condition.

    Args:
        condition: the boolean condition
        message: the exception message to raise
        exc_cls: the exception class
    """
    if not condition:
        raise exc_cls(message)


def check_arg(condition: bool, message: str) -> None:
    """
    Check a condition over an argument (i.e. raises ValueError in case it is false).

    Args:
        condition: the condition to check
        message: the error message
    """
    assert_(condition, message, exc_cls=ValueError)


def ensure_float(arg: Any) -> None:
    """Check that the argument is a float."""
    check_arg(isinstance(arg, (float, np.float64)), f"expected a float, got {arg}")


def positive_float(arg: float) -> float:
    """Check a float is positive."""
    ensure_float(arg)
    check_arg(arg >= 0, "provided float is not positive")
    return arg


def digitize_data(
    data: NDArray[np.float64], param_grid: List[NDArray[np.float64]]
) -> NDArray[np.float64]:
    """
    Return a discretized version of the input sorted_array.

    Args:
        data: the input array to be discretized
        param_grid: the discrete values allowed

    Returns:
        the discretized array
    """
    digitalized_data: NDArray[np.float64] = np.zeros(shape=data.shape)

    for i in range(data.shape[1]):
        digitalized_data[:, i] = get_closest(param_grid[i], data[:, i])

    return digitalized_data


def get_closest(sorted_array: NDArray, values: NDArray) -> NDArray:
    """Fast way to find the nearest element in 'sorted_array' for each element in 'values'.

    Solution taken from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array/46184652#46184652.

    Args:
        sorted_array: a sorted array
        values: an array of arbitrary values

    Returns:
        for each element in the 'values' array the closest element in 'sorted_array' is returned
    """
    # get insert positions
    idxs = np.searchsorted(sorted_array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = (idxs == len(sorted_array)) | (
        np.fabs(values - sorted_array[np.maximum(idxs - 1, 0)])
        < np.fabs(values - sorted_array[np.minimum(idxs, len(sorted_array) - 1)])
    )
    idxs[prev_idx_is_less] -= 1

    return sorted_array[idxs]


def is_symmetric(a: NDArray[np.float64]) -> bool:
    """
    Check if a matrix is symmetric.

    Args:
        a: the matrix

    Returns:
        True if the matrix is symmetric
    """
    return a.shape[0] == a.shape[1] and (a == a.T).all()


class NumpyArrayEncoder(JSONEncoder):
    """
    Custom JSONEncoder for Numpy arrays.

    Solution from https://pynative.com/python-serialize-numpy-ndarray-into-json/.
    """

    def default(self, o: Any) -> Any:
        """Implement custom JSON serialization for NumPy NDArrays."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        return JSONEncoder.default(self, o)  # pragma: no cover


def get_random_seed(random_generator: np.random.Generator) -> int:
    """
    Get a random seed from a random generator.

    Sample an integer in the range [0, 2^32 - 1].

    Args:
        random_generator: the random generator to be used for sampling the random seed.

    Returns:
        the random seed.
    """
    return random_generator.integers(2**32 - 1)
