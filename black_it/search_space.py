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

"""This module contains the definition of the search space abstractions."""

from typing import List

import numpy as np
from numpy.typing import NDArray


class SearchSpace:  # pylint: disable=too-few-public-methods
    """A class that contains information on the search grid of explorable parameters."""

    def __init__(
        self,
        parameters_bounds: NDArray[np.float64],
        parameters_precision: NDArray[np.float64],
        verbose: bool,
    ):
        """
        Initialize the SearchSpace object.

        The values of parameters_bounds and parameters_precision parameters have
        to satisfy the following constraints, otherwise an exception (subclass
        of SearchSpaceError) is raised:

        - parameters_bounds must be a two-elements array (or
          BoundsNotOfSizeTwoError is raised)
        - the two sub-arrays of parameter_bounds must have the same length (or
          BoundsOfDifferentLengthError is raised)
        - parameters_precision array must have the same number of elements than
          the two sub-arrays in parameters_bounds (or BadPrecisionLengthError is
          raised)
        - lower bounds and upper bounds cannot have the same value (or
          SameLowerAndUpperBoundError is raised)
        - every lower bound must be lower than the corresponding upper bound (or
          LowerBoundGreaterThanUpperBoundError)
        - 0 is an invalid value for a precision (or PrecisionZeroError is
          raised)
        - any given parameter precision has to be strictly lower than the
          allowed parameter span (or PrecisionGreaterThanBoundsRangeError is
          raised)

        Args:
            parameters_bounds: lower and upper bounds of the parameters.
            parameters_precision: resolution of the grid of parameters.
            verbose: whether to print or not the information on the search space.
        """
        SearchSpace._check_bounds(parameters_bounds, parameters_precision)

        # The bounds we were given are well formed. Save them.
        self._parameters_bounds = parameters_bounds
        self._parameters_precision = parameters_precision

        # Initialize search grid
        self._param_grid: List[NDArray[np.float64]] = []
        self._space_size = 1
        for i in range(self.dims):
            new_col = np.arange(
                parameters_bounds[0][i],
                parameters_bounds[1][i] + 0.0000001,
                parameters_precision[i],
            )
            new_col = np.round(new_col, -int(np.log10(parameters_precision[i])))
            self._param_grid.append(new_col)
            self._space_size *= len(new_col)

        if verbose:
            print("\n***")
            print(f"Number of free params:       {self.dims}.")
            print(f"Explorable param space size: {self.space_size}.")
            print("***\n")

    @staticmethod
    def _check_bounds(
        parameters_bounds: NDArray[np.float64],
        parameters_precision: NDArray[np.float64],
    ) -> None:
        """
        Ensure parameter_bounds and parameter_precision have acceptable values.

        This is an helper function for SearchSpace.__init__().

        Args:
            parameters_bounds: lower and upper bounds of the parameters
            parameters_precision: resolution of the grid of parameters
        """
        # ensure parameters_bounds is a two-elements array
        if len(parameters_bounds) != 2:
            raise BoundsNotOfSizeTwoError(len(parameters_bounds))

        # ensure the two sub-arrays of parameter_bounds (which are the min and
        # max bounds for each parameter) have the same length
        if len(parameters_bounds[0]) != len(parameters_bounds[1]):
            raise BoundsOfDifferentLengthError(
                len(parameters_bounds[0]),
                len(parameters_bounds[1]),
            )

        # ensure parameters_precision array has as many elements as either one
        # of parameters_bounds sub-array
        if len(parameters_precision) != len(parameters_bounds[0]):
            raise BadPrecisionLengthError(
                len(parameters_precision),
                len(parameters_bounds[0]),
            )

        for i, (lower_bound, upper_bound, precision) in enumerate(
            zip(parameters_bounds[0], parameters_bounds[1], parameters_precision)
        ):
            # ensure lower bounds and upper bounds do not have the same value
            if lower_bound == upper_bound:
                raise SameLowerAndUpperBoundError(i, lower_bound)

            # ensure each lower bound is lower than the corresponding upper one
            if lower_bound > upper_bound:
                raise LowerBoundGreaterThanUpperBoundError(i, lower_bound, upper_bound)

            # a precision of 0 is not meaningful
            if precision == 0:
                raise PrecisionZeroError(i)

            # any given parameter precision has to be strictly lower than the
            # allowed parameter span
            if precision > (upper_bound - lower_bound):
                raise PrecisionGreaterThanBoundsRangeError(
                    i, lower_bound, upper_bound, precision
                )

    @property
    def param_grid(self) -> List[NDArray[np.float64]]:
        """Discretized parameter space containing all possible candidates for calibration."""
        return self._param_grid

    @property
    def parameters_bounds(self) -> NDArray[np.float64]:
        """Two dimensional array containing lower and upper bounds for each parameter."""
        return self._parameters_bounds

    @property
    def parameters_precision(self) -> NDArray[np.float64]:
        """One dimensional array containing the precisions for each parameter."""
        return self._parameters_precision

    @property
    def dims(self) -> int:
        """Return the number of model parameters configured for calibration."""
        return len(self._parameters_precision)

    @property
    def space_size(self) -> int:
        """Cardinality of the potential parameter space to be searched."""
        return self._space_size


class SearchSpaceError(ValueError):
    """Base class for the exceptions raised by SearchSpace when its construction fails.

    If you need to distinguish a specific error, please do not rely on parsing
    the error message, because it is considered unstable: catch instead the
    specific exception type you want to handle, and use the additional fields
    each subtype exposes (for example: BadPrecisionLengthError.bounds_length)
    """


class BoundsNotOfSizeTwoError(SearchSpaceError):
    """Raised when bounds are not a two-dimensional array.

    Attributes:
        count_bounds_subarrays (int): wrong number of subarrays the
            parameter_bounds array was made of. It will be different than 2.
    """

    def __init__(self, count_bounds_subarrays: int) -> None:  # noqa: D107
        super().__init__(
            f"parameters_bounds must be a two dimensional array. This one has "
            f"size {count_bounds_subarrays}."
        )
        self.count_bounds_subarrays = count_bounds_subarrays


class BoundsOfDifferentLengthError(SearchSpaceError):
    """Raised when the lower and upper bounds do not have the same number of elements.

    Attributes:
        lower_bounds_length (int): number of elements in the subarray 0. It will
            be different than upper_bounds_length
        upper_bounds_length (int): number of elements in the subarray 1. It will
            be different than lower_bounds_length
    """

    def __init__(  # noqa: D107
        self, lower_bounds_length: int, upper_bounds_length: int
    ) -> None:
        super().__init__(
            f"parameters_bounds subarrays must be of the same length. Lower "
            f"bounds length: {lower_bounds_length}, upper bounds length: "
            f"{upper_bounds_length}."
        )
        self.lower_bounds_length = lower_bounds_length
        self.upper_bounds_length = upper_bounds_length


class BadPrecisionLengthError(SearchSpaceError):
    """Raised when the parameters_precision array has a different length than the bounds'.

    Attributes:
        precisions_length (int): number of elements of the precision subarray
        bounds_length (int): number of elements of the two bounds subbarrays
    """

    def __init__(  # noqa: D107
        self, precisions_length: int, bounds_length: int
    ) -> None:
        super().__init__(
            f"parameters_precision array has {precisions_length} elements. Its "
            f"length, instead, has to be {bounds_length}, the same as the bounds'."
        )
        self.precisions_length = precisions_length
        self.bounds_length = bounds_length


class SameLowerAndUpperBoundError(SearchSpaceError):
    """Raised when the lower and the upper bound of a parameter have the same value.

    Attributes:
        param_index (int): 0-based index of the parameter presenting the error
        bound_value (float): common value of the parameter bound
    """

    def __init__(self, param_index: int, bound_value: float) -> None:  # noqa: D107
        super().__init__(
            f"Parameter {param_index}'s lower and upper bounds have the same "
            f"value ({bound_value}). This calibrator cannot handle that. "
            f"Please redefine externally your model in order to hardcode "
            f"parameter {param_index} to {bound_value}."
        )
        self.param_index = param_index
        self.bound_value = bound_value


class LowerBoundGreaterThanUpperBoundError(SearchSpaceError):
    """Raised when the lower bound of a parameter is greater than its upper bound.

    Attributes:
        param_index (int): 0-based index of the parameter presenting the error
        lower_bound (float): lower bound. It will be higher than upper bound
        upper_bound (float): upper bound. It will be lower than lower bound
    """

    def __init__(  # noqa: D107
        self, param_index: int, lower_bound: float, upper_bound: float
    ) -> None:
        super().__init__(
            f"Parameter {param_index}'s lower bound ({lower_bound}) must be "
            f"lower than its upper bound ({upper_bound})."
        )
        self.param_index = param_index
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class PrecisionZeroError(SearchSpaceError):
    """Raised when a parameter precision is set to 0.

    Attributes:
        param_index (int): 0-based index of the parameter presenting the error
    """

    def __init__(self, param_index: int) -> None:  # noqa: D107
        super().__init__(f"Parameter {param_index}'s precision cannot be zero.")
        self.param_index = param_index


class PrecisionGreaterThanBoundsRangeError(SearchSpaceError):
    """Raised when the precision step is greater than the parameter range.

    Attributes:
        param_index (int): 0-based index of the parameter presenting the error
        lower_bound (float): lower bound
        upper_bound (float): upper bound
        precision (float): the illegal precision. It will be greater than
            (upper_bound - lower_bound)
    """

    def __init__(  # noqa: D107
        self, param_index: int, lower_bound: float, upper_bound: float, precision: float
    ) -> None:
        super().__init__(
            f"Parameter {param_index}'s allowed range is [{lower_bound}, "
            f"{upper_bound}]. The requested precision is {precision}, but this "
            f"number cannot be greater than the bounds range "
            f"({upper_bound}-{lower_bound}={upper_bound-lower_bound})."
        )
        self.param_index = param_index
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.precision = precision
