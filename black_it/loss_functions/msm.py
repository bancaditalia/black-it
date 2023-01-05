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

"""
Method-of-moments loss function.

This module contains the implementation of the loss function
based on the 'method of moments'.
"""
from enum import Enum
from typing import Callable, List, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray

from black_it.loss_functions.base import BaseLoss
from black_it.utils.base import is_symmetric
from black_it.utils.time_series import get_mom_ts_1d

MomentCalculator = Callable[[NDArray], NDArray]
"""
Type of a custom moment calculator: a function that takes an NDArray and returns
an NDArray.
"""


class _CovarianceMatrixType(Enum):
    """Enumeration of allowed covariance matrix types."""

    IDENTITY = "identity"
    INVERSE_VARIANCE = "inverse_variance"

    def __str__(self) -> str:
        """Get the string representation."""
        return self.value


class MethodOfMomentsLoss(BaseLoss):
    """Class for the 'method of moments' loss."""

    def __init__(
        self,
        covariance_mat: Union[str, NDArray[np.float64]] = "identity",
        coordinate_weights: Optional[NDArray[np.float64]] = None,
        moment_calculator: MomentCalculator = get_mom_ts_1d,
        coordinate_filters: Optional[List[Optional[Callable]]] = None,
        standardise_moments: bool = False,
    ):
        """
        Initialize the loss function based on the 'method of moments'.

        Returns the MSM objective function, i.e. the square difference between
        the moments of the two time series.
        By default the loss computes the moments using
        black_it.utils.time_series.get_mom_ts_1d(), which computes an
        18-dimensional vector of statistics.

        You can alter the behaviour passing a custom function to
        moment_calculator. Please note that there is a constraint between the
        moment calculator and the size of the covariance matrix.

        Args:
            covariance_mat: covariance matrix between the moments.
                'identity' (the default) gives the identity matrix, 'inverse_variance' gives the diagonal
                matrix of the estimated inverse variances. Alternatively, one can specify a numerical symmetric matrix
                whose size must be equal to the number of elements that the moment calculator returns.
            coordinate_weights: importance of each coordinate. By default all
                coordinates are treated equally.
            moment_calculator: a function that takes a 1D time series and
                returns a series of moments. The default is
                black_it.utils.time_series.get_mom_ts_1d()
            coordinate_filters: filters/transformations to be applied to each simulated series before
                the loss computation.
            standardise_moments: if True all moments are divided by the absolute value of the real moments,
                default is False.
        """
        MethodOfMomentsLoss._validate_covariance_and_calculator(
            moment_calculator, covariance_mat
        )

        super().__init__(coordinate_weights, coordinate_filters)
        self._covariance_mat = covariance_mat
        self._moment_calculator = moment_calculator
        self._standardise_moments = standardise_moments

    @staticmethod
    def _validate_covariance_and_calculator(
        moment_calculator: MomentCalculator,
        covariance_mat: Union[NDArray[np.float64], str],
    ) -> None:
        """
        Validate the covariance matrix.

        Args:
            moment_calculator: the moment calculator
            covariance_mat: the covariance matrix, either as a string or as a numpy array

        Returns:
            None

        Raises:
            ValueError: if the covariance matrix is not valid.
                If it is a string, ot can be invalid if it is not one of the possible options.
                If it is a numpy array, it can be invalid if it is not symmetric or if moment_calculator
                is the default get_mom_ts_1d and the covariance matrix's shape
                is not 18. Other possible errors won't be caught by this
                function, and can only be detected at runtime.
        """
        if isinstance(covariance_mat, str):

            try:
                _CovarianceMatrixType(covariance_mat)
            except ValueError as exc:
                raise ValueError(
                    f"expected one of {list(map(lambda x: x.value, _CovarianceMatrixType))}, got {covariance_mat})"
                ) from exc
            return

        if isinstance(covariance_mat, np.ndarray):
            # a non null covariance_mat was given
            if not is_symmetric(covariance_mat):
                raise ValueError(
                    "the provided covariance matrix is not valid as it is not a symmetric matrix"
                )
            if (moment_calculator is get_mom_ts_1d) and (covariance_mat.shape[0] != 18):
                raise ValueError(
                    "the provided covariance matrix is not valid as it has a wrong shape: "
                    f"expected 18, got {covariance_mat.shape[0]}"
                )
            return

        raise ValueError(
            "please specify a valid covariance matrix, either as a string or directly as a numpy array"
        )

    def compute_loss_1d(
        self, sim_data_ensemble: NDArray[np.float64], real_data: NDArray[np.float64]
    ) -> float:
        """
        Compute the loss based on the 'method of moments'.

        Returns the MSM objective function, i.e. the square difference between the moments of the two time series.

        Args:
            sim_data_ensemble: the first operand
            real_data: the second operand

        Returns:
            the MSM loss over a specific coordinate.
        """
        # compute the moments for the simulated ensemble
        ensemble_sim_mom_1d = np.array(
            [self._moment_calculator(s) for s in sim_data_ensemble]
        )

        # compute moments of the real time series
        real_mom_1d = self._moment_calculator(real_data)

        # write moments in relative terms is needed
        if self._standardise_moments:
            ensemble_sim_mom_1d = ensemble_sim_mom_1d / (abs(real_mom_1d)[None, :])
            real_mom_1d = real_mom_1d / abs(real_mom_1d)

        # mean simulated moments
        sim_mom_1d = np.mean(ensemble_sim_mom_1d, axis=0)

        g = real_mom_1d - sim_mom_1d

        if self._covariance_mat == _CovarianceMatrixType.IDENTITY.value:
            loss_1d = g.dot(g)
            return loss_1d
        if self._covariance_mat == _CovarianceMatrixType.INVERSE_VARIANCE.value:
            W = np.diag(
                1.0 / np.mean((real_mom_1d[None, :] - ensemble_sim_mom_1d) ** 2, axis=0)
            )
        else:
            self._covariance_mat = cast(NDArray[np.float64], self._covariance_mat)
            W = self._covariance_mat
        try:
            loss_1d = g.dot(W).dot(g)
        except ValueError as e:
            covariance_size = W.shape[0]
            moments_size = g.shape[0]

            if covariance_size == moments_size:
                # this value error is not due to a mismatch between the
                # covariance matrix size and the number moments. Let's raise the
                # original error.
                raise

            raise ValueError(
                f"The size of the covariance matrix ({covariance_size}) "
                f"and the number of moments ({moments_size}) should be identical"
            ) from e

        return loss_1d
