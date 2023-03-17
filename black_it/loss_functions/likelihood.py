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

"""This module defines the 'LikelihoodLoss'  class."""
import warnings
from typing import Callable, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from black_it.loss_functions.base import BaseLoss


def kernel(sq_dist: NDArray[np.float64], h: float, D: int) -> NDArray[np.float64]:
    """Compute a kernel density estimation using a Gaussian kernel."""
    k = np.exp(-(sq_dist / (2 * h**2))) / (h**D * (2 * np.pi) ** (D / 2.0))
    return k


class LikelihoodLoss(BaseLoss):
    """Class for a likelihood loss computation using a Gaussian kernel function.

    The likelihood computation can be useful within the approach proposed in
    "Grazzini et al, Bayesian estimation of agent-based models, JEDC (2017)".

    """

    def __init__(
        self,
        coordinate_weights: Optional[NDArray] = None,
        coordinate_filters: Optional[List[Optional[Callable]]] = None,
        h: Union[str, float] = "silverman",
    ):
        """
        Initialize the loss function.

        Args:
            coordinate_weights: the weights of the loss coordinates.
            coordinate_filters: filters/transformations to be applied to each simulated series before
                the loss computation.
            h: the bandwidth of the kernel used to estimate the likelihood
        """
        super().__init__(coordinate_weights, coordinate_filters)
        self.h = h

    @staticmethod
    def _get_bandwidth_silverman(N: int, D: int) -> float:
        """Return a reasonable bandwidth value computed using the Silverman's rule of thumb."""
        h = ((N * (D + 2)) / 4) ** (-1 / (D + 4))
        return h

    @staticmethod
    def _get_bandwidth_scott(N: int, D: int) -> float:
        """Return a reasonable bandwidth value computed using the Scott's rule of thumb."""
        h = N ** (-1 / (D + 4))
        return h

    def compute_loss(
        self, sim_data_ensemble: NDArray[np.float64], real_data: NDArray[np.float64]
    ) -> float:
        """
        Compute the loss between simulated and real data.

        Args:
            sim_data_ensemble: an ensemble of simulated data, of shape (ensemble_size, N, D)
            real_data: the real data, of shape (T, D)

        Returns:
            The loss value.
        """
        R = sim_data_ensemble.shape[0]  # number of repetitions
        S = sim_data_ensemble.shape[1]  # simulation length
        D = sim_data_ensemble.shape[2]  # number of dimensions

        if self.coordinate_weights is not None:
            warnings.warn(  # noqa: B028
                "The LikelihoodLoss cannot take any coordinate weights. "
                "The provided coordinate_weights will be ignored",
                RuntimeWarning,
            )

        filters = self._check_coordinate_filters(D)
        filtered_data = self._filter_data(filters, sim_data_ensemble)
        sim_data_ensemble = np.transpose(filtered_data, (1, 2, 0))

        h = self._check_bandwidth(S, D)
        sq_dists_r_t_s = (
            1.0
            / D
            * np.sum(
                (sim_data_ensemble[:, None, :, :] - real_data[None, :, None, :]) ** 2,
                axis=3,
            )
        )

        kernel_r_t_s = kernel(sq_dists_r_t_s, h, D)
        lik_real_series_r_t = np.sum(kernel_r_t_s, axis=2) / S
        log_lik_real_series_r = np.sum(np.log(lik_real_series_r_t), axis=1)
        log_lik_real_series = np.sum(log_lik_real_series_r, axis=0) / R
        return -log_lik_real_series

    def _check_bandwidth(self, S: int, D: int) -> float:
        """Check the bandwidth self.h and return a usable one."""
        h: float

        if isinstance(self.h, str):
            if self.h == "silverman":
                h = self._get_bandwidth_silverman(S, D)
            elif self.h == "scott":
                h = self._get_bandwidth_scott(S, D)
            else:
                raise KeyError(
                    "Select a valid rule of thumb (either 'silverman' or 'scott') "
                    "or directly a numerical value for the bandwidth"
                )
        else:
            h = self.h

        return h

    def compute_loss_1d(
        self, sim_data_ensemble: NDArray[np.float64], real_data: NDArray[np.float64]
    ) -> float:
        """Compute likelihood loss on a single dimension, not available."""
        raise NotImplementedError(
            "The likelihood cannot be currently computed on a single dimension."
        )
