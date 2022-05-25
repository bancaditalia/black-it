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

"""This module defines the 'BaseLoss' base class."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from black_it.utils.base import assert_


class BaseLoss(ABC):
    """BaseLoss interface."""

    def __init__(self, coordinate_weights: Optional[NDArray] = None):
        """
        Initialize the loss function.

        Args:
            coordinate_weights: the weights of the loss coordinates.
        """
        self.coordinate_weights = coordinate_weights

    def compute_loss(
        self, sim_data_ensemble: NDArray[np.float64], real_data: NDArray[np.float64]
    ) -> float:
        """
        Compute the loss between simulated and real data.

        Args:
            sim_data_ensemble: an ensemble of simulated data, of shape (ensemble_size, N, D)
            real_data: the real data, of shape (N, D)

        Returns:
            The loss value.
        """
        num_coords = real_data.shape[1]

        if self.coordinate_weights is None:
            weights = np.ones(num_coords) / num_coords
        else:
            nb_coordinate_weights = len(self.coordinate_weights)
            assert_(
                nb_coordinate_weights == num_coords,
                (
                    "the length of coordinate_weights should be equal "
                    f"to the number of coordinates, got {nb_coordinate_weights} and {num_coords}"
                ),
                exc_cls=ValueError,
            )
            weights = self.coordinate_weights

        loss = 0
        for i in range(num_coords):
            loss += (
                self.compute_loss_1d(sim_data_ensemble[:, :, i], real_data[:, i])
                * weights[i]
            )

        return loss

    @abstractmethod
    def compute_loss_1d(
        self, sim_data_ensemble: NDArray[np.float64], real_data: NDArray[np.float64]
    ) -> float:
        """
        Return the loss between a specific coordinate of two time series.

        Concrete classes have to override this method in order to implement new
        loss functions.

        Args:
            sim_data_ensemble: an ensemble of simulated 1D series, shape (ensemble_size, N, 1)
            real_data: the real data, shape (N, 1)

        Returns:
            the computed loss over the specific coordinate
        """
