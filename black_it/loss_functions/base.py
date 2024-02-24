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

"""This module defines the 'BaseLoss' base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from black_it.utils.base import _assert

if TYPE_CHECKING:
    from collections.abc import Sequence

TimeSeriesFilter = Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
"""A filter that receives a time series and returns its filtered version. Used by the BaseLoss constructor."""


class BaseLoss(ABC):
    """BaseLoss interface."""

    def __init__(
        self,
        coordinate_weights: NDArray | None = None,
        coordinate_filters: Sequence[Callable | None] | None = None,
    ) -> None:
        """Initialize the loss function.

        Args:
            coordinate_weights: the weights of the loss coordinates.
            coordinate_filters: filters/transformations to be applied to each simulated series before
                the loss computation.
        """
        self.coordinate_weights = coordinate_weights
        self.coordinate_filters = coordinate_filters

    def compute_loss(
        self,
        sim_data_ensemble: NDArray[np.float64],
        real_data: NDArray[np.float64],
    ) -> float:
        """Compute the loss between simulated and real data.

        Args:
            sim_data_ensemble: an ensemble of simulated data, of shape (ensemble_size, N, D)
            real_data: the real data, of shape (N, D)

        Returns:
            The loss value.
        """
        num_coords = real_data.shape[1]

        weights = self._check_coordinate_weights(num_coords)
        filters = self._check_coordinate_filters(num_coords)

        filtered_data = self._filter_data(filters, sim_data_ensemble)

        loss = 0
        for i in range(num_coords):
            loss += self.compute_loss_1d(filtered_data[i], real_data[:, i]) * weights[i]

        return loss

    @staticmethod
    def _filter_data(
        filters: Sequence[Callable | None],
        sim_data_ensemble: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Filter the simulated time series."""
        filtered_data = []

        for i, filter_ in enumerate(filters):
            if filter_ is None:
                filtered_data_1d = sim_data_ensemble[:, :, i]
            else:
                filtered_data_1d = np.array(
                    [
                        filter_(sim_data_ensemble[j, :, i])
                        for j in range(sim_data_ensemble.shape[0])
                    ],
                )
            filtered_data.append(filtered_data_1d)

        return np.array(filtered_data)

    def _check_coordinate_weights(self, num_coords: int) -> NDArray[np.float64]:
        """Check self.coordinate_weights and return usable weights."""
        weights: NDArray[np.float64]

        if self.coordinate_weights is None:
            weights = np.ones(num_coords) / num_coords
        else:
            nb_coordinate_weights = len(self.coordinate_weights)
            _assert(
                nb_coordinate_weights == num_coords,
                (
                    "the length of coordinate_weights should be equal "
                    f"to the number of coordinates, got {nb_coordinate_weights} and {num_coords}"
                ),
                exception_class=ValueError,
            )
            weights = self.coordinate_weights

        return weights

    def _check_coordinate_filters(self, num_coords: int) -> Sequence[Callable | None]:
        """Check self.coordinate_filters and return usable filters."""
        filters: Sequence[Callable | None]

        if self.coordinate_filters is None:
            # a list of identity functions
            filters = [None] * num_coords
        else:
            nb_coordinate_filters = len(self.coordinate_filters)
            _assert(
                nb_coordinate_filters == num_coords,
                (
                    "the length of coordinate_filters should be equal "
                    f"to the number of coordinates, got {nb_coordinate_filters} and {num_coords}"
                ),
                exception_class=ValueError,
            )
            filters = self.coordinate_filters

        return filters

    @abstractmethod
    def compute_loss_1d(
        self,
        sim_data_ensemble: NDArray[np.float64],
        real_data: NDArray[np.float64],
    ) -> float:
        """Return the loss between a specific coordinate of two time series.

        Concrete classes have to override this method in order to implement new
        loss functions.

        Args:
            sim_data_ensemble: an ensemble of simulated 1D series, shape (ensemble_size, N, 1)
            real_data: the real data, shape (N, 1)

        Returns:
            the computed loss over the specific coordinate
        """
