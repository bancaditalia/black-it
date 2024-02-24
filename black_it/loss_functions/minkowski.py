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

"""This module contains the implementation of the quadratic loss."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from scipy.spatial.distance import minkowski  # type: ignore[import]

from black_it.loss_functions.base import BaseLoss

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class MinkowskiLoss(BaseLoss):
    """Class for the Minkowski loss."""

    def __init__(
        self,
        p: int = 2,
        coordinate_weights: NDArray | None = None,
        coordinate_filters: list[Callable | None] | None = None,  # noqa: ARG002
    ) -> None:
        """Loss computed using a Minkowski distance.

        The [Minkowski distance](https://en.wikipedia.org/wiki/Minkowski_distance)
        is a generalization of both the Manhattan distance (p=1) and the Euclidean distance (p=2).

        This function computes the Minkowski distance between two series.

        Note: this class is a wrapper of scipy.spatial.distance.minkowski

        Args:
            p: The order of the norm used to compute the distance between real and simulated series
            coordinate_weights: The order of the norm used to compute the distance between real and simulated series
            coordinate_filters: filters/transformations to be applied to each simulated series before
                the loss computation.
        """
        self.p = p
        super().__init__(coordinate_weights)

    def compute_loss_1d(
        self,
        sim_data_ensemble: NDArray[np.float64],
        real_data: NDArray[np.float64],
    ) -> float:
        """Call scipy.spatial.distance.minkowski() on its arguments.

        Args:
            sim_data_ensemble: the first operand
            real_data: the second operand

        Returns:
            The computed loss over the specific coordinate.
        """
        # average simulated time series
        sim_data_ensemble = sim_data_ensemble.mean(axis=0)

        return minkowski(sim_data_ensemble, real_data, p=self.p)
