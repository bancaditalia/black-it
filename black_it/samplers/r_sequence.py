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

"""This module contains the implementation of the R-sequence sampler."""

import numpy as np
from numpy.typing import NDArray

from black_it.samplers.base import BaseSampler
from black_it.search_space import SearchSpace


class RSequenceSampler(BaseSampler):
    """The R-sequence sampler."""

    @staticmethod
    def r_sequence(seed: int, dims: int) -> NDArray[np.float64]:
        """
        Build R-sequence (http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/).

        Args:
            seed: seed of the sequence
            dims: Size of space.

        Returns:
            Array of params uniformly placed in d-dimensional unit cube.
        """
        phi: float = 2.0
        for _ in range(10):
            phi = pow(1 + phi, 1.0 / (dims + 1))

        # FROM KK:
        # alpha = np.array([pow(1./phi, i+1) for i in range(d)])  # flake8: noqa
        # params = np.array([(0.5 + alpha*(i+1)) % 1 for i in range(nStart, nStop)])  # flake8: noqaq

        # FROM ORIGINAL ARTICLE:
        alpha: NDArray[np.float64] = np.zeros(dims, dtype=np.float64)
        for i in range(dims):
            alpha[i] = pow(1 / phi, i + 1) % 1

        points: NDArray[np.float64] = (0.5 + alpha * (seed + 1)) % 1

        return points

    def single_sample(
        self,
        seed: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Sample a single point uniformly within the search space.

        Args:
            seed: random seed
            search_space: an object containing the details of the parameter search space
            existing_points: the parameters already sampled (not used)
            existing_losses: the loss corresponding to the sampled parameters (not used)

        Returns:
            the parameter sampled
        """
        sampled_point: NDArray[np.float64] = np.zeros(
            shape=search_space.dims, dtype=np.float64
        )
        unit_cube_points: NDArray[np.float64] = RSequenceSampler.r_sequence(
            seed=seed, dims=search_space.dims
        )

        p_bounds: NDArray[np.float64] = search_space.parameters_bounds
        for param_index in range(search_space.dims):
            sampled_point[param_index] = p_bounds[0][param_index] + unit_cube_points[
                param_index
            ] * (p_bounds[1][param_index] - p_bounds[0][param_index])

        return sampled_point
