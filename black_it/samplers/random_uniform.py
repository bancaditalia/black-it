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

"""This module contains the implementation of the random uniform sampler."""

import numpy as np
from numpy import random
from numpy.random import default_rng
from numpy.typing import NDArray

from black_it.samplers.base import BaseSampler
from black_it.search_space import SearchSpace


class RandomUniformSampler(BaseSampler):
    """Random uniform sampling."""

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
        random_generator: random.Generator = default_rng(seed)

        sampled_point: NDArray[np.float64] = np.zeros(
            shape=search_space.dims, dtype=np.float64
        )

        for i, params in enumerate(search_space.param_grid):
            sampled_point[i] = random_generator.choice(params)

        return sampled_point
