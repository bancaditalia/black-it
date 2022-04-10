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
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from black_it.samplers.base import _DEFAULT_MAX_DEDUPLICATION_PASSES, BaseSampler
from black_it.search_space import SearchSpace


class RandomUniformSampler(BaseSampler):
    """Random uniform sampling."""

    def __init__(
        self,
        batch_size: int,
        random_state: Optional[int] = None,
        max_deduplication_passes: int = _DEFAULT_MAX_DEDUPLICATION_PASSES,
    ) -> None:
        """Initialize the random uniform sampler."""
        super().__init__(batch_size, random_state, max_deduplication_passes)

    def sample_batch(
        self,
        nb_samples: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Sample uniformly from the search space.

        Args:
            nb_samples: the number of points to sample
            search_space: an object containing the details of the parameter search space
            existing_points: the parameters already sampled
            existing_losses: the loss corresponding to the sampled parameters

        Returns:
            the sampled parameters (an array of shape `(self.batch_size, search_space.dims)`)
        """
        candidates = np.zeros((nb_samples, search_space.dims))
        for i, params in enumerate(search_space.param_grid):
            candidates[:, i] = self._random_generator.choice(params, size=(nb_samples,))
        return candidates
