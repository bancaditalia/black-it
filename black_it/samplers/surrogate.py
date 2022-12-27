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

"""This module defines the 'MLSurrogateSampler' base class."""

from abc import abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from black_it.samplers.base import BaseSampler
from black_it.samplers.random_uniform import RandomUniformSampler
from black_it.search_space import SearchSpace
from black_it.utils.base import digitize_data


class MLSurrogateSampler(BaseSampler):
    """
    MLSurrogateSampler interface.

    This is the base class for all machine learning surrogate samplers.
    """

    def __init__(
        self,
        batch_size: int,
        random_state: Optional[int] = None,
        max_deduplication_passes: int = 5,
        candidate_pool_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the sampler.

        Args:
            batch_size: the number of points sampled every time the sampler is called
            random_state: the internal state of the sampler, fixing this numbers the sampler behaves deterministically
            max_deduplication_passes: maximum number of duplication passes done to avoid sampling repeated parameters
            candidate_pool_size: number of randomly sampled points on which the ML surrogate predictions are evaluated
        """
        super().__init__(batch_size, random_state, max_deduplication_passes)

        if candidate_pool_size is not None:
            self._candidate_pool_size = candidate_pool_size
        else:
            self._candidate_pool_size = 1000 * batch_size

    @property
    def candidate_pool_size(self) -> int:
        """Get the candidate pool size."""
        return self._candidate_pool_size

    def sample_candidates(
        self,
        candidate_pool_size: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Get a large pool of candidate parameters."""
        candidates = RandomUniformSampler(
            candidate_pool_size, random_state=self._get_random_seed()
        ).sample_batch(
            candidate_pool_size, search_space, existing_points, existing_losses
        )
        return candidates

    @abstractmethod
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Abstract method to fit the loss function of an ML surrogate."""

    @abstractmethod
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Abstract method for the predictions of an ML surrogate."""

    def sample_batch(
        self,
        batch_size: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Sample a number of new parameters fixed by the 'batch_size' attribute.

        Args:
            batch_size: number of samples to collect
            search_space: an object containing the details of the parameter search space
            existing_points: the parameters already sampled
            existing_losses: the loss corresponding to the sampled parameters

        Returns:
            the new parameters
        """
        # Get a large pool of potential candidates
        candidates = self.sample_candidates(
            self.candidate_pool_size, search_space, existing_points, existing_losses
        )

        # Train surrogate model
        self.fit(existing_points, existing_losses)

        # Predictions of surrogate on large pool of candidates
        predictions = self.predict(candidates)

        # Select candidates with lowest predicted loss value
        sorting_indices: NDArray[np.int64] = np.argsort(predictions)
        sampled_points: NDArray[np.float64] = candidates[sorting_indices][:batch_size]

        return digitize_data(sampled_points, search_space.param_grid)
