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
from typing import Optional

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from black_it.samplers.base import _DEFAULT_MAX_DEDUPLICATION_PASSES, BaseSampler
from black_it.search_space import SearchSpace

_MIN_SEQUENCE_START_INDEX = 20
_MAX_SEQUENCE_START_INDEX = 2**16


class RSequenceSampler(BaseSampler):
    """The R-sequence sampler."""

    def __init__(
        self,
        batch_size: int,
        random_state: Optional[int] = None,
        max_deduplication_passes: int = _DEFAULT_MAX_DEDUPLICATION_PASSES,
    ) -> None:
        """
        Initialize the sampler.

        Args:
            batch_size: the number of points sampled every time the sampler is called
            random_state: the random state of the sampler, fixing this number the sampler behaves deterministically
            max_deduplication_passes: (non-negative integer) the maximum number of deduplication passes that are made
                after every batch sampling. Default: 0, i.e. no deduplication happens.
        """
        super().__init__(batch_size, random_state, max_deduplication_passes)

        self._reset()

    @property
    def random_state(self) -> Optional[int]:
        """Get the random state."""
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: Optional[int]) -> None:
        """Set the random state."""
        self._random_state = random_state
        self._random_generator = default_rng(self.random_state)
        self._reset()

    def _reset(self) -> None:
        """Reset the index of the sequence."""
        self._sequence_index = self.random_generator.integers(
            _MIN_SEQUENCE_START_INDEX, _MAX_SEQUENCE_START_INDEX
        )
        self._sequence_start = self.random_generator.random()

    def sample_batch(
        self,
        nb_samples: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Sample points using Halton sequence.

        Args:
            nb_samples: the number of samples
            search_space: an object containing the details of the parameter search space
            existing_points: the parameters already sampled (not used)
            existing_losses: the loss corresponding to the sampled parameters (not used)

        Returns:
            the parameter sampled
        """
        batch = np.zeros((nb_samples, search_space.dims))
        for i in range(nb_samples):
            batch[i] = self.single_sample(self._sequence_index, search_space)
            self._sequence_index += 1
        return batch

    def single_sample(
        self, seed: int, search_space: SearchSpace
    ) -> NDArray[np.float64]:
        """
        Sample a single point uniformly within the search space.
        Args:
            seed: random seed
            search_space: an object containing the details of the parameter search space
        Returns:
            the parameter sampled
        """
        sampled_point: NDArray[np.float64] = np.zeros(
            shape=search_space.dims, dtype=np.float64
        )
        unit_cube_points: NDArray[np.float64] = self._r_sequence(
            seed=seed, dims=search_space.dims
        )

        p_bounds: NDArray[np.float64] = search_space.parameters_bounds
        for param_index in range(search_space.dims):
            sampled_point[param_index] = p_bounds[0][param_index] + unit_cube_points[
                param_index
            ] * (p_bounds[1][param_index] - p_bounds[0][param_index])

        return sampled_point

    def _r_sequence(self, seed: int, dims: int) -> NDArray[np.float64]:
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

        # FROM ORIGINAL ARTICLE:
        alpha: NDArray[np.float64] = np.zeros(dims, dtype=np.float64)
        for i in range(dims):
            alpha[i] = pow(1 / phi, i + 1) % 1

        points: NDArray[np.float64] = (self._sequence_start + alpha * (seed + 1)) % 1
        return points
