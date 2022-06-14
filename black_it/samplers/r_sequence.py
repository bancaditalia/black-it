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

from black_it.samplers.base import BaseSampler
from black_it.search_space import SearchSpace
from black_it.utils.base import check_arg, digitize_data

_MIN_SEQUENCE_START_INDEX = 20
_MAX_SEQUENCE_START_INDEX = 2**16


class RSequenceSampler(BaseSampler):
    """The R-sequence sampler."""

    def __init__(
        self,
        batch_size: int,
        random_state: Optional[int] = None,
        max_deduplication_passes: int = 5,
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

        self._sequence_index: int
        self._sequence_start: float
        self._reset()

    @classmethod
    def compute_phi(cls, nb_dims: int) -> float:
        """
        Get an approximation of phi^nb_dims.

        Args:
            nb_dims: the number of dimensions.

        Returns:
            phi^nb_dims
        """
        check_arg(1 <= nb_dims, f"nb_dims should be greater than 0, got {nb_dims}")
        phi: float = 2.0
        old_phi = None
        while old_phi != phi:
            old_phi = phi
            phi = pow(1 + phi, 1.0 / (nb_dims + 1))
        return phi

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
        batch_size: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Sample points using the R-sequence.

        Args:
            batch_size: the number of samples
            search_space: an object containing the details of the parameter search space
            existing_points: the parameters already sampled (not used)
            existing_losses: the loss corresponding to the sampled parameters (not used)

        Returns:
            the parameter sampled
        """
        unit_cube_points: NDArray[np.float64] = self._r_sequence(
            batch_size, search_space.dims
        )
        p_bounds: NDArray[np.float64] = search_space.parameters_bounds
        sampled_points = p_bounds[0] + unit_cube_points * (p_bounds[1] - p_bounds[0])
        return digitize_data(sampled_points, search_space.param_grid)

    def _r_sequence(self, nb_samples: int, dims: int) -> NDArray[np.float64]:
        """
        Compute the R-sequence (http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/).

        Args:
            nb_samples: number of points to sample
            dims: the number of dimensions

        Returns:
            Set of params uniformly placed in d-dimensional unit cube.
        """
        phi = self.compute_phi(dims)
        alpha: NDArray[np.float64] = np.power(1 / phi, np.arange(1, dims + 1)).reshape(
            (1, -1)
        )
        end_index = self._sequence_index + nb_samples
        indexes = np.arange(self._sequence_index, end_index).reshape((-1, 1))
        points: NDArray[np.float64] = (self._sequence_start + indexes.dot(alpha)) % 1
        self._sequence_index = end_index
        return points
