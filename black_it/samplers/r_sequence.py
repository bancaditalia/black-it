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
from black_it.utils.base import check_arg, digitize_data

_MIN_SEQUENCE_START_INDEX = 20
_MAX_SEQUENCE_START_INDEX = 2**16


class _NPhiApproximator:
    """
    Helper class to compute n-dimensional approximation of the golden ratio.

    Its main purpose is to memoize already computed approximation of Phi^n.
    """

    def __init__(self) -> None:
        """Initialize the object."""
        self._cached_phi = np.array([], dtype=np.float64)

    def get_phi_vector(self, nb_dims: int) -> NDArray[np.float64]:
        """
        Get the vector of [phi^1, phi^2, ..., phi^n].

        Args:
            nb_dims: the number of dimensions.

        Returns:
            The n-dimensional vector of phis.
        """
        nb_cached_phis = len(self._cached_phi)
        if nb_dims <= nb_cached_phis:
            return self._cached_phi[:nb_dims]

        new_phis = self.compute_phi_vector(nb_cached_phis + 1, nb_dims)
        self._cached_phi = np.append(self._cached_phi, new_phis)
        assert len(self._cached_phi) == nb_dims
        return self._cached_phi

    @classmethod
    def compute_phi_vector(cls, start_dim: int, end_dim: int) -> NDArray[np.float64]:
        """
        Compute phi vector.

        Args:
            start_dim: the starting dimension for phi.
            end_dim: the ending dimension for phi.

        Returns:
            the phi vector [phi^{start_dim}, phi^{start_dim+1}, ..., phi^{end_dim}]
        """
        check_arg(
            1 <= start_dim <= end_dim,
            "start_dim and end_dim should be such that 0 <= start_dim <= end_dim; "
            f"got {start_dim} and {end_dim}",
        )
        nb_phis = end_dim - start_dim + 1
        previous_phis = 2.0 * np.ones(shape=nb_phis, dtype=np.float64)
        exponents = [1.0 / (dim + 1) for dim in range(start_dim, end_dim + 1)]
        current_phis = np.power(1 + previous_phis, exponents)
        while not np.allclose(current_phis, previous_phis, rtol=0.0, atol=0.0):
            previous_phis = current_phis
            current_phis = np.power(1 + current_phis, exponents)
        return current_phis


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
        self._phi_approximator = _NPhiApproximator()

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

    def _r_sequence(self, nb_samples: int, dims: int) -> NDArray[np.float64]:
        """
        Compute the R-sequence (http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/).

        Args:
            nb_samples: number of points to sample
            dims: the number of dimensions

        Returns:
            Set of params uniformly placed in d-dimensional unit cube.
        """
        phis = self._phi_approximator.get_phi_vector(dims)
        alpha: NDArray[np.float64] = (1 / phis).reshape((1, -1))
        end_index = self._sequence_index + nb_samples
        indexes = np.arange(self._sequence_index, end_index).reshape((-1, 1))
        points: NDArray[np.float64] = (self._sequence_start + indexes.dot(alpha)) % 1
        self._sequence_index += end_index
        return points

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
        unit_cube_points: NDArray[np.float64] = self._r_sequence(
            nb_samples, search_space.dims
        )
        p_bounds: NDArray[np.float64] = search_space.parameters_bounds
        sampled_points = p_bounds[0] + unit_cube_points * (p_bounds[1] - p_bounds[0])
        return digitize_data(sampled_points, search_space.param_grid)
