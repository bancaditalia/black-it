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

"""
Implementation of the `blackbox` sampler.

The `blackbox` sampler is based on this paper:

    Knysh, Paul, and Yannis Korkolis. "Blackbox: A procedure for parallel optimization of expensive black-box
    functions." arXiv preprint arXiv:1605.00998 (2016).

Our implementation is heavily based on the original implementation:

    https://github.com/paulknysh/blackbox

"""
from enum import Enum
from typing import Optional, cast

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from black_it.samplers.base import BaseSampler
from black_it.samplers.cors import CORSSampler
from black_it.samplers.r_sequence import RSequenceSampler
from black_it.search_space import SearchSpace


class _SamplerState(Enum):
    """Sampler state."""

    EXPLORATION = "random"
    EXPLOITATION = "exploitation"


class KnyshKorkolisSampler(BaseSampler):  # pylint: disable=too-many-instance-attributes
    """Implement the `blackbox` sampler proposed by Knysh and Korkolis."""

    _sampler_state: _SamplerState
    _exploration_sampler: Optional[RSequenceSampler]
    _exploitation_sampler: Optional[CORSSampler]
    _batch_id: int
    _points: Optional[NDArray[np.float64]]
    _losses: NDArray[np.float64]
    _last_batch_start_index: int
    _last_batch_end_index: int

    def __init__(  # pylint: disable=too-many-arguments
        self,
        batch_size: int,
        nb_exploration_batches: int,
        max_samples: int,
        rho0: float = 0.5,
        p: float = 1.0,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the `blackbox` sampler.

        Args:
             batch_size: the batch size
             nb_exploration_batches: the minimum number of exploration batches
             max_samples: the maximum number samples used to compute the density decay
             rho0: initial "balls density"
             p: rate of "balls density" decay (p=1 - linear, p>1 - faster, 0<p<1 - slower)
             random_state: the random state
             verbose: activate verbose mode
        """
        super().__init__(
            batch_size, random_state=random_state, max_duplication_passes=0
        )
        self._nb_exploration_batches = nb_exploration_batches
        self._max_samples = max_samples
        self._rho0 = rho0
        self._p = p
        self._verbose = verbose

        self.reset()

    @property
    def random_state(self) -> Optional[int]:
        """Get the random state."""
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: Optional[int]) -> None:
        """Set the random state."""
        self._random_state = random_state
        self._random_generator = default_rng(self.random_state)
        self.reset()

    def sample_batch(
        self,
        batch_size: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sample a batch of points."""
        if self._batch_id != 0:
            # not the first time sample batch is called. Save losses of new points
            new_losses = existing_losses[
                self._last_batch_start_index : self._last_batch_end_index
            ]
            self._losses = np.concatenate([self._losses, new_losses])

        if (
            self._sampler_state == _SamplerState.EXPLORATION
            and self._batch_id >= self._nb_exploration_batches
        ):
            # the next time, the exploitation sampler will be used
            self._set_exploitation_state()

        if self._sampler_state == _SamplerState.EXPLORATION:
            batch = self._sample_batch_exploration(
                batch_size, search_space, existing_points, existing_losses
            )
        elif self._sampler_state == _SamplerState.EXPLOITATION:
            batch = self._sample_batch_exploitation(
                batch_size, search_space, existing_points, existing_losses
            )
        else:
            raise ValueError("invalid state")

        self._batch_id += 1
        return batch

    def _sample_batch_exploration(
        self,
        batch_size: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sample a batch of points in exploration mode."""
        if self._batch_id == 0:
            self._points = np.zeros((0, search_space.dims))
            self._exploration_sampler = RSequenceSampler(
                self.batch_size,
                random_state=self.random_state,
                max_deduplication_passes=0,
            )

        self._last_batch_start_index = len(existing_points)
        self._last_batch_end_index = self._last_batch_start_index + batch_size

        batch = cast(RSequenceSampler, self._exploration_sampler).sample_batch(
            batch_size, search_space, existing_points, existing_losses
        )
        self._points = np.concatenate([self._points, batch])
        return batch

    def _sample_batch_exploitation(
        self,
        batch_size: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sample a batch of points in exploitation mode."""
        return cast(CORSSampler, self._exploitation_sampler).sample_batch(
            batch_size, search_space, existing_points, existing_losses
        )

    def reset(self) -> None:
        """Reset the sampler."""
        self._sampler_state = _SamplerState.EXPLORATION
        self._exploration_sampler = None
        self._exploitation_sampler = None
        self._batch_id = 0
        self._last_batch_start_index = 0
        self._last_batch_end_index = 0
        self._points = None
        self._losses = np.array([])

    def _set_exploitation_state(self) -> None:
        """Set the exploitation state."""
        self._sampler_state = _SamplerState.EXPLOITATION
        self._exploitation_sampler = CORSSampler(
            self.batch_size,
            cast(NDArray[np.float64], self._points),
            self._losses,
            max_samples=self._max_samples,
            rho0=self._rho0,
            p=self._p,
            random_state=self._get_random_seed(),
            verbose=self._verbose,
        )
