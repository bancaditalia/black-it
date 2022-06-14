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

"""This module contains the implementation for the Halton sampler."""
import itertools
from typing import Iterator, List, Optional

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from black_it.samplers.base import BaseSampler
from black_it.search_space import SearchSpace
from black_it.utils.base import check_arg, digitize_data

_MIN_SEQUENCE_START_INDEX = 20
_MAX_SEQUENCE_START_INDEX = 2**16


class HaltonSampler(BaseSampler):
    """
    Halton low discrepancy sequence.

    This snippet implements the Halton sequence following the generalization of
    a sequence of *Van der Corput* in n-dimensions.
    """

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
            max_deduplication_passes: the maximum number of sample deduplication passes.
        """
        super().__init__(batch_size, random_state, max_deduplication_passes)
        self._prime_number_generator = _CachedPrimesCalculator()

        # drop first N entries to avoid linear correlation
        self._reset_sequence_index()

    @property
    def random_state(self) -> Optional[int]:
        """Get the random state."""
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: Optional[int]) -> None:
        """Set the random state."""
        self._random_state = random_state
        self._random_generator = default_rng(self.random_state)
        self._reset_sequence_index()

    def _reset_sequence_index(self) -> None:
        """Reset the sequence index pointer."""
        self._sequence_index = self.random_generator.integers(
            _MIN_SEQUENCE_START_INDEX, _MAX_SEQUENCE_START_INDEX
        )

    def sample_batch(
        self,
        batch_size: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Sample points using Halton sequence.

        Args:
            batch_size: the number of samples
            search_space: an object containing the details of the parameter search space
            existing_points: the parameters already sampled (not used)
            existing_losses: the loss corresponding to the sampled parameters (not used)

        Returns:
            the parameter sampled
        """
        unit_cube_points: NDArray[np.float64] = self._halton(
            batch_size, search_space.dims
        )
        p_bounds: NDArray[np.float64] = search_space.parameters_bounds
        sampled_points = p_bounds[0] + unit_cube_points * (p_bounds[1] - p_bounds[0])
        return digitize_data(sampled_points, search_space.param_grid)

    def _halton(self, nb_samples: int, dims: int) -> NDArray[np.float64]:
        """
        Get a Halton sequence.

        It uses a simple prime number generator, which takes the first `dims` primes.

        Args:
            nb_samples: number of samples
            dims: the number of dimensions of the space to sample from the unitary cube

        Returns:
            sequence of Halton.
        """
        bases: NDArray[np.int64] = self._prime_number_generator.get_n_primes(dims)
        # Generate a sample using a Halton sequence.
        sample: NDArray[np.float64] = halton(
            sample_size=nb_samples, bases=bases, n_start=self._sequence_index
        )

        # increment sequence start index for the next sampling
        self._sequence_index += nb_samples
        return sample


class _PrimesIterator:
    """
    This class implements an iterator that iterates over all primes via unbounded Sieve of Erathosthenes.

    Adapted from:

        https://wthwdik.wordpress.com/2007/08/30/an-unbounded-sieve-of-eratosthenes/

    It caches the sequence of primes up to the highest n.
    """

    def __init__(self) -> None:
        """Initialize the iterator."""
        self._primes = [[2, 2]]
        self._candidate = 2

    def __iter__(self) -> Iterator:
        """Make the class iterable."""
        return self

    def __next__(self) -> int:
        """Get the next prime number."""
        while True:
            self._candidate = self._candidate + 1
            for i in self._primes:
                while self._candidate > i[1]:
                    i[1] = i[0] + i[1]

                if self._candidate == i[1]:
                    break
            else:
                # if here, we have i == primes[-1]:
                self._primes.append([self._candidate, self._candidate])
                return self._candidate


class _CachedPrimesCalculator:  # pylint: disable=too-few-public-methods
    """Utility class to compute and cache the first n prime numbers."""

    def __init__(self) -> None:
        """Initialize the object."""
        self._primes_iterator = _PrimesIterator()
        self._cached_primes: List[int] = [2]

    def get_n_primes(self, n: int) -> NDArray[np.int64]:
        """
        Get the first n primes.

        Args:
            n: the number of primes.

        Returns:
            a list containing the first n primes.
        """
        check_arg(n >= 1, "input must be greater than 0")
        if n <= len(self._cached_primes):
            return np.array(self._cached_primes[:n])

        nb_next_primes = n - len(self._cached_primes)
        next_primes = itertools.islice(self._primes_iterator, nb_next_primes)
        self._cached_primes.extend(next_primes)
        return np.array(self._cached_primes[:n])


def halton(
    sample_size: int, bases: NDArray[np.int64], n_start: int
) -> NDArray[np.float64]:
    """
    Van der Corput sequence, generalized as to accept a starting point in the sequence.

    Args:
        sample_size:  number of element of the sequence
        bases: bases of the sequence
        n_start: starting point of the sequence

    Returns:
        sequence of Halton
    """
    check_arg(sample_size > 0, "sample size must be greater than zero")
    check_arg(bool((bases > 1).all()), "based must be greater than one")
    check_arg(n_start >= 0, "n_start must be greater or equal zero")
    nb_bases = len(bases)
    sequence: NDArray[np.float64] = np.zeros(shape=(sample_size, nb_bases))
    for index in range(n_start + 1, sample_size + n_start + 1):
        n_th_numbers: NDArray[np.float64] = np.zeros(shape=nb_bases)
        denoms: NDArray[np.float64] = np.ones(shape=nb_bases)
        done: NDArray[np.bool8] = np.zeros(shape=nb_bases, dtype=np.bool8)
        i = np.repeat(np.int64(index), repeats=nb_bases)
        while (i > 0).any():
            i, remainders = np.divmod(i, bases)
            denoms *= bases
            # mask reminders in case i = 0
            remainders[done] = 0.0
            n_th_numbers += remainders / denoms
            done[i == 0] = True
        sequence[index - 1 - n_start, :] = n_th_numbers
    return sequence
