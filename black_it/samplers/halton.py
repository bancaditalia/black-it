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

"""This module contains the implementation fo the Halton sampler."""

import numpy as np
from numpy.typing import NDArray

from black_it.samplers.base import BaseSampler
from black_it.search_space import SearchSpace


class HaltonSampler(BaseSampler):
    """
    Halton low discrepancy sequence.

    This snippet implements the Halton sequence following the generalization of
    a sequence of *Van der Corput* in n-dimensions.
    """

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
        unit_cube_points: NDArray[np.float64] = HaltonSampler.halton(
            seed=seed, dims=search_space.dims
        )

        p_bounds: NDArray[np.float64] = search_space.parameters_bounds
        for param_index in range(search_space.dims):
            sampled_point[param_index] = p_bounds[0][param_index] + unit_cube_points[
                param_index
            ] * (p_bounds[1][param_index] - p_bounds[0][param_index])

        return sampled_point

    @staticmethod
    def primes_from_2_to_n(n: int) -> NDArray[np.int64]:
        """
        Prime numbers from 2 to n.

        From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.

        Args:
            n: sup bound with ``n >= 6``.

        Returns:
            primes in 2 <= p < n.
        """
        sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool_)
        for i in range(1, int(n**0.5) // 3 + 1):
            if sieve[i]:
                k = 3 * i + 1 | 1
                sieve[k * k // 3 :: 2 * k] = False
                sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
        return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

    @staticmethod
    def van_der_corput(
        sample_size: int, base: int = 2, n_start: int = 0
    ) -> NDArray[np.float64]:
        """
        Van der Corput sequence, generalized as to accept a starting point in the sequence.

        Args:
            sample_size:  number of element of the sequence
            base: base of the sequence
            n_start: starting point of the sequence

        Returns:
            sequence of Van der Corput
        """
        sequence: NDArray[np.float64] = np.zeros(shape=sample_size - n_start)
        for index in range(n_start, sample_size):
            n_th_number: float = 0.0
            denom: float = 1.0
            i: int = index
            while i > 0:
                i, remainder = divmod(i, base)
                denom *= base
                n_th_number += remainder / denom
            sequence[index - n_start] = n_th_number
        return sequence

    @staticmethod
    def halton(seed: int = 0, dims: int = 1) -> NDArray[np.float64]:
        """
        Halton sequence.

        Changed in order to accept a starting point in the sequence and to adapt to the calibration program.

        Args:
            seed: seed of sequence
            dims: dimension

        Returns:
            sequence of Halton.
        """
        big_number: int = 10
        while "Not enough primes":
            bases: NDArray[np.int64] = HaltonSampler.primes_from_2_to_n(big_number)[
                :dims
            ]
            if len(bases) == dims:
                break
            big_number += 1000

        n_samples: int = 1
        # Generate a sample using a Van der Corput sequence per dimension.
        sample: NDArray[np.float64] = np.zeros(shape=dims)
        for i, prime in enumerate(bases):
            sample[i] = HaltonSampler.van_der_corput(
                sample_size=n_samples + 1 + seed, base=prime, n_start=seed
            )[1:]

        return sample
