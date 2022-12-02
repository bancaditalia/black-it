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

"""This module contains the implementation of the best-batch sampler."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import betabinom

from black_it.samplers.base import BaseSampler
from black_it.search_space import SearchSpace
from black_it.utils.base import _assert


class BestBatchSampler(BaseSampler):
    """This class implements the best-batch sampler.

    The sampler is a very essential type of genetic algorithm that takes the parameters corresponding
      to the current lowest loss values and perturbs them slightly in a purely random fashion.
    The sampler first chooses the total number of coordinates to perturb via a beta-binomial distribution
      BetaBin(dims, a, b) --where dims is the total number of dimensions in the search space --, it then selects
      that many coordinate randomly, and perturbs them uniformly within the range specified by 'perturbation_range'.

    """

    def __init__(
        self,
        batch_size: int,
        random_state: Optional[int] = None,
        max_deduplication_passes: int = 5,
        a: float = 3.0,
        b: float = 1.0,
        perturbation_range: int = 6,
    ):
        """
        Initialize the sampler.

        Args:
            batch_size: the number of points sampled every time the sampler is called
            random_state: the random state of the sampler, fixing this number the sampler behaves deterministically
            max_deduplication_passes: the maximum number of deduplication passes that are made
            a: the a parameter of the beta-binomial distribution
            b: the b parameter of the beta-binomial distribution
            perturbation_range: the range of the perturbation applied. The actual perturbation will be in the range
                plus/minus the perturbation_range times the precision of the specific parameter coordinate
        """
        _assert(
            a > 0.0,
            "'a' should be greater than zero",
        )
        _assert(
            b > 0.0,
            "'b' should be greater than zero",
        )
        _assert(
            perturbation_range > 1,
            "'perturbation_range' should be greater than one",
        )

        super().__init__(batch_size, random_state, max_deduplication_passes)
        self.a = a
        self.b = b
        self.perturbation_range = perturbation_range

    def sample_batch(
        self,
        batch_size: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Sample from the search space using a genetic algorithm.

        Args:
            batch_size: the number of points to sample
            search_space: an object containing the details of the parameter search space
            existing_points: the parameters already sampled
            existing_losses: the loss corresponding to the sampled parameters

        Returns:
            the sampled parameters (an array of shape `(self.batch_size, search_space.dims)`)
        """
        if len(existing_points) < batch_size:
            raise ValueError(
                "best-batch sampler requires a number of existing points "
                f"which is at least the batch size {batch_size}, "
                f"got {len(existing_points)}"
            )

        # sort existing params
        candidate_points: NDArray[np.float64] = existing_points[
            np.argsort(existing_losses)
        ][:batch_size, :]

        candidate_point_indexes: NDArray[np.int64] = self.random_generator.integers(
            0, batch_size, size=batch_size
        )
        sampled_points: NDArray[np.float64] = np.copy(
            candidate_points[candidate_point_indexes]
        )

        beta_binom_rv = betabinom(n=search_space.dims - 1, a=self.a, b=self.b)
        beta_binom_rv.random_state = self.random_generator

        for sampled_point in sampled_points:
            num_shocks: NDArray[np.int64] = beta_binom_rv.rvs(size=1) + 1
            params_shocked: NDArray[np.int64] = self.random_generator.choice(
                search_space.dims, tuple(num_shocks), replace=False
            )
            for index in params_shocked:
                shock_size: int = self.random_generator.integers(
                    1, self.perturbation_range
                )
                shock_sign: int = (self.random_generator.integers(0, 2) * 2) - 1

                delta: float = search_space.parameters_precision[index]
                shift: float = delta * shock_sign * shock_size
                sampled_point[index] += shift

                sampled_point[index] = np.clip(
                    sampled_point[index],
                    search_space.parameters_bounds[0][index],
                    search_space.parameters_bounds[1][index],
                )

        return sampled_points
