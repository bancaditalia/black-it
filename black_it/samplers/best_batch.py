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

import numpy as np
from numpy import random
from numpy.random import default_rng
from numpy.typing import NDArray
from scipy.stats import betabinom

from black_it.samplers.base import BaseSampler
from black_it.search_space import SearchSpace


class BestBatchSampler(BaseSampler):
    """This class implements the best-batch sampler."""

    def single_sample(
        self,
        seed: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Sample a single parameter.

        Args:
            seed: random seed
            search_space: an object containing the details of the parameter search space
            existing_points: the parameters already sampled
            existing_losses: the loss corresponding to the sampled parameters

        Returns:
            the parameter sampled

        Raises:
            ValueError: if the existing_points array is empty
        """
        if len(existing_points) < self.batch_size:
            raise ValueError(
                "best-batch sampler requires a number of existing points "
                f"which is at least the batch size {self.batch_size}, "
                f"got {len(existing_points)}"
            )

        random_generator: random.Generator = default_rng(seed)

        # sort existing params
        candidate_points: NDArray[np.float64] = existing_points[
            np.argsort(existing_losses)
        ][: self.batch_size, :]

        candidate_point_index: int = random_generator.integers(0, self.batch_size)
        sampled_point: NDArray[np.float64] = np.copy(
            candidate_points[candidate_point_index]
        )

        beta_binom_rv = betabinom(n=search_space.dims - 1, a=3.0, b=1.0)
        beta_binom_rv.random_state = random_generator
        num_shocks: NDArray[np.int64] = beta_binom_rv.rvs(size=1) + 1

        params_shocked: NDArray[np.int64] = random_generator.choice(
            search_space.dims, tuple(num_shocks), replace=False
        )

        for index in params_shocked:
            shock_size: int = random_generator.integers(1, 6)
            shock_sign: int = (random_generator.integers(0, 2) * 2) - 1

            delta: float = search_space.parameters_precision[index]
            shift: float = delta * shock_sign * shock_size
            sampled_point[index] += shift

            if sampled_point[index] < search_space.parameters_bounds[0][index]:
                sampled_point[index] = search_space.parameters_bounds[0][index]
            elif sampled_point[index] > search_space.parameters_bounds[1][index]:
                sampled_point[index] = search_space.parameters_bounds[1][index]

        return sampled_point
