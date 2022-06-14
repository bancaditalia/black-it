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
from numpy.typing import NDArray
from scipy.stats import betabinom

from black_it.samplers.base import BaseSampler
from black_it.search_space import SearchSpace


class BestBatchSampler(BaseSampler):
    """This class implements the best-batch sampler."""

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

        beta_binom_rv = betabinom(n=search_space.dims - 1, a=3.0, b=1.0)
        beta_binom_rv.random_state = self.random_generator

        for sampled_point in sampled_points:
            num_shocks: NDArray[np.int64] = beta_binom_rv.rvs(size=1) + 1
            params_shocked: NDArray[np.int64] = self.random_generator.choice(
                search_space.dims, tuple(num_shocks), replace=False
            )
            for index in params_shocked:
                shock_size: int = self.random_generator.integers(1, 6)
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
