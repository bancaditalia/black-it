# Black-box ABM Calibration Kit (Black-it)
# Copyright (C) 2021-2023 Banca d'Italia
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

"""This module defines the 'BaseSampler' base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from black_it.utils.seedable import BaseSeedable

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from black_it.search_space import SearchSpace


class BaseSampler(BaseSeedable, ABC):
    """BaseSampler interface.

    This is the base class for all samplers.
    """

    def __init__(
        self,
        batch_size: int,
        random_state: int | None = None,
        max_deduplication_passes: int = 5,
    ) -> None:
        """Initialize the sampler.

        Args:
            batch_size: the number of points sampled every time the sampler is called
            random_state: the internal state of the sampler, fixing this numbers the sampler behaves deterministically
            max_deduplication_passes: maximum number of duplication passes done to avoid sampling repeated parameters
        """
        BaseSeedable.__init__(self, random_state=random_state)
        self.batch_size: int = batch_size
        self.max_deduplication_passes = max_deduplication_passes

    @abstractmethod
    def sample_batch(
        self,
        batch_size: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sample a number of new parameters fixed by the 'batch_size' attribute.

        Args:
            batch_size: number of samples to collect
            search_space: an object containing the details of the parameter search space
            existing_points: the parameters already sampled
            existing_losses: the loss corresponding to the sampled parameters

        Returns:
            the new parameters
        """

    def sample(
        self,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sample from the search space.

        Args:
            search_space: an object containing the details of the parameter search space
            existing_points: the parameters already sampled
            existing_losses: the loss corresponding to the sampled parameters

        Returns:
            the sampled parameters
        """
        samples = self.sample_batch(
            self.batch_size,
            search_space,
            existing_points,
            existing_losses,
        )

        for n in range(self.max_deduplication_passes):
            duplicates = self.find_and_get_duplicates(samples, existing_points)

            num_duplicates = len(duplicates)

            if num_duplicates == 0:
                break

            new_samples = self.sample_batch(
                num_duplicates,
                search_space,
                existing_points,
                existing_losses,
            )
            samples[duplicates] = new_samples

            if n == self.max_deduplication_passes - 1:
                print(
                    f"Warning: Repeated samples still found after {self.max_deduplication_passes} duplication passes."
                    " This is probably due to a small search space.",
                )

        return samples

    @staticmethod
    def find_and_get_duplicates(
        new_points: NDArray[np.float64],
        existing_points: NDArray[np.float64],
    ) -> list:
        """Find the points in 'new_points' that are already present in 'existing_points'.

        Args:
            new_points: candidates points for the sampler
            existing_points: previously sampled points

        Returns:
            the location of the duplicates in 'new_points'
        """
        all_points = np.concatenate((existing_points, new_points))
        unq, count = np.unique(all_points, axis=0, return_counts=True)
        repeated_groups = unq[count > 1]

        repeated_pos = []
        if len(repeated_groups) > 0:
            for repeated_group in repeated_groups:
                repeated_idx = np.argwhere(np.all(new_points == repeated_group, axis=1))
                for index in repeated_idx:
                    repeated_pos.append(index[0])  # noqa: PERF401

        return repeated_pos
