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

"""This module defines the 'BaseSampler' base class."""
import functools
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from black_it.search_space import SearchSpace
from black_it.utils.base import check_arg, get_random_seed

_DEFAULT_MAX_DEDUPLICATION_PASSES = 5


def find_and_get_duplicates(
    new_points: NDArray[np.float64], existing_points: NDArray[np.float64]
) -> List:
    """
    Find the points in 'new_points' that are already present in 'existing_points'.

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
                repeated_pos.append(index[0])

    return repeated_pos


def remove_duplicates(
    max_deduplication_passes: int = _DEFAULT_MAX_DEDUPLICATION_PASSES,
) -> Callable:
    """
    Wrap sampling function so to return a sample set without duplicates.

    Args:
        max_deduplication_passes: the maximum number of deduplication passes to do.

    Returns:
        A new sampling function that returns a deduplicated sample set (if possible).
    """

    def decorator(
        sample_function: Callable[
            [int, SearchSpace, NDArray, NDArray], NDArray[np.float64]
        ]
    ) -> Callable[[int, SearchSpace, NDArray, NDArray], NDArray[np.float64]]:
        """
        Decorate the sample function.

        Args:
            sample_function: the sampling function.

        Returns:
            the wrapped sample_function.
        """

        @functools.wraps(sample_function)
        def wrapper(
            batch_size: int,
            search_space: SearchSpace,
            existing_points: NDArray[np.float64],
            existing_losses: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            """
            Get a set of unique samples.

            Args:
                batch_size: number of samples to collect
                search_space: an object containing the details of the parameter search space
                existing_points: the parameters already sampled
                existing_losses: the loss corresponding to the sampled parameters

            Returns:
                the deduplicated sample.
            """
            samples = sample_function(
                batch_size, search_space, existing_points, existing_losses
            )

            duplicates = find_and_get_duplicates(samples, existing_points)
            num_duplicates = len(duplicates)

            if num_duplicates > 0:

                for _ in range(max_deduplication_passes):

                    new_samples = sample_function(
                        num_duplicates, search_space, existing_points, existing_losses
                    )
                    samples[duplicates] = new_samples

                    duplicates = find_and_get_duplicates(samples, existing_points)
                    num_duplicates = len(duplicates)

                    if num_duplicates == 0:
                        break

            if num_duplicates > 0:
                print(
                    f"Warning: {num_duplicates} repeated samples still found after {max_deduplication_passes} "
                    f"deduplication passes. This is probably due to a small search space."
                )
            return samples

        return wrapper

    return decorator


class BaseSampler(ABC):
    """This is the abstract base class for all samplers."""

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
        check_arg(batch_size > 0, "'batch_size' should be greater than 0")
        check_arg(
            max_deduplication_passes >= 0,
            "'max_deduplication_passes' size should be greater or equal than 0",
        )
        self.__batch_size: int = batch_size
        self.__max_deduplication_passes: int = max_deduplication_passes
        self.random_state: Optional[int] = random_state

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self.__batch_size

    @property
    def max_deduplication_passes(self) -> int:
        """Get the maximum number of deduplication passes."""
        return self.__max_deduplication_passes

    @property
    def random_state(self) -> Optional[int]:
        """Get the random state."""
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: Optional[int]) -> None:
        """Set the random state."""
        self._random_state = random_state
        self._random_generator = default_rng(self.random_state)

    @property
    def random_generator(self) -> np.random.Generator:
        """Get the random generator."""
        return self._random_generator

    def _get_random_seed(self) -> int:
        """Get new random seed from the current random generator."""
        return get_random_seed(self._random_generator)

    @abstractmethod
    def sample_batch(  # pylint: disable=method-hidden
        self,
        nb_samples: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Sample a batch of points from the search space.

        Args:
            nb_samples: the number of points to sample
            search_space: the search space
            existing_points: the existing points already sampled
            existing_losses: the existing losses already sampled.

        Returns:
            The sampled batch.
        """

    def sample(
        self,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Sample from the search space.

        This method returns an array of `batch_size` points belonging to the search space.

        Args:
            search_space: an object containing the details of the parameter search space
            existing_points: the parameters already sampled
            existing_losses: the loss corresponding to the sampled parameters

        Returns:
            the sampled parameters (an array of shape `(self.batch_size, search_space.dims)`)
        """
        sample_batch_without_duplicates = remove_duplicates(
            self.max_deduplication_passes
        )(self.sample_batch)
        return sample_batch_without_duplicates(
            self.batch_size, search_space, existing_points, existing_losses
        )
