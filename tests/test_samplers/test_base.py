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
"""This module contains tests for the samplers.base module."""
import numpy as np
from numpy.typing import NDArray

from black_it.samplers.base import BaseSampler
from black_it.search_space import SearchSpace


def test_find_and_get_duplicates() -> None:
    """Test the find_and_get_duplicates method."""
    existing_points = np.array([[0, 1, 2], [0, 1, 2], [3, 4, 5]])
    new_points = np.array(
        [[0, 1, 2], [0, 1, 2], [3, 4, 5], [9, 10, 11], [12, 13, 14], [12, 13, 14]]
    )

    BaseSampler.__abstractmethods__ = frozenset()

    sampler = BaseSampler(batch_size=1)  # type: ignore  # pylint: disable=abstract-class-instantiated
    duplicates = sampler.find_and_get_duplicates(new_points, existing_points)

    assert duplicates == [0, 1, 2, 4, 5]


class TestSetRandomState:  # pylint: disable=attribute-defined-outside-init,too-many-instance-attributes
    """Test 'BaseSampler.random_state' setter."""

    class MyCustomSampler(BaseSampler):
        """
        Custom sampler for testing purposes.

        Test that two consecutive samples with the same give the same result.
        """

        def single_sample(
            self,
            seed: int,
            search_space: SearchSpace,
            existing_points: NDArray[np.float64],
            existing_losses: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            """Sample a single parameter."""
            np.random.seed(seed)
            return np.random.rand(3)

    def setup(self) -> None:
        """Set up the tests."""
        self.bounds = [[0.10, 0.10, 0.10], [1.00, 1.00, 1.00]]
        self.bounds_step = [0.01, 0.01, 0.01]
        self.default_seed = 42
        self.batch_size = 32
        self.sampler = TestSetRandomState.MyCustomSampler(
            self.batch_size, random_state=self.default_seed
        )
        self.search_space = SearchSpace(self.bounds, self.bounds_step, False)
        # if SearchSpace has been successfully constructed we are assured that
        # bounds[0] and bounds[1] are of the same length.
        self.existing_points = np.zeros((0, len(self.bounds[0])))
        self.existing_losses = np.zeros(0)

    def test_set_random_state_gives_same_result(self) -> None:
        """Test sampler gives same result after setting random state."""
        seed = 11
        self.sampler.random_state = seed
        expected_result_1 = self.sampler.sample(
            self.search_space, self.existing_points, self.existing_losses
        )
        expected_result_2 = self.sampler.sample(
            self.search_space, self.existing_points, self.existing_losses
        )
        self.sampler.random_state = seed
        actual_result_1 = self.sampler.sample(
            self.search_space, self.existing_points, self.existing_losses
        )
        actual_result_2 = self.sampler.sample(
            self.search_space, self.existing_points, self.existing_losses
        )
        assert (expected_result_1 == actual_result_1).all()
        assert (expected_result_2 == actual_result_2).all()

    def test_set_random_state_works_correctly(self) -> None:
        """Test 'BaseSampler.set_random_state' setter actually sets the internal random state."""
        assert self.sampler.random_state == self.default_seed
        self.sampler.random_state = 1
        assert self.sampler.random_state == 1
