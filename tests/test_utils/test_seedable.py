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

"""This module contains tests for the utils/seedable.py module."""

import hypothesis
from hypothesis import settings
from hypothesis.strategies import integers

from black_it.utils.seedable import BaseSeedable


def test_seedable_instantiation() -> None:
    """Test instantiation of the 'BaseSeedable' class."""
    BaseSeedable()


def test_random_state_default() -> None:
    """Test that default random state is None."""
    seedable = BaseSeedable()
    assert seedable.random_state is None


@settings(deadline=None)
@hypothesis.given(integers(min_value=0))
def test_random_state_setters(random_seed: int) -> None:
    """Test that setting random state via constructor or via property setter is the same."""
    # first seedable's random state is set via constructor
    seedable1 = BaseSeedable(random_state=random_seed)
    # second seedable's random state is set via property setter
    seedable2 = BaseSeedable()
    seedable2.random_state = random_seed

    assert seedable1.random_state == seedable2.random_state
    assert seedable1.random_generator.random() == seedable2.random_generator.random()


@settings(deadline=None)
@hypothesis.given(integers(min_value=0))
def test_consecutive_random_state_sets(random_seed: int) -> None:
    """Test that setting random state multiple times works as expected."""
    nb_iterations = 5000
    seedable = BaseSeedable()

    # reset random state, first time
    seedable.random_state = random_seed
    expected_values_1 = [
        seedable.random_generator.random() for _ in range(nb_iterations)
    ]

    # reset random state, second time
    seedable.random_state = random_seed
    expected_values_2 = [
        seedable.random_generator.random() for _ in range(nb_iterations)
    ]

    assert expected_values_1 == expected_values_2


@settings(deadline=None)
@hypothesis.given(integers(min_value=0))
def test_get_random_seed(random_seed: int) -> None:
    """Test the method 'get_random_seed'."""
    seedable = BaseSeedable()
    nb_iterations = 1000

    seedable.random_state = random_seed
    expected_values_1 = [seedable._get_random_seed() for _ in range(nb_iterations)]

    seedable.random_state = random_seed
    expected_values_2 = [seedable._get_random_seed() for _ in range(nb_iterations)]

    assert expected_values_1 == expected_values_2
    assert all(0 <= value <= 2**32 - 1 for value in expected_values_1)
