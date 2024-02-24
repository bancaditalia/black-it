# Black-box ABM Calibration Kit (Black-it)
# Copyright (C) 2021-2024 Banca d'Italia
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

"""This module contains the definition of a 'seedable' base class."""
from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.random import default_rng

if TYPE_CHECKING:
    import numpy as np


class BaseSeedable:
    """BaseSeedable base class.

    This is the base class for all objects that need to keep a random state.

    In particular, the interface provides the following features:
    - it is able to accept a random seed from the client;
    - it allows the client reset the random seed;
    - it provides a getter to the internal random generator;
    - it allows to sample a random seed (in the range [0, 2^32 - 1]).
    """

    def __init__(
        self,
        random_state: int | None = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            random_state: the internal state of the sampler, fixing this numbers the object (e.g. a calibrator,
                a sampler, or a scheduler) behaves deterministically
        """
        self.__random_state: int | None
        self.__random_generator: np.random.Generator

        # this triggers the property setter
        self.random_state = random_state

    @property
    def random_state(self) -> int | None:
        """Get the random state."""
        return self.__random_state

    @random_state.setter
    def random_state(self, random_state: int | None) -> None:
        """Set the random state."""
        self._set_random_state(random_state)

    def _set_random_state(self, random_state: int | None) -> None:
        """Set the random state, private use."""
        self.__random_state = random_state
        self.__random_generator = default_rng(self.random_state)

    @property
    def random_generator(self) -> np.random.Generator:
        """Get the random generator."""
        return self.__random_generator

    def _get_random_seed(self) -> int:
        """Get new random seed from the current random generator."""
        return get_random_seed(self.__random_generator)


def get_random_seed(random_generator: np.random.Generator) -> int:
    """Get a random seed from a random generator.

    Sample an integer in the range [0, 2^32 - 1].

    Args:
        random_generator: the random generator to be used for sampling the random seed.

    Returns:
        the random seed.
    """
    return random_generator.integers(2**32 - 1)
