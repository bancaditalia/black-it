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

"""This module defines the 'BaseScheduler' base class."""
from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from black_it.utils.seedable import BaseSeedable

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from black_it.samplers.base import BaseSampler


class BaseScheduler(BaseSeedable, ABC):
    """BaseScheduler interface.

    This is the base class for all schedulers.
    """

    def __init__(
        self,
        samplers: Sequence[BaseSampler],
        random_state: int | None = None,
    ) -> None:
        """Initialize the scheduler.

        Args:
            samplers: the list of samplers to be scheduled
            random_state: the random seed for the scheduler behaviour
        """
        # need to set __samplers first because _set_random_state requires samplers to be set
        self._samplers = tuple(samplers)
        BaseSeedable.__init__(self, random_state)

    @property
    def samplers(self) -> tuple[BaseSampler, ...]:
        """Get the sequence of samplers."""
        return self._samplers

    def _set_random_state(self, random_state: int | None) -> None:
        """Set the random state (private use)."""
        super()._set_random_state(random_state)
        for sampler in self.samplers:
            sampler.random_state = self._get_random_seed()

    def start_session(self) -> None:
        """Set up the scheduler for a new session.

        The default is a no-op.
        """

    @abstractmethod
    def get_next_sampler(self) -> BaseSampler:
        """Get the sampler to use for the next batch."""

    @abstractmethod
    def update(
        self,
        batch_id: int,
        new_params: NDArray[np.float64],
        new_losses: NDArray[np.float64],
        new_simulated_data: NDArray[np.float64],
    ) -> None:
        """Update the state of the scheduler after each batch.

        Args:
            batch_id: the batch id of the . Must be an integer equal or greater than 0.
            new_params: the new set of parameters sampled in this batch.
            new_losses: the new set of losses corresponding to the batch.
            new_simulated_data: the new set of simulated data, one for each sampled parameter.
        """

    def end_session(self) -> None:
        """Tear down the scheduler at the end of the session.

        The default is a no-op.
        """

    @contextlib.contextmanager
    def session(self) -> Generator[None, None, None]:
        """Start the session of the scheduler with a context manager."""
        self.start_session()
        yield
        self.end_session()
