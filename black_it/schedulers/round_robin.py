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

"""This module implements the 'RoundRobinScheduler' scheduler."""

import numpy as np
from numpy._typing import NDArray

from black_it.samplers.base import BaseSampler
from black_it.schedulers.base import BaseScheduler


class RoundRobinScheduler(BaseScheduler):
    """This class implement a simple round-robin sampler scheduler.

    The round-robin scheduler takes in input a list of samplers [S_0, S_1, ..., S_{n-1}],
      and, at batch i, it proposes the (i % n)-th sampler.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the round-robin scheduler."""
        super().__init__(*args, **kwargs)

        self._batch_id = 0

    def get_next_sampler(self) -> BaseSampler:
        """Get the next sampler."""
        return self.samplers[self._batch_id % len(self.samplers)]

    def update(
        self,
        batch_id: int,  # noqa: ARG002
        new_params: NDArray[np.float64],  # noqa: ARG002
        new_losses: NDArray[np.float64],  # noqa: ARG002
        new_simulated_data: NDArray[np.float64],  # noqa: ARG002
    ) -> None:
        """Update the state of the scheduler after each batch."""
        self._batch_id += 1
