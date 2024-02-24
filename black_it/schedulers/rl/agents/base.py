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

"""Base module for RL agents."""

from abc import ABC, abstractmethod
from typing import Generic, SupportsFloat

from gymnasium.core import ActType, ObsType

from black_it.utils.seedable import BaseSeedable


class Agent(Generic[ObsType, ActType], BaseSeedable, ABC):
    """Interface for RL agents."""

    @abstractmethod
    def learn(
        self,
        state: ObsType,
        action: ActType,
        reward: SupportsFloat,
        next_state: ObsType,
    ) -> None:
        """Learn from an agent-environment interaction timestep.

        Args:
            state: the observation from where the action was taken
            action: the chosen action
            reward: the observed reward
            next_state: the next observation
        """
        raise NotImplementedError

    @abstractmethod
    def policy(self, state: ObsType) -> ActType:
        """Get the next action.

        Args:
            state: the state from where to take the action

        Returns:
            the chosen action.
        """
        raise NotImplementedError
