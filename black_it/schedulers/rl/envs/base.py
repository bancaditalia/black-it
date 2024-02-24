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

"""This package contains implementations of Gym environments for the RL Scheduler."""
from __future__ import annotations

from abc import ABC, abstractmethod
from queue import Queue
from typing import TYPE_CHECKING, Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, RenderFrame
from gymnasium.spaces import Discrete

from black_it.utils.base import _assert

if TYPE_CHECKING:
    from numpy._typing import NDArray


class CalibrationEnv(gym.Env[ObsType, np.int64], ABC):
    """A Gym environment that wraps the calibration task."""

    def __init__(self, nb_samplers: int) -> None:
        """Initialize the environment."""
        self._nb_samplers = nb_samplers
        self._out_queue: Queue = Queue()
        self._in_queue: Queue = Queue()

        self._curr_best_loss: float | None = None

        self.action_space = Discrete(self._nb_samplers)

    @abstractmethod
    def reset_state(self) -> ObsType:
        """Get the initial state."""

    @abstractmethod
    def get_reward(self, best_param: NDArray, best_loss: float) -> float:
        """Get the current reward."""

    @abstractmethod
    def get_next_observation(self) -> ObsType:
        """Get the next observation."""

    def reset(
        self,
        *,
        seed: int | None = None,  # noqa: ARG002
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment."""
        return self.reset_state(), {}

    def step(
        self,
        action: np.int64,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Do a step."""
        _assert(self.action_space.contains(action))
        self._out_queue.put(action)
        result = self._in_queue.get()
        if result is None:
            # calibration ended
            return self.reset_state(), 0.0, False, True, {}
        best_param, best_loss = result
        reward = self.get_reward(best_param, best_loss)
        next_obs = self.get_next_observation()
        return next_obs, reward, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Render the environment (not implemented)."""
        raise NotImplementedError
