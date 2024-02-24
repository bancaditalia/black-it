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

"""This module includes the implementation of a non-stationary epsilon-greedy agent."""
from __future__ import annotations

from typing import SupportsFloat, cast

import numpy as np

from black_it.schedulers.rl.agents.base import Agent


class MABEpsilonGreedy(Agent[int, int]):
    """Implementation of a MAB eps-greedy algorithm."""

    def __init__(
        self,
        n_actions: int,
        alpha: float,
        eps: float,
        initial_values: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        """Initialize the agent object.

        Args:
            n_actions: the number of actions
            alpha: the learning rate
            eps: the epsilon parameter
            initial_values: the initial value for the Q-function
            random_state: the random state
        """
        super().__init__(random_state=random_state)
        self.n_actions = n_actions
        self.actions_count = [0] * self.n_actions

        self.Q = [initial_values] * self.n_actions
        self.alpha = alpha
        self.eps = eps
        self.initial_values = initial_values

    def get_step_size(self, action: int) -> float:
        """Get the step size."""
        return 1 / self.actions_count[action] if self.alpha == -1 else self.alpha

    def learn(
        self,
        state: int,  # noqa: ARG002
        action: int,
        reward: SupportsFloat,
        next_state: int,  # noqa: ARG002
    ) -> None:
        """Learn from an agent-environment interaction timestep."""
        self.actions_count[action] += 1

        step_size = self.get_step_size(action)

        # do the learning
        self.Q[action] += step_size * (cast(float, reward) - self.Q[action])

    def policy(self, _obs: int) -> int:
        """Get the action for this observation."""
        best_action = np.argmax(self.Q)

        random_e = self.random_generator.random()
        if not random_e < self.eps:
            action = best_action
        else:
            options = np.arange(self.n_actions)
            action = self.random_generator.choice(options, 1)[0]

        return int(action)

    def reset(self) -> None:
        """Reset the agent."""
        self.Q = [0.0] * self.n_actions
        self.actions_count = [0] * self.n_actions
