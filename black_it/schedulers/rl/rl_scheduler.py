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

"""This module implements the 'RLScheduler' scheduler."""
import threading
from queue import Queue
from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
from numpy._typing import NDArray

from black_it.samplers.base import BaseSampler
from black_it.samplers.halton import HaltonSampler
from black_it.schedulers.base import BaseScheduler
from black_it.schedulers.rl.agents.base import Agent
from black_it.schedulers.rl.envs.base import CalibrationEnv


class RLScheduler(BaseScheduler):  # pylint: disable=too-many-instance-attributes
    """
    This class implement a RL-based scheduler.

    It is agnostic wrt the RL algorithm being used.
    """

    def __init__(
        self,
        samplers: Sequence[BaseSampler],
        agent: Agent,
        env: CalibrationEnv,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize the scheduler."""
        self._original_samplers = samplers
        new_samplers, self._halton_sampler_id = self._add_or_get_bootstrap_sampler(
            samplers
        )

        self._agent = agent
        self._env = env

        super().__init__(new_samplers, random_state)

        self._in_queue: Queue = self._env._out_queue  # pylint: disable=protected-access
        self._out_queue: Queue = self._env._in_queue  # pylint: disable=protected-access

        self._best_param: Optional[float] = None
        self._best_loss: Optional[float] = None

        self._agent_thread: Optional[threading.Thread] = None
        self._stopped: bool = True

    def _set_random_state(self, random_state: Optional[int]) -> None:
        """Set the random state (private use)."""
        super()._set_random_state(random_state)
        for sampler in self.samplers:
            sampler.random_state = self._get_random_seed()
        self._agent.random_state = self._get_random_seed()
        self._env.reset(seed=self._get_random_seed())

    @classmethod
    def _add_or_get_bootstrap_sampler(
        cls, samplers: Sequence[BaseSampler]
    ) -> Tuple[Sequence[BaseSampler], int]:
        """
        Add or retrieve a sampler for bootstrapping.

        Many samplers do require some "bootstrapping" of the calibration process, i.e. a set of parameters
          whose loss has been already evaluated, e.g. samplers based on ML surrogates or on evolutionary approaches.
          Therefore, this scheduler must guarantee that the first proposed sampler is one that does not need previous
          model evaluations in input. One of such samplers is the Halton sampler

        Therefore, this function checks that the HaltonSampler is present in the set of samplers. If so, it returns
          the same set of samplers, and the index corresponding to that sampler in the sequence. Otherwise, a new
          instance of HaltonSampler is added to the list as first element.

        Args:
            samplers: the list of available samplers

        Returns:
            The pair (new_samplers, halton_sampler_id).
        """
        sampler_types = {type(s): i for i, s in enumerate(samplers)}
        if HaltonSampler in sampler_types:
            return samplers, sampler_types[HaltonSampler]

        new_sampler = HaltonSampler(batch_size=1)
        return tuple(list(samplers) + cast(List[BaseSampler], [new_sampler])), len(
            samplers
        )

    def _train(self) -> None:
        """Run the training loop."""
        state = self._env.reset()
        while not self._stopped:
            # Get the action chosen by the agent
            action = self._agent.policy(state)
            # Interact with the environment
            next_state, reward, _, _, _ = self._env.step(action)
            # Learn from interaction
            self._agent.learn(state, action, reward, next_state)
            state = next_state

    def start_session(self) -> None:
        """Set up the scheduler for a new session."""
        if not self._stopped:
            raise ValueError("cannot start session: the session has already started")
        self._stopped = False
        self._agent_thread = threading.Thread(target=self._train)
        self._agent_thread.start()

    def get_next_sampler(self) -> BaseSampler:
        """Get the next sampler."""
        if self._best_loss is None:
            # first call, return halton sampler
            return self.samplers[self._halton_sampler_id]
        chosen_sampler_id = self._in_queue.get()
        chosen_sampler = self.samplers[chosen_sampler_id]
        return chosen_sampler

    def update(
        self,
        batch_id: int,
        new_params: NDArray[np.float64],
        new_losses: NDArray[np.float64],
        new_simulated_data: NDArray[np.float64],
    ) -> None:
        """Update the RL scheduler."""
        best_new_loss = float(np.min(new_losses))
        if self._best_loss is None:
            self._best_loss = best_new_loss
            self._best_param = new_params[np.argmin(new_losses)]
            self._env._curr_best_loss = (  # pylint: disable=protected-access
                best_new_loss
            )
            return
        if best_new_loss < cast(float, self._best_loss):
            self._best_loss = best_new_loss
            self._best_param = new_params[np.argmin(new_losses)]

        self._out_queue.put((self._best_param, self._best_loss))

    def end_session(self) -> None:
        """Tear down the scheduler at the end of the session."""
        if self._stopped:
            raise ValueError("cannot start session: the session has not started yet")
        self._stopped = True
        self._out_queue.put(None)
        cast(threading.Thread, self._agent_thread).join()
