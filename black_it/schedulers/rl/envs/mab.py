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

"""This module defines the calibration gym environment for MAB algorithms."""

from numpy._typing import NDArray

from black_it.schedulers.rl.envs.base import CalibrationEnv


class MABCalibrationEnv(CalibrationEnv[int]):
    """A calibration environment for MAB algorithms."""

    def get_next_observation(self) -> int:  # noqa: PLR6301
        """Get the next observation."""
        return 0

    def reset_state(self) -> int:  # noqa: PLR6301
        """Get the initial state."""
        return 0

    def get_reward(
        self,
        best_param: NDArray,  # noqa: ARG002
        best_loss: float,
    ) -> float:
        """Get the reward."""
        if self._curr_best_loss is None:
            msg = "cannot get reward, curr_best_loss should be already set"
            raise ValueError(msg)
        reward = 0.0
        if best_loss < self._curr_best_loss:
            reward = (self._curr_best_loss - best_loss) / self._curr_best_loss
            self._curr_best_loss = best_loss
        return reward
