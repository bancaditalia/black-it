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
"""This module contains tests for the particle swarm sampler."""
import numpy as np
import pytest
from numpy.typing import NDArray

from black_it.samplers.particle_swarm import ParticleSwarmSampler
from black_it.search_space import SearchSpace

expected_params = np.array(
    [
        [0.64, 0.27],
        [0.04, 0.02],
        [0.81, 0.91],
        [0.61, 0.73],
        [0.64, 0.65],
        [0.32, 0.0],
        [1.0, 0.43],
        [0.78, 0.41],
        [0.61, 0.95],
        [0.55, 0.0],
        [1.0, 0.0],
        [0.91, 0.1],
        [0.56, 1.0],
        [0.7, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.49, 0.98],
        [0.79, 0.0],
        [1.0, 0.0],
        [0.99, 0.0],
        [0.43, 0.86],
        [0.76, 0.0],
        [1.0, 0.0],
        [0.88, 0.0],
        [0.37, 0.64],
        [0.64, 0.0],
        [1.0, 0.0],
        [0.72, 0.0],
        [0.33, 0.42],
        [0.42, 0.0],
        [1.0, 0.0],
        [0.52, 0.0],
        [0.27, 0.2],
        [0.17, 0.0],
        [0.97, 0.0],
        [0.3, 0.0],
        [0.21, 0.0],
        [0.0, 0.0],
        [0.85, 0.0],
        [0.08, 0.0],
    ]
)


@pytest.mark.parametrize("global_minimum_across_samplers", [True, False])
def test_particle_swarm_2d(global_minimum_across_samplers: bool) -> None:
    """Test the particle swarm sampler in 2d."""

    def target_loss(xy: NDArray[np.float64]) -> float:
        return 5 * (xy[0] ** 2 + xy[1] ** 2)

    points = np.zeros((0, 2))
    losses = np.array([])

    param_grid = SearchSpace(
        parameters_bounds=np.array([[0, 1], [0, 1]]).T,
        parameters_precision=np.array([0.01, 0.01]),
        verbose=False,
    )

    batch_size = 4
    nb_iterations = 10
    sampler = ParticleSwarmSampler(
        batch_size=batch_size,
        random_state=0,
        global_minimum_across_samplers=global_minimum_across_samplers,
    )

    sampled_params = np.zeros((0, 2))

    for _ in range(nb_iterations):
        current_positions = sampler.sample(param_grid, points, losses)
        current_losses = [target_loss(point) for point in current_positions]

        points = np.concatenate([points, current_positions])
        losses = np.concatenate([losses, current_losses])

        sampled_params = np.concatenate((sampled_params, current_positions))

    assert np.allclose(expected_params, sampled_params)
