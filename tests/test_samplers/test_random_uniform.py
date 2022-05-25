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
"""This module contains tests for the random uniform sampler."""
import numpy as np

from black_it.samplers.random_uniform import RandomUniformSampler
from black_it.search_space import SearchSpace

expected_params = np.array(
    [
        [0.85, 0.64],
        [0.47, 0.51],
        [0.84, 0.26],
        [0.81, 0.08],
        [0.73, 0.95],
        [0.67, 0.81],
        [0.44, 0.54],
        [0.95, 0.63],
    ]
)


def test_random_uniform() -> None:
    """Test the random uniform sampler."""
    sampler = RandomUniformSampler(batch_size=8, internal_seed=0)

    param_grid = SearchSpace(
        parameters_bounds=np.array([[0, 1], [0, 1]]).T,
        parameters_precision=np.array([0.01, 0.01]),
        verbose=False,
    )
    new_params = sampler.sample(param_grid, np.zeros((0, 2)), np.zeros((0, 2)))

    assert np.allclose(expected_params, new_params)


def test_random_uniform_uniqueness() -> None:
    """Test the random uniform uniqueness."""
    existing_points = expected_params[:1]
    expected_params_unique = np.vstack((np.array([[0.72, 0.33]]), expected_params[1:]))

    sampler = RandomUniformSampler(batch_size=8, internal_seed=0)

    param_grid = SearchSpace(
        parameters_bounds=np.array([[0, 1], [0, 1]]).T,
        parameters_precision=np.array([0.01, 0.01]),
        verbose=False,
    )
    new_params = sampler.sample(param_grid, existing_points, np.zeros((1, 2)))

    assert np.allclose(expected_params_unique, new_params)
