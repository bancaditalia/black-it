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
"""This module contains tests for the cors sampler."""
import numpy as np
from numpy.typing import NDArray

from black_it.samplers.cors import CORSSampler
from black_it.search_space import SearchSpace

expected_params = np.array([[3.95, -4.43], [-9.18, -9.67], [6.27, 8.26], [2.13, 4.59]])


def test_cors_2d() -> None:
    """Test the CORS sampler, 2d."""
    # construct a fake grid of evaluated losses
    # define 4 initial points available to the sampler
    all_points = np.array([[0.0, 5.0], [5.0, 0.0], [0.0, -5.0], [-5.0, 0.0]])

    # initial values for the losses
    def loss_function(parameters: NDArray) -> float:
        return parameters[0] ** 2 + parameters[1] ** 2

    all_losses = np.array([loss_function(param) for param in all_points])

    sampler = CORSSampler(batch_size=4, random_state=0, max_samples=4, verbose=False)

    param_grid = SearchSpace(
        parameters_bounds=np.array([[-10, 10], [-10, 10]]).T,
        parameters_precision=np.array([0.01, 0.01]),
        verbose=False,
    )
    new_params = sampler.sample(param_grid, all_points, all_losses)

    assert np.allclose(expected_params, new_params)
