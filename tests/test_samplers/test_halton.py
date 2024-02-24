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
"""This module contains tests for the Halton sampler."""
import numpy as np

from black_it.samplers.halton import HaltonSampler
from black_it.search_space import SearchSpace

expected_params = np.array(
    [
        [0.1, 0.39],
        [0.6, 0.72],
        [0.35, 0.17],
        [0.85, 0.5],
        [0.22, 0.84],
        [0.72, 0.28],
        [0.47, 0.61],
        [0.97, 0.95],
    ],
)


def test_halton_2d() -> None:
    """Test the Halton sampler, 2d."""
    sampler = HaltonSampler(batch_size=8, random_state=0)
    param_grid = SearchSpace(
        parameters_bounds=np.array([[0, 1], [0, 1]]).T,
        parameters_precision=np.array([0.01, 0.01]),
        verbose=False,
    )
    new_params = sampler.sample(param_grid, np.zeros((0, 2)), np.zeros((0, 2)))

    assert np.allclose(expected_params, new_params)


def test_halton_when_not_enough_primes() -> None:
    """Test Halton sampler when during a sampling there are no enough bases."""
    sampler = HaltonSampler(batch_size=8)
    nb_params = 10
    param_grid = SearchSpace(
        parameters_bounds=np.asarray([[0, 1]] * nb_params).T,
        parameters_precision=np.array([0.01] * nb_params),
        verbose=False,
    )
    sampler.sample(param_grid, np.zeros((0, nb_params)), np.zeros(0))
