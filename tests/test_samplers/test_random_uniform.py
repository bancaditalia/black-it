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
        [0.85, 0.17],
        [0.64, 0.82],
        [0.51, 0.65],
        [0.27, 0.92],
        [0.31, 0.5],
        [0.04, 0.61],
        [0.07, 0.98],
        [0.01, 0.73],
    ]
)


def test_random_uniform() -> None:
    """Test the random uniform sampler."""
    sampler = RandomUniformSampler(batch_size=8, random_state=0)

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

    # sampled_params_after_deduplication is the element that is resampled in place of expected_params[0].
    # We add expected_params[0] as existing point; and since the test is deterministic,
    # expected_params[0] will be sampled and detected as duplicate.
    # then, sampled_params_after_deduplication will be the next element to be sampled.
    element_after_deduplication = np.array([[0.63, 0.54]])
    expected_params_unique = np.vstack(
        (element_after_deduplication, expected_params[1:])
    )

    sampler = RandomUniformSampler(batch_size=8, random_state=0)

    param_grid = SearchSpace(
        parameters_bounds=np.array([[0, 1], [0, 1]]).T,
        parameters_precision=np.array([0.01, 0.01]),
        verbose=False,
    )
    new_params = sampler.sample(param_grid, existing_points, np.zeros((1, 2)))

    assert np.allclose(expected_params_unique, new_params)
