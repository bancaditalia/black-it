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
"""This module contains tests for the random forest sampler."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from black_it.samplers.random_forest import RandomForestSampler
from black_it.search_space import SearchSpace

expected_params = np.array([[0.0, 0.07], [0.04, 0.03], [0.05, 0.01], [0.09, 0.01]])


def test_random_forest_2d() -> None:
    """Test the random forest sampler, 2d."""
    # construct a fake grid of evaluated losses
    xs = np.linspace(0, 1, 6)
    ys = np.linspace(0, 1, 6)
    xys_list = []
    losses_list = []

    for x in xs:
        for y in ys:
            xys_list.append([x, y])
            losses_list.append(x**2 + y**2)

    xys = np.array(xys_list)
    losses = np.array(losses_list)

    sampler = RandomForestSampler(batch_size=4, internal_seed=0)
    param_grid = SearchSpace(
        parameters_bounds=np.array([[0, 1], [0, 1]]).T,
        parameters_precision=np.array([0.01, 0.01]),
        verbose=False,
    )
    new_params = sampler.sample(param_grid, xys, losses)

    assert np.allclose(expected_params, new_params)


def test_random_forest_single_sample_raises_error() -> None:
    """Test that 'RandomForestSampler.single_sample' raises NotImplementedError."""
    sampler = RandomForestSampler(batch_size=4, internal_seed=0)
    with pytest.raises(
        NotImplementedError,
        match="for RandomForestSampler the parallelization is hard coded in sample",
    ):
        sampler.single_sample(MagicMock(), MagicMock(), MagicMock(), MagicMock())


def test_random_forest_candidate_pool_size() -> None:
    """Test candidate pool size parameter."""
    batch_size = 8

    sampler1 = RandomForestSampler(batch_size=batch_size)
    # default is batch_size * 1000
    assert sampler1.candidate_pool_size == batch_size * 1000

    expected_candidate_pool_size = 32
    sampler2 = RandomForestSampler(
        batch_size=batch_size, candidate_pool_size=expected_candidate_pool_size
    )
    assert sampler2.candidate_pool_size == expected_candidate_pool_size
