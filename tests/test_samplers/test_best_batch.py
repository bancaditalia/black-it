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
"""This module contains tests for the best-batch sampler."""
import numpy as np
import pytest

from black_it.samplers.best_batch import BestBatchSampler
from black_it.search_space import SearchSpace

expected_params = np.array(
    [
        [0.43, 0.22],
        [0.41, 0.01],
        [0.03, 0.42],
        [0.2, 0.04],
        [0.23, 0.0],
        [0.05, 0.05],
        [0.0, 0.03],
        [0.03, 0.04],
    ],
)


def test_best_batch_2d() -> None:
    """Test the best-batch sampler, 2d."""
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

    sampler = BestBatchSampler(batch_size=8, random_state=0)
    param_grid = SearchSpace(
        parameters_bounds=np.array([[0, 1], [0, 1]]).T,
        parameters_precision=np.array([0.01, 0.01]),
        verbose=False,
    )
    new_params = sampler.sample(param_grid, xys, losses)

    assert np.allclose(expected_params, new_params)


def test_best_batch_clipping() -> None:
    """Test the best-batch clipping when sampling."""
    lower_bound, upper_bound = 0.499, 0.5
    sampler = BestBatchSampler(batch_size=8, random_state=0)
    param_grid = SearchSpace(
        parameters_bounds=np.array(
            [[lower_bound, upper_bound], [lower_bound, upper_bound]],
        ).T,
        parameters_precision=np.array([0.001, 0.001]),
        verbose=False,
    )

    existing_points = np.random.rand(8, 2)
    existing_losses = np.random.rand(8)

    new_params = sampler.sample(param_grid, existing_points, existing_losses)
    assert (new_params == lower_bound).any() or (new_params == upper_bound).any()


def test_best_batch_sample_requires_batch_size_existing_points() -> None:
    """Test that BestBatch.sample must be greater or equal than batch size."""
    nb_points = 8
    batch_size = 16
    existing_points = np.random.rand(nb_points, 2)
    existing_losses = np.random.rand(nb_points)
    param_grid = SearchSpace(
        parameters_bounds=np.array([[0, 1], [0, 1]]).T,
        parameters_precision=np.array([0.01, 0.01]),
        verbose=False,
    )
    sampler = BestBatchSampler(batch_size=batch_size, random_state=0)
    with pytest.raises(
        ValueError,
        match="best-batch sampler requires a number of "
        f"existing points which is at least the "
        f"batch size {batch_size}, got {len(existing_points)}",
    ):
        sampler.sample(param_grid, existing_points, existing_losses)
