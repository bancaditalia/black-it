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
"""This module contains tests for the xgboost sampler."""
import numpy as np

from black_it.samplers.xgboost import XGBoostSampler
from black_it.search_space import SearchSpace

expected_params = np.array([[0.53, 0.66], [0.77, 0.03], [0.02, 0.14], [0.09, 0.64]])


def test_xgboost_2d() -> None:
    """Test the xgboost sampler, 2d."""
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

    sampler = XGBoostSampler(batch_size=4, random_state=0)
    param_grid = SearchSpace(
        parameters_bounds=np.array([[0, 1], [0, 1]]).T,
        parameters_precision=np.array([0.01, 0.01]),
        verbose=False,
    )
    new_params = sampler.sample(param_grid, xys, losses)

    assert np.allclose(expected_params, new_params)
