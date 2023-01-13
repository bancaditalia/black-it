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
import pytest
import xgboost as xgb

from black_it.calibrator import Calibrator
from black_it.loss_functions.msm import MethodOfMomentsLoss
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.xgboost import XGBoostSampler
from black_it.search_space import SearchSpace

from ..fixtures.test_models import BH4  # type: ignore

expected_params = np.array([[0.24, 0.26], [0.26, 0.02], [0.08, 0.24], [0.15, 0.15]])


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


def test_clip_losses() -> None:
    """Test the xgboost _clip_losses method."""
    true_params = [
        0.0,  # g1
        0.0,  # b1
        0.9,  # g2
        0.2,  # b2
        0.9,  # g3
        -0.2,  # b3
        1.01,  # g4
        0.01,
    ]  # b4

    parameter_bounds = [
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0],  # lower bounds
        [0.1, 0.1, 1.0, 1.0, 1.0, 0.0, 1.1, 1.0],
    ]  # upper bounds

    precisions = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    target_series = BH4(true_params, N=1000, seed=0)

    halton = HaltonSampler(batch_size=3)
    xgboost = XGBoostSampler(batch_size=3)

    loss = MethodOfMomentsLoss()

    cal = Calibrator(
        real_data=target_series,
        samplers=[halton, xgboost],
        loss_function=loss,
        model=BH4,
        parameters_bounds=parameter_bounds,
        parameters_precision=precisions,
        ensemble_size=3,
        random_state=0,
    )

    # the calibration breaks due to losses exceeding the limits of float32

    with pytest.raises(
        xgb.core.XGBoostError,
        match=r"Label contains NaN, infinity or a value too large",
    ):
        _, losses = cal.calibrate(1)
