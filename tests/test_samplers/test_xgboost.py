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
"""This module contains tests for the xgboost sampler."""

import sys
from typing import cast

import numpy as np

from black_it.calibrator import Calibrator
from black_it.loss_functions.msm import MethodOfMomentsLoss
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.xgboost import XGBoostSampler
from black_it.search_space import SearchSpace
from examples.models.economics.brock_hommes import BH4

# This is a temporary workaround to make tests to run also on Windows and Mac.
#   See: https://github.com/bancaditalia/black-it/issues/49
if sys.platform == "win32":
    expected_params = np.array([[0.24, 0.26], [0.19, 0.11], [0.13, 0.22], [0.11, 0.05]])
elif sys.platform == "darwin":
    expected_params = np.array([[0.24, 0.26], [0.15, 0.08], [0.01, 0.03], [0.28, 0.21]])
else:
    expected_params = np.array([[0.24, 0.26], [0.37, 0.21], [0.43, 0.14], [0.11, 0.04]])

MAX_FLOAT32: float = cast("float", np.finfo(np.float32).max)
MIN_FLOAT32: float = cast("float", np.finfo(np.float32).min)
EPS_FLOAT32: float = cast("float", np.finfo(np.float32).eps)


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
    true_params = np.array(
        [
            0.0,  # g1
            0.0,  # b1
            0.9,  # g2
            0.2,  # b2
            0.9,  # g3
            -0.2,  # b3
            1.01,  # g4
            0.01,  # b4
        ],
    )

    parameter_bounds = [
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0],  # lower bounds
        [0.1, 0.1, 1.0, 1.0, 1.0, 0.0, 1.1, 1.0],  # upper bounds
    ]

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

    # verify that the calibration does not break,
    # it would without the call to _clip_losses
    _, losses = cal.calibrate(2)

    assert np.allclose(
        losses,
        np.array(
            [0.18388932, 0.58118863, 0.84728975, 0.87882275, 0.88818152, 1.2508034],
        ),
    )

    # verify that _clip_losses works as expected
    y = np.array([0.0, -1e40, 1e40])
    y2 = xgboost._clip_losses(y)  # noqa: SLF001

    assert (y2 == np.array([0.0, MIN_FLOAT32 + EPS_FLOAT32, MAX_FLOAT32 - EPS_FLOAT32])).all()
