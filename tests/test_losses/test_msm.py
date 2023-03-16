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
"""This module contains tests for the MSM loss."""
import re

import numpy as np
import pytest
from numpy.typing import NDArray

from black_it.loss_functions.msm import MethodOfMomentsLoss


def test_msm_default() -> None:
    """Test the 'method of moments' loss."""
    expected_loss = 2.830647081075395

    np.random.seed(11)
    # ensemble size 2, time series length 100, number of variables 3
    series_sim = np.random.normal(0, 1, (2, 100, 3))
    series_real = np.random.normal(0, 1, (100, 3))

    loss_func = MethodOfMomentsLoss()

    loss = loss_func.compute_loss(series_sim, series_real)

    assert np.isclose(expected_loss, loss)


def test_msm_default_calculator_custom_covariance_matrix() -> None:
    """Test the MSM loss when the covariance matrix is not None."""
    expected_loss = 16.49853079135471

    np.random.seed(11)
    series_sim = np.random.normal(0, 1, (2, 100, 3))
    series_real = np.random.normal(0, 1, (100, 3))

    random_mat = np.random.rand(18, 18)
    covariance_matrix = random_mat.T.dot(random_mat)

    loss_func = MethodOfMomentsLoss(covariance_mat=covariance_matrix)

    loss = loss_func.compute_loss(series_sim, series_real)

    assert np.isclose(expected_loss, loss)


def test_msm_default_calculator_non_symmetric_covariance_matrix() -> None:
    """Test the MSM loss raises error when the provided matrix is not symmetric."""
    with pytest.raises(
        ValueError,
        match="the provided covariance matrix is not valid as it is not a symmetric matrix",
    ):
        MethodOfMomentsLoss(covariance_mat=np.random.rand(2, 3))


def test_msm_default_calculator_wrong_shape_covariance_matrix() -> None:
    """Test the MSM loss raises error when the covariance matrix has wrong shape."""
    dimension = 20
    random_mat = np.random.rand(dimension, dimension)
    wrong_covariance_matrix = random_mat.T.dot(random_mat)
    with pytest.raises(
        ValueError,
        match=f"the provided covariance matrix is not valid as it has a wrong shape: expected 18, got {dimension}",
    ):
        MethodOfMomentsLoss(covariance_mat=wrong_covariance_matrix)


def test_msm_custom_calculator() -> None:
    """Test the 'method of moments' loss with a custom calculator."""
    expected_loss = 1.0

    series_sim = np.array([[0, 1]]).T
    series_real = np.array([[1, 2]]).T

    def custom_moment_calculator(time_series: NDArray) -> NDArray:
        moments = np.array([np.mean(time_series)])
        return moments

    loss_func = MethodOfMomentsLoss(moment_calculator=custom_moment_calculator)
    loss = loss_func.compute_loss(series_sim[None, :, :], series_real)
    assert np.isclose(expected_loss, loss)


def test_msm_custom_calculator_wrong_shape_covariance_matrix() -> None:
    """Test the 'method of moments' loss with a custom calculator and a custom covariance of the wrong shape."""
    series_sim = np.array([[0, 1]]).T
    series_real = np.array([[1, 2]]).T

    def custom_moment_calculator(time_series: NDArray) -> NDArray:
        moments = np.array([np.mean(time_series)])
        return moments

    dimension = 3
    random_mat = np.random.rand(dimension, dimension)
    wrong_covariance_matrix = random_mat.T.dot(random_mat)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The size of the covariance matrix (3) and the number of moments (1) should be identical"
        ),
    ):
        loss_func = MethodOfMomentsLoss(
            moment_calculator=custom_moment_calculator,
            covariance_mat=wrong_covariance_matrix,
        )
        loss_func.compute_loss(series_sim[None, :, :], series_real)
