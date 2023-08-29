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
"""This module contains tests for the Gaussian process sampler."""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from black_it.samplers.gaussian_process import GaussianProcessSampler, _AcquisitionTypes
from black_it.search_space import SearchSpace


class TestGaussianProcess2D:
    """Test GaussianProcess sampling."""

    def setup(self) -> None:
        """Set up the test."""
        self.xys, self.losses = self._construct_fake_grid()

    @classmethod
    def _construct_fake_grid(
        cls,
        n: int = 3,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Construct a fake grid of evaluated losses."""
        xs = np.linspace(0, 1, n)
        ys = np.linspace(0, 1, n)
        xys_list = []
        losses_list = []

        np.random.seed(0)
        for x in xs:
            for y in ys:
                x = x + np.random.normal(0, 1e-2)
                y = y + np.random.normal(0, 1e-2)
                xys_list.append([x, y])
                losses_list.append(x**2 + y**2)

        return np.asarray(xys_list), np.asarray(losses_list)

    @pytest.mark.parametrize(
        ("acquisition", "optimize_restarts", "expected_params"),
        [
            (
                "mean",
                1,
                np.array([[0.0, 0.01], [0.01, 0.01], [0.0, 0.02], [0.01, 0.0]]),
            ),
            (
                "mean",
                3,
                np.array([[0.0, 0.01], [0.01, 0.01], [0.0, 0.02], [0.01, 0.0]]),
            ),
            (
                "expected_improvement",
                1,
                np.array([[0.0, 0.01], [0.0, 0.02], [0.01, 0.01], [0.01, 0.0]]),
            ),
            (
                "expected_improvement",
                3,
                np.array([[0.0, 0.01], [0.0, 0.02], [0.01, 0.01], [0.01, 0.0]]),
            ),
        ],
    )
    def test_gaussian_process_2d(
        self,
        acquisition: str,
        optimize_restarts: int,
        expected_params: NDArray | None,
    ) -> None:
        """Test the Gaussian process sampler, 2d."""
        sampler = GaussianProcessSampler(
            batch_size=4,
            random_state=0,
            optimize_restarts=optimize_restarts,
            acquisition=acquisition,
        )
        param_grid = SearchSpace(
            parameters_bounds=np.array([[0, 1], [0, 1]]).T,
            parameters_precision=np.array([0.01, 0.01]),
            verbose=False,
        )
        new_params = sampler.sample(param_grid, self.xys, self.losses)

        assert np.allclose(cast(NDArray, expected_params), new_params)


def test_gaussian_process_sample_warning_too_large_dataset() -> None:
    """Test GaussianProcessSampler.sample wiht too many existing points."""
    sampler = GaussianProcessSampler(batch_size=4, random_state=0, acquisition="mean")
    param_grid = SearchSpace(
        parameters_bounds=np.array([[0, 1], [0, 1]]).T,
        parameters_precision=np.array([0.01, 0.01]),
        verbose=False,
    )
    # very high number of samples
    (
        xys,
        losses,
    ) = TestGaussianProcess2D._construct_fake_grid(  # noqa: SLF001
        n=23,
    )
    with pytest.warns(
        RuntimeWarning,
        match="Standard GP evaluations can be expensive "
        "for large datasets, consider implementing "
        "a sparse GP",
    ):
        sampler.sample(param_grid, xys, losses)


def test_gaussian_process_sample_wrong_acquisition() -> None:
    """Test GaussianProcessSampler initialization with wrong acquisition."""
    wrong_acquisition = "wrong_acquisition"
    with pytest.raises(
        ValueError,
        match="expected one of the following acquisition types: "
        rf"\[{' '.join(map(str, _AcquisitionTypes))}\], "
        f"got {wrong_acquisition}",
    ):
        GaussianProcessSampler(4, acquisition=wrong_acquisition)
