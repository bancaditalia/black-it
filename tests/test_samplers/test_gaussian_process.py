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
"""This module contains tests for the Gaussian process sampler."""
from typing import Optional, Tuple, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from black_it.samplers.gaussian_process import GaussianProcessSampler, _AcquisitionTypes
from black_it.search_space import SearchSpace


class TestGaussianProcess2D:  # pylint: disable=attribute-defined-outside-init
    """Test GaussianProcess sampling."""

    def setup(self) -> None:
        """Set up the test."""
        self.xys, self.losses = self._construct_fake_grid()

    @classmethod
    def _construct_fake_grid(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Construct a fake grid of evaluated losses."""
        xs = np.linspace(0, 1, 6)
        ys = np.linspace(0, 1, 6)
        xys_list = []
        losses_list = []

        for x in xs:
            for y in ys:
                xys_list.append([x, y])
                losses_list.append(x**2 + y**2)
        return np.asarray(xys_list), np.asarray(losses_list)

    @pytest.mark.parametrize(
        "acquisition,optimize_restarts,expected_params",
        [
            [
                "mean",
                1,
                np.array([[0.01, 0.0], [0.01, 0.01], [0.02, 0.0], [0.01, 0.02]]),
            ],
            [
                "mean",
                5,
                np.array([[0.01, 0.0], [0.01, 0.01], [0.02, 0.0], [0.01, 0.02]]),
            ],
            [
                "expected_improvement",
                1,
                None,  # skip this due to reproducibility issue
            ],
            [
                "expected_improvement",
                5,
                None,  # skip this due to reproducibility issue
            ],
        ],
    )
    def test_gaussian_process_2d(
        self,
        acquisition: str,
        optimize_restarts: int,
        expected_params: Optional[NDArray],
    ) -> None:
        """Test the Gaussian process sampler, 2d."""
        sampler = GaussianProcessSampler(
            batch_size=4,
            internal_seed=0,
            optimize_restarts=optimize_restarts,
            acquisition=acquisition,
        )
        param_grid = SearchSpace(
            parameters_bounds=np.array([[0, 1], [0, 1]]).T,
            parameters_precision=np.array([0.01, 0.01]),
            verbose=False,
        )
        new_params = sampler.sample(param_grid, self.xys, self.losses)

        if expected_params is None:
            pytest.skip(
                "The test with acquisition=expected_improvement is temporarily "
                "skipped due to reproducibility issues."
            )
        assert np.allclose(cast(NDArray, expected_params), new_params)


def test_gaussian_process_single_sample_raises_not_implemented_error() -> None:
    """Test that 'GaussianProcessSampler.single_sample' raises NotImplementedError."""
    sampler = GaussianProcessSampler(4)
    with pytest.raises(
        NotImplementedError,
        match="single_sample is not supported by GaussianProcessSampler",
    ):
        sampler.single_sample(MagicMock(), MagicMock(), MagicMock(), MagicMock())


def test_gaussian_process_sample_warning_too_large_dataset() -> None:
    """Test GaussianProcessSampler.sample wiht too many existing points."""
    sampler = GaussianProcessSampler(batch_size=4, internal_seed=0, acquisition="mean")
    param_grid = SearchSpace(
        parameters_bounds=np.array([[0, 1], [0, 1]]).T,
        parameters_precision=np.array([0.01, 0.01]),
        verbose=False,
    )
    xs = np.linspace(0, 1, 1000)
    ys = np.linspace(0, 1, 3)
    existing_points = xs.reshape((-1, 1)).dot(ys.reshape((1, -1)))
    losses = np.zeros(existing_points.shape)
    with pytest.raises(
        RuntimeWarning,
        match="Standard GP evaluations can be expensive "
        "for large datasets, consider implementing "
        "a sparse GP",
    ):
        sampler.sample(param_grid, existing_points, losses)


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
