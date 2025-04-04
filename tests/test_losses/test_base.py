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

"""This module contains tests for the base loss_functions module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from black_it.loss_functions.base import BaseLoss

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class TestComputeLoss:
    """Tests for BaseLoss.compute_loss."""

    class MyCustomLoss(BaseLoss):
        """A custom loss function for testing purposes."""

        def __init__(
            self,
            loss_constant: float,
            coordinate_weights: NDArray | None = None,
            coordinate_filters: tuple[Callable | None, ...] | None = None,
        ) -> None:
            """Initialize the custom loss."""
            super().__init__(coordinate_weights, coordinate_filters)
            self.loss_constant = loss_constant

        def compute_loss_1d(
            self,
            sim_data_ensemble: NDArray[np.float64],
            real_data: NDArray[np.float64],  # noqa: ARG002
        ) -> float:
            """Compute the loss (constant)."""
            return float(np.sum(sim_data_ensemble) * self.loss_constant)

    loss_constant: float = 1.0
    nb_sim_ensemble = 3
    nb_sim_coords: int = 2
    nb_real_coords: int = 2
    nb_sim_rows: int = 10
    nb_real_rows: int = 10

    coordinate_weights: NDArray[np.float64] | None
    coordinate_filters: tuple[Callable | None, ...] | None

    # instance attributes
    loss: MyCustomLoss
    sim_data: NDArray[np.float64]
    real_data: NDArray[np.float64]

    def setup_method(self) -> None:
        """Set up the tests."""
        self.loss = TestComputeLoss.MyCustomLoss(
            self.loss_constant,
            self.coordinate_weights,
            self.coordinate_filters,
        )
        self.sim_data = np.ones(
            (self.nb_sim_ensemble, self.nb_sim_rows, self.nb_sim_coords),
        )
        self.real_data = np.ones((self.nb_real_rows, self.nb_real_coords))

    @property
    def nb_coordinate_weights(self) -> int:
        """Get the number of coordinate weights."""
        if self.coordinate_weights is None:
            return 0
        return len(self.coordinate_weights)

    @property
    def nb_coordinate_filters(self) -> int:
        """Get the number of coordinate filters."""
        if self.coordinate_filters is None:
            return 0
        return len(self.coordinate_filters)


class TestComputeLossWhenCoordWeightsIsNotNone(TestComputeLoss):
    """Test BaseLoss.compute_loss when coordinate weights is not None."""

    coordinate_weights = np.asarray([1.0, 2.0])
    coordinate_filters = None

    def test_run(self) -> None:
        """Test BaseLoss.compute_loss when coordinate weights is not None."""
        result = self.loss.compute_loss(self.sim_data, self.real_data)

        assert result == self.coordinate_weights[0] * np.sum(
            self.sim_data[:, :, 0],
        ) + self.coordinate_weights[1] * np.sum(self.sim_data[:, :, 1])


class TestComputeLossWhenNbCoordWeightsIsWrong(TestComputeLoss):
    """Test BaseLoss.compute_loss when number of coordinate weights does not match data."""

    coordinate_weights = np.asarray([1.0])
    coordinate_filters = None

    def test_run(self) -> None:
        """Test BaseLoss.compute_loss when number of coordinate weights does not match data."""
        expected_message_error = (
            "the length of coordinate_weights should be equal to the number of "
            f"coordinates, got {self.nb_coordinate_weights} and {self.nb_sim_coords}"
        )
        with pytest.raises(ValueError, match=expected_message_error):
            self.loss.compute_loss(self.sim_data, self.real_data)


class TestComputeLossWhenCoordFiltersIsNotNone(TestComputeLoss):
    """Test BaseLoss.compute_loss when coordinate filters is not None."""

    coordinate_weights = None
    coordinate_filters = (lambda x: x * 0, lambda x: x * 0)

    def test_run(self) -> None:
        """Test BaseLoss.compute_loss when coordinate filters is not None."""
        result = self.loss.compute_loss(self.sim_data, self.real_data)
        assert result == 0


class TestComputeLossWhenNbCoordFiltersIsWrong(TestComputeLoss):
    """Test BaseLoss.compute_loss when number of coordinate weights does not match data."""

    coordinate_weights = None
    coordinate_filters: tuple[Callable] = (lambda x: x,)

    def test_run(self) -> None:
        """Test BaseLoss.compute_loss when number of coordinate filters does not match data."""
        expected_message_error = (
            "the length of coordinate_filters should be equal to the number of "
            f"coordinates, got {self.nb_coordinate_filters} and {self.nb_sim_coords}"
        )
        with pytest.raises(ValueError, match=expected_message_error):
            self.loss.compute_loss(self.sim_data, self.real_data)
