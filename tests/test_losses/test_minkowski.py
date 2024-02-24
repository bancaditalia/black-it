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
"""This module contains tests for the Minkowski loss."""
import numpy as np

from black_it.loss_functions.minkowski import MinkowskiLoss


def test_minkowski() -> None:
    """Test the Minkowski loss."""
    expected_losses = [2.5, 1.8027756377319946, 1.6355331550942949]

    series_sim = np.array([np.array([[2, 0, 0]]).T, np.array([[1, 0, 0]]).T])
    series_real = np.array([[0, 1, 0]]).T

    for i, p in enumerate([1, 2, 3]):
        loss_func = MinkowskiLoss(p=p)

        loss = loss_func.compute_loss(series_sim, series_real)

        assert np.isclose(expected_losses[i], loss)
