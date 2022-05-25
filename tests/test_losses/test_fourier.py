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
"""This module contains tests for the Fourier loss."""
import numpy as np

from black_it.loss_functions.fourier import (
    FourierLoss,
    gaussian_low_pass_filter,
    ideal_low_pass_filter,
)


def test_fourier_ideal_low_pass() -> None:
    """Test the Fourier loss with the ideal low-pass filter."""
    np.random.seed(11)
    series_real = np.sin(np.linspace(0, 50, 1000))[:, None]
    series_sim = series_real + np.random.normal(0, 0.1, series_real.shape)
    euclidean_loss = np.sqrt(np.sum((series_sim - series_real) ** 2))

    # check that for no filter (f=1.0) this loss is approximately equivalent to
    # the Euclidean loss and that with increasingly aggressive filters (up to
    # f=0.01) the loss goes towards zero.
    expected_losses = [euclidean_loss, 2.23, 0.97, 0.27]
    for i, f in enumerate([1.0, 0.5, 0.1, 0.01]):
        loss_func = FourierLoss(f=f, frequency_filter=ideal_low_pass_filter)
        loss = loss_func.compute_loss(series_sim[None, :, :], series_real)
        assert np.isclose(expected_losses[i], loss, atol=0.01)


def test_fourier_gaussian_low_pass() -> None:
    """Test the Fourier loss with the gaussian low-pass filter."""
    np.random.seed(11)
    series_real = np.sin(np.linspace(0, 50, 1000))[:, None]
    series_sim = series_real + np.random.normal(0, 0.1, series_real.shape)

    # check that with increasingly aggressive filters (up to f=0.01) the loss
    # goes towards zero.
    expected_losses = [2.75, 0.95, 0.27]
    for i, f in enumerate([1.0, 0.1, 0.01]):
        loss_func = FourierLoss(f=f, frequency_filter=gaussian_low_pass_filter)
        loss = loss_func.compute_loss(series_sim[None, :, :], series_real)
        assert np.isclose(expected_losses[i], loss, atol=0.01)
