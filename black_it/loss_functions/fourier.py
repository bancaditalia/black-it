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

"""This module contains the implementation of the Fast Fourier Transform loss."""
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from black_it.loss_functions.base import BaseLoss
from black_it.utils.base import assert_

FrequencyFilter = Callable[[NDArray[np.complex128], float], NDArray[np.complex128]]
"""A filter that receives the signal in the frequency domain and returns its
filtered version. Used by FourierLoss constructor.

In this version, the filter supports a single parameter: no multi-band filtering
is supported yet.
"""


def ideal_low_pass_filter(
    signal_frequencies: NDArray[np.complex128],
    f: float,
) -> NDArray[np.complex128]:
    """Ideal low-pass filter.

    Args:
        signal_frequencies: the input signal transformed in the frequency
            domain.
        f: the fraction of frequencies to keep unchanged, for f=1 the filter is
            just the identity.

    Returns:
        the filtered frequencies
    """
    # number of low-frequency component to keep
    n = int(np.round(f * signal_frequencies.shape[0]))

    # ideal low-pass filter
    mask = np.zeros(signal_frequencies.shape[0])
    mask[:n] = 1.0
    filtered_frequencies = signal_frequencies * mask

    return filtered_frequencies


def gaussian_low_pass_filter(
    signal_frequencies: NDArray[np.complex128],
    f: float,
) -> NDArray[np.complex128]:
    """Gaussian low-pass filter.

    Args:
        signal_frequencies: the input signal transformed in the frequency
            domain.
        f: the fraction of frequencies to keep, this will determine the length
            scale of the Gaussian filter

    Returns:
        the filtered frequencies
    """
    # number of low-frequency component to keep
    sigma = np.round(f * signal_frequencies.shape[0])

    # gaussian low-pass filter
    mask = np.exp(-np.arange(signal_frequencies.shape[0]) ** 2 / (2 * sigma**2))
    filtered_frequencies = signal_frequencies * mask

    return filtered_frequencies


class FourierLoss(BaseLoss):
    """Class for the Fourier loss."""

    def __init__(
        self,
        frequency_filter: FrequencyFilter = gaussian_low_pass_filter,
        f: float = 0.8,
        coordinate_weights: Optional[NDArray] = None,
    ) -> None:
        """Loss computed using a distance in the Fourier space of the time series.

        This loss is equivalent to the Euclidean loss computed on the time
        series after a Fourier-filter.
        The parameter f controls the fraction of frequencies that are kept in
        the Fourier series.

        Args:
            frequency_filter: the function used to filter the fourier
                frequencies before the distance is computed.
            f: fraction of fourier components to keep when computing the
                distance between the time series. This parameter will be passed
                to frequency_filter.
            coordinate_weights: relative weights of the losses computed over
                different time series coordinates.
        """
        assert_(
            0.0 < f <= 1.0,
            "'f' must be in the interval (0.0, 1.0]",
        )
        self.frequency_filter = frequency_filter
        self.f = f
        super().__init__(coordinate_weights)

    def compute_loss_1d(
        self, sim_data_ensemble: NDArray[np.float64], real_data: NDArray[np.float64]
    ) -> float:
        """Compute Euclidean distance between the Fourier transform of the two time series.

        Args:
            sim_data_ensemble: the first operand
            real_data: the second operand

        Returns:
            The computed loss over the specific coordinate.
        """
        f_real_data = np.fft.rfft(real_data, axis=0)
        N = f_real_data.shape[0]
        f_real_data = self.frequency_filter(f_real_data, self.f)
        # computer mean fft transform of simulated ensemble
        f_sim_data = []
        for s in sim_data_ensemble:
            f_sim_data_ = np.fft.rfft(s, axis=0)
            f_sim_data_ = self.frequency_filter(f_sim_data_, self.f)
            f_sim_data.append(f_sim_data_)

        f_sim_data = np.array(f_sim_data).mean(0)

        loss_1d = np.sqrt(np.sum((abs(f_sim_data - f_real_data)) ** 2) / N)

        return loss_1d
