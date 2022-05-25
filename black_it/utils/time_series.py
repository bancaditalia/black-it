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

"""This module contains utility functions to deal with time series."""

import numpy as np
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew


def get_mom_ts(time_series: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute specific moments from a time series.

    Args:
        time_series: the time series

    Returns:
        the moments
    """
    moments = np.array([get_mom_ts_1d(ts_d) for ts_d in time_series.T])
    return moments.T


def get_mom_ts_1d(time_series: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute specific moments from a time series.

    Args:
        time_series: the time series

    Returns:
        the moments from a time series
    """
    # array of all moments (to be filled)
    avg_vec_mom = np.zeros(18)

    # first 4 moments and auto-correlations of the time series
    avg_vec_mom[0] = np.mean(time_series)
    avg_vec_mom[1] = np.std(time_series)
    s, k = skew(time_series), kurtosis(time_series)  # pylint: disable=invalid-name
    avg_vec_mom[2] = np.sign(s) * np.power(abs(s), 1.0 / 3.0)
    avg_vec_mom[3] = np.sign(k) * np.power(abs(k), 1.0 / 4.0)

    ts_acf = sm.tsa.acf(time_series, nlags=5, fft=False)
    avg_vec_mom[4] = ts_acf[1]
    avg_vec_mom[5] = ts_acf[2]
    avg_vec_mom[6] = ts_acf[3]
    avg_vec_mom[7] = ts_acf[4]
    avg_vec_mom[8] = ts_acf[5]

    # first 4 moments and auto-correlations of the absolute differences of the time series
    abs_diff = np.absolute(np.diff(time_series))

    avg_vec_mom[9] = np.mean(abs_diff)
    avg_vec_mom[10] = np.std(abs_diff)
    s, k = skew(abs_diff), kurtosis(abs_diff)  # pylint: disable=invalid-name
    avg_vec_mom[11] = np.sign(s) * np.power(abs(s), 1.0 / 3.0)
    avg_vec_mom[12] = np.sign(k) * np.power(abs(k), 1.0 / 4.0)

    ts_diff_acf = sm.tsa.acf(abs_diff, nlags=5, fft=False)
    avg_vec_mom[13] = ts_diff_acf[1]
    avg_vec_mom[14] = ts_diff_acf[2]
    avg_vec_mom[15] = ts_diff_acf[3]
    avg_vec_mom[16] = ts_diff_acf[4]
    avg_vec_mom[17] = ts_diff_acf[5]

    np.nan_to_num(avg_vec_mom, False)

    return avg_vec_mom
