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

from typing import Tuple

import numpy as np
import scipy.sparse as sps
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


def hp_filter(
    time_series: NDArray[np.float64], lamb: float = 1600
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply the HP filter to a time series.

    Args:
        time_series: the one dimensional series to be filtered
        lamb: the Hodrick-Prescott smoothing parameter, a value of 1600 is suggested for quarterly data

    Returns:
        the cycle and the trend of the HP decomposition of the time series

    References:
        Hodrick, R.J, and E. C. Prescott. 1980. "Postwar U.S. Business Cycles: An Empirical Investigation."
        Carnegie Mellon University discussion paper no. 451.
    """
    nobs = len(time_series)
    I = sps.eye(nobs, nobs)  # noqa:E741
    offsets = np.array([0, 1, 2])
    data = np.repeat([[1.0], [-2.0], [1.0]], nobs, axis=1)
    K = sps.dia_matrix((data, offsets), shape=(nobs - 2, nobs))

    trend = sps.linalg.spsolve(I + lamb * K.T.dot(K), time_series, use_umfpack=True)
    cycle = time_series - trend

    return cycle, trend


def hp_cycle_lamb1600_filter(time_series: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Get the cycle part of the HP filter to a time series using lamb = 1600.

    Args:
        time_series: the one dimensional series to be filtered

    Returns:
        the cycle part of the HP decomposition of the time series
    """
    return hp_filter(time_series, lamb=1600)[0]


def log_and_hp_filter(time_series: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Take the log of the values and remove the HP trend from them.

    Args:
        time_series: the one dimensional series to be filtered

    Returns:
        the filtered time series
    """
    transformed_series = (
        np.log(time_series) - hp_filter(np.log(time_series), lamb=1600)[1]
    )
    return transformed_series


def diff_log_demean_filter(time_series: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Take the difference of log of the values and remove the mean from them.

    Args:
        time_series: the one dimensional series to be filtered

    Returns:
        the filtered time series
    """
    log = np.log(time_series)
    diff_log = np.diff(log, prepend=log[0])
    transformed_series = diff_log - np.mean(diff_log)
    return transformed_series
