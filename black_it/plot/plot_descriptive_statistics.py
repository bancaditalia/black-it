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
#

"""This module contains helper functions for plotting TS descriptive statistics."""
from __future__ import annotations

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import statsmodels.api as sm  # type: ignore[import]


def ts_stats(time_series_raw: list[float]) -> None:
    """Show TS graphical descriptive statistics."""
    color = "darkslateblue"
    alpha = 0.8

    ts = np.array(time_series_raw, dtype="double")
    sq_diff = np.append(
        np.absolute(np.diff(ts)),
        0,
    )  # not precise! shouldn't append '0'!

    ts_acf = sm.tsa.acf(ts, nlags=20)
    sq_diff_acf = sm.tsa.acf(sq_diff, nlags=20)

    ts_p_acf = sm.tsa.pacf(ts, nlags=20)
    sq_diff_p_acf = sm.tsa.pacf(sq_diff, nlags=20)

    window_w = 20

    fig, _ax = plt.subplots(3, 4, figsize=(window_w, 15), sharex=False, sharey=False)

    sp1 = plt.subplot(3, 4, 1)
    plt.hist(ts, 50, facecolor=color, alpha=alpha)
    sp1.set_title("Ts hist")

    sp2 = plt.subplot(3, 4, 2)
    plt.plot(ts_acf, marker="o", linestyle="None", c=color, alpha=alpha)
    sp2.set_title("Ts acf")

    sp3 = plt.subplot(3, 4, 3)
    plt.plot(ts_p_acf, marker="o", linestyle="None", c=color, alpha=alpha)
    sp3.set_title("Ts pacf")

    sp4 = plt.subplot(3, 4, 4)
    plt.plot(ts[0:1000], c=color, alpha=alpha)
    sp4.set_title("Ts sample")

    sp5 = plt.subplot(3, 4, 5)
    plt.hist(sq_diff, 50, facecolor=color, alpha=alpha)
    sp5.set_title("Abs 1st diff hist")

    sp6 = plt.subplot(3, 4, 6)
    plt.plot(sq_diff_acf, marker="o", linestyle="None", c=color, alpha=alpha)
    sp6.set_title("Abs 1st diff acf")

    sp7 = plt.subplot(3, 4, 7)
    plt.plot(sq_diff_p_acf, marker="o", linestyle="None", c=color, alpha=alpha)
    sp7.set_title("Abs 1st diff pacf")

    sp8 = plt.subplot(3, 4, 8)
    plt.plot(sq_diff[0:1000], c=color, alpha=alpha)
    sp8.set_title("Abs 1st diff sample")

    sp9 = plt.subplot(3, 4, 9)
    for i in range(len(ts) - 3):
        plt.plot(ts[i + 1 : i + 3], ts[i : i + 2], alpha=0.10, c=color)
    sp9.set_title("Ts X(t) vs X(t-1) traj")

    sp10 = plt.subplot(3, 4, 10)
    for i in range(len(ts) - 4):
        plt.plot(sq_diff[i + 1 : i + 3], sq_diff[i : i + 2], alpha=0.10, c=color)
    sp10.set_title("Abs 1st diff X(t) vs X(t-1) traj")

    sp11 = plt.subplot(3, 4, 11)
    plt.plot(
        ts[1 : len(ts)],
        ts[0 : len(ts) - 1],
        marker="o",
        linestyle="None",
        alpha=0.10,
        c=color,
    )
    sp11.set_title("Ts X(t) vs X(t-1) params")

    sp12 = plt.subplot(3, 4, 12)
    plt.plot(
        sq_diff[1 : len(ts)],
        sq_diff[0 : len(ts) - 1],
        marker="o",
        linestyle="None",
        alpha=0.10,
        c=color,
    )
    sp12.set_title("Abs 1st diff X(t) vs X(t-1) params")

    fig.suptitle("Descriptive stats", fontsize=16)
