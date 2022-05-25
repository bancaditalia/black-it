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
#

"""This module contains helper functions for plotting TS descriptive statistics."""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


def ts_stats(ts: List[float]) -> None:  # pylint: disable=too-many-locals
    """Show TS graphical descriptive statistics."""
    color = "darkslateblue"
    alpha = 0.8

    Ts = np.array(ts, dtype="double")
    SqDiff = np.append(
        np.absolute(np.diff(Ts)), 0
    )  # not precise! shouldn't append '0'!

    TsAcf = sm.tsa.acf(Ts, nlags=20)
    SqDiffAcf = sm.tsa.acf(SqDiff, nlags=20)

    TsPAcf = sm.tsa.pacf(Ts, nlags=20)
    SqDiffPAcf = sm.tsa.pacf(SqDiff, nlags=20)

    WINDOW_W = 20

    fig, _ax = plt.subplots(3, 4, figsize=(WINDOW_W, 15), sharex=False, sharey=False)

    sp1 = plt.subplot(3, 4, 1)
    plt.hist(Ts, 50, facecolor=color, alpha=alpha)
    sp1.set_title("Ts hist")

    sp2 = plt.subplot(3, 4, 2)
    plt.plot(TsAcf, marker="o", linestyle="None", c=color, alpha=alpha)
    sp2.set_title("Ts acf")

    sp3 = plt.subplot(3, 4, 3)
    plt.plot(TsPAcf, marker="o", linestyle="None", c=color, alpha=alpha)
    sp3.set_title("Ts pacf")

    sp4 = plt.subplot(3, 4, 4)
    plt.plot(Ts[0:1000], c=color, alpha=alpha)
    sp4.set_title("Ts sample")

    sp5 = plt.subplot(3, 4, 5)
    plt.hist(SqDiff, 50, facecolor=color, alpha=alpha)
    sp5.set_title("Abs 1st diff hist")

    sp6 = plt.subplot(3, 4, 6)
    plt.plot(SqDiffAcf, marker="o", linestyle="None", c=color, alpha=alpha)
    sp6.set_title("Abs 1st diff acf")

    sp7 = plt.subplot(3, 4, 7)
    plt.plot(SqDiffPAcf, marker="o", linestyle="None", c=color, alpha=alpha)
    sp7.set_title("Abs 1st diff pacf")

    sp8 = plt.subplot(3, 4, 8)
    plt.plot(SqDiff[0:1000], c=color, alpha=alpha)
    sp8.set_title("Abs 1st diff sample")

    sp9 = plt.subplot(3, 4, 9)
    for i in range(len(Ts) - 3):
        plt.plot(Ts[i + 1 : i + 3], Ts[i : i + 2], alpha=0.10, c=color)
    sp9.set_title("Ts X(t) vs X(t-1) traj")

    sp10 = plt.subplot(3, 4, 10)
    for i in range(len(Ts) - 4):
        plt.plot(SqDiff[i + 1 : i + 3], SqDiff[i : i + 2], alpha=0.10, c=color)
    sp10.set_title("Abs 1st diff X(t) vs X(t-1) traj")

    sp11 = plt.subplot(3, 4, 11)
    plt.plot(
        Ts[1 : len(Ts)],
        Ts[0 : len(Ts) - 1],
        marker="o",
        linestyle="None",
        alpha=0.10,
        c=color,
    )
    sp11.set_title("Ts X(t) vs X(t-1) params")

    sp12 = plt.subplot(3, 4, 12)
    plt.plot(
        SqDiff[1 : len(Ts)],
        SqDiff[0 : len(Ts) - 1],
        marker="o",
        linestyle="None",
        alpha=0.10,
        c=color,
    )
    sp12.set_title("Abs 1st diff X(t) vs X(t-1) params")

    fig.suptitle("Descriptive stats", fontsize=16)
