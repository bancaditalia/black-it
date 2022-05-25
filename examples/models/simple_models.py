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

"""
A module with a series of simple non-ABM models.

List of models:
- Normal0
- MarkovC_KP
- AR1
- ARMA2
- ARMAARCH2
- ARMAARCH4
- RWSB1
- RWSB2
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import alpha, bernoulli


def NormalM(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """Normal samples with adjustable mean."""
    np.random.seed(seed=seed)
    y = np.random.normal(theta[0], 1, N)
    return np.atleast_2d(y).T


def NormalMV(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """Normal samples with adjustable mean and variance."""
    np.random.seed(seed=seed)

    y = np.random.normal(theta[0], theta[1], N)
    return np.atleast_2d(y).T


def NormalBer_3P(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """Bernoulli + Normal samples."""
    np.random.seed(seed=seed)

    y = np.zeros(N)
    b = bernoulli.rvs(theta[0], size=N)
    for i in range(N):
        if b[i] == 1:
            y[i] = np.random.normal(theta[1], 0.1, 1)
        else:
            y[i] = np.random.normal(theta[2], 0.1, 1)

    return np.atleast_2d(y).T


def NormalBer_5P(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """Bernoulli + Normal samples."""
    np.random.seed(seed=seed)

    y = np.zeros(N)
    b = bernoulli.rvs(theta[0], size=N)
    for i in range(N):
        if b[i] == 1:
            y[i] = np.random.normal(theta[1], theta[3], 1)
        else:
            y[i] = np.random.normal(theta[2], theta[4], 1)

    return np.atleast_2d(y).T


def Alpha(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """Alpha iid samples.

    theta[0] shape param
    theta[1] position param
    theta[2] scale param
    """
    np.random.seed(seed=seed)

    y = (alpha.rvs(theta[0], size=N) + theta[1]) * theta[2]

    return np.atleast_2d(y).T


def MarkovC_2P(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """Markov chain samples."""
    np.random.seed(seed=seed)

    sigma = 0.0
    y = np.zeros(N)
    x = [0, 1]

    s = 0
    for i in range(N):
        if s == 0:
            b = bernoulli.rvs(theta[0], size=1)
            if b == 1:
                s = 1
                y[i] = np.random.normal(x[1], sigma, 1)
            else:
                y[i] = np.random.normal(x[0], sigma, 1)
        if s == 1:
            b = bernoulli.rvs(theta[1], size=1)
            if b == 1:
                s = 0
                y[i] = np.random.normal(x[0], sigma, 1)
            else:
                y[i] = np.random.normal(x[1], sigma, 1)

    return np.atleast_2d(y).T


def MarkovC_3P(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """Markov chain samples."""
    np.random.seed(seed=seed)

    sigma = 0.0
    y = np.zeros(N)
    x = [0, 1, 2]

    s = 0

    for i in range(N):
        b = bernoulli.rvs(theta[s], size=1)
        if b == 1:
            s = s + 1
            if s == len(x):
                s = 0

        y[i] = np.random.normal(x[s], sigma, 1)

    return np.atleast_2d(y).T


def MarkovC_KP(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """Markov chain samples."""
    np.random.seed(seed=seed)

    sigma = 0.0
    y = np.zeros(N)

    s = 0

    for i in range(N):
        b = bernoulli.rvs(theta[s], size=1)
        if b == 1:
            s = s + 1
            if s == len(theta):
                s = 0

        y[i] = np.random.normal(s, sigma, 1)

    return np.atleast_2d(y).T


def AR1(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """AR(1) model.

    Model 1 in Platt (2019)
    """
    np.random.seed(seed=seed)

    # AR(1)
    # Y_t = p0*Y_t-1 + eps
    y = np.zeros(N)
    y[0] = np.random.normal(0, 1, 1)
    for i in range(1, N):
        y[i] = theta[0] * y[i - 1] + np.random.normal(0, 1, 1)

    return np.atleast_2d(y).T


def AR1Ber_3P(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """AR(1) + Bernoulli."""
    np.random.seed(seed=seed)

    y = np.zeros(N)
    e = np.random.normal(0, 0.1, N)
    b = bernoulli.rvs(theta[0], size=N)
    # s = 0

    y[0] = e[0]
    for i in range(1, N):
        if b[i] == 1:
            # if s == 0: s = 1
            # else: s = 0
            y[i] = theta[1] * y[i - 1] + e[i]
        else:
            y[i] = theta[2] * y[i - 1] + e[i]

    return np.atleast_2d(y).T


def AR1_2P(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """AR(1) with 2 parameters."""
    np.random.seed(seed=seed)

    # AR(1)
    # Y_t = p0*Y_t-1 + eps
    y = np.zeros(N)
    y[0] = np.random.normal(0, 1, 1)
    for i in range(1, N):
        y[i] = theta[0] * y[i - 1] + np.random.normal(0, theta[1], 1)
    return np.atleast_2d(y).T


def AR1_3P(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """AR(1) with 3 parameters."""
    np.random.seed(seed=seed)

    # AR(1)
    # Y_t = p0*Y_t-1 + eps
    y = np.zeros(N)
    y[0] = np.random.normal(0, 1, 1)
    for i in range(1, N):
        y[i] = theta[0] * y[i - 1] + np.random.normal(theta[2], theta[1], 1)

    return np.atleast_2d(y).T


def ARMA2(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """ARMA(1, 1) model."""
    np.random.seed(seed=seed)

    e = np.random.normal(0, 1, N + 1)
    y = np.zeros(N + 1)
    for i in range(1, N):
        y[i] = theta[0] * y[i - 1] + theta[1] * e[i - 1] + e[i]

    return np.atleast_2d(y[1:]).T


def ARMAARCH2(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """ARMA(2,2) ARCH(2) model.

    Model 2 of Platt (2019) - Param set 1 [a0,a1]
    [*a0,*a1,a2,b1,b2,c0,c1,c2] [*0,*0.7,0.1,0.2,0.2,0.25,0.5,0.3]
    """
    np.random.seed(seed=seed)

    e = np.random.normal(0, 1, N + 2)
    s = np.zeros(N + 2)
    y = np.zeros(N + 2)
    for i in range(2, N + 1):
        s[i] = np.sqrt(0.25 + 0.5 * (e[i - 1] ** 2) + 0.3 * (e[i - 2] ** 2))
    for i in range(2, N + 1):
        y[i] = (
            theta[0]
            + theta[1] * y[i - 1]
            + 0.1 * y[i - 2]
            + 0.2 * s[i - 1] * e[i - 1]
            + 0.2 * s[i - 2] * e[i - 2]
            + s[i] * e[i]
        )

    return np.atleast_2d(y[2:]).T


def ARMAARCH4(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """
    ARMA(2,2) ARCH(2) model.

    DONOVAN MODEL 2 - PARAM SET 2 [b1,b2,c0,c1,c2]
    [a0,a1,a2,*b1,*b2,*c0,*c1,*c2] [0,0.7,0.1,*0.2,*0.2,*0.25,*0.5,*0.3]
    """
    np.random.seed(seed=seed)

    e = np.random.normal(0, 1, N + 2)
    s = np.zeros(N + 2)
    y = np.zeros(N + 2)
    for i in range(2, N + 1):
        s[i] = np.sqrt(
            theta[2] + theta[3] * (e[i - 1] ** 2) + theta[4] * (e[i - 2] ** 2)
        )
    for i in range(2, N + 1):
        y[i] = (
            0
            + 0.7 * y[i - 1]
            + 0.1 * y[i - 2]
            + theta[0] * s[i - 1] * e[i - 1]
            + theta[1] * s[i - 2] * e[i - 2]
            + s[i] * e[i]
        )

    return np.atleast_2d(y[2:]).T


def ARMAARCH6(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """
    ARMA(2,2) ARCH(2).

    DONOVAN MODEL 2 - PARAM SET 2 [b1,b2,c0,c1,c2]
    [a0,a1,a2,*b1,*b2,*c0,*c1,*c2] [0,0.7,0.1,*0.2,*0.2,*0.25,*0.5,*0.3]
    """
    np.random.seed(seed=seed)

    e = np.random.normal(0, 1, N + 2)
    s = np.zeros(N + 2)
    y = np.zeros(N + 2)
    for i in range(2, N + 1):
        s[i] = np.sqrt(
            theta[2]
            + theta[3] * ((s[i - 1] * e[i - 1]) ** 2)
            + theta[4] * ((s[i - 2] * e[i - 2]) ** 2)
        )
    for i in range(2, N + 1):
        y[i] = (
            0
            + 0.7 * y[i - 1]
            + 0.1 * y[i - 2]
            + theta[0] * (s[i - 1] * e[i - 1])
            + theta[1] * (s[i - 2] * e[i - 2])
            + (s[i] * e[i])
        )

    return np.atleast_2d(y[2:]).T


def ARMAARCH4v2(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """
    ARMA(2,2) ARCH(2).

    DONOVAN MODEL 2 - PARAM SET 2 [b1,b2,c0,c1,c2]
    [a0,a1,a2,*b1,*b2,*c0,*c1,*c2] [0,0.7,0.1,*0.2,*0.2,*0.25,*0.5,*0.3]
    """
    np.random.seed(seed=seed)

    e = np.random.normal(0, 1, N + 2)
    s = np.zeros(N + 2)
    y = np.zeros(N + 2)
    for i in range(2, N + 1):
        s[i] = np.sqrt(
            theta[2] + theta[3] * ((y[i - 1]) ** 2) + theta[4] * ((y[i - 2]) ** 2)
        )
        y[i] = (
            0
            + 0.7 * y[i - 1]
            + 0.1 * y[i - 2]
            + theta[0] * (s[i - 1] * e[i - 1])
            + theta[1] * (s[i - 2] * e[i - 2])
            + (s[i] * e[i])
        )
    return np.atleast_2d(y[2:]).T


def RWSB1(theta: Sequence[int], N: int, seed: int) -> NDArray[np.float64]:
    """
    RW with structural break.

    DONOVAN MODEL 3 - PARAM SET 1 [tau]
    [*tau,sigma1,sigma2,drift1,drift2] [*700,0.1,0.2,1,2]
    """
    np.random.seed(seed=seed)

    e = np.zeros(N)
    e[0 : theta[0]] = np.random.normal(0, 0.1, theta[0])
    e[theta[0] :] = np.random.normal(0, 0.2, N - theta[0])

    d = np.full(N, 1)
    d[theta[0] :] = 2

    y = np.zeros(N + 1)

    for i in range(N):
        y[i + 1] = y[i] + d[i] + e[i]

    y = np.diff(y[1:])

    return np.atleast_2d(y).T


def RWSB2(theta: Sequence[float], N: int, seed: int) -> NDArray[np.float64]:
    """
    RW with structural break.

    DONOVAN MODEL 3 - PARAM SET 2 [sigma1,sigma2]
    [tau,*sigma1,*sigma2,drift1,drift2] [700,*0.1,*0.2,1,2]
    """
    np.random.seed(seed=seed)

    e = np.zeros(N)
    e[0:700] = np.random.normal(0, theta[0], 700)
    e[700:] = np.random.normal(0, theta[1], N - 700)

    d = np.full(N, 1)
    d[700:] = 2

    y = np.zeros(N + 1)

    for i in range(N):
        y[i + 1] = y[i] + d[i] + e[i]

    y = np.diff(y[1:])
    return np.atleast_2d(y).T
