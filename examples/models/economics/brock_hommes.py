# Black-box ABM Calibration Kit (Black-it)
# Copyright (C) 2021-2023 Banca d'Italia
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

from typing import Sequence

import numpy as np
from scipy.special import softmax

# ****
# Brock and Hommes 1998
# ****


def BH2(theta: Sequence[float], N: int, seed: int):  # noqa: N802, N803
    """
    Model from Brock and Hommes 1998.
    4.1.2. Fundamentalists versus trend chasers

    theta structure:
    [g1, b1,  g2, b2]

    to replicate BH98 4.1.2 - Fig2(a)
    [0.0,0.0,1.2,0.0]
    beta = 3.6
    sigma, a, R = 1.0, 1.0, 1.1

    Args:
        theta: parameters
        N: length of simulation
        seed: random seed

    Returns:
        simulated series
    """
    np.random.seed(seed=seed)

    R = 1.10  # noqa: N806
    beta = 3.6
    sigma = 1.0
    a = 1.0
    div_eps_min = 0  # -0.05
    div_eps_max = 0  # 0.05
    C = 1.0  # noqa: N806

    x_lag2 = 0.10
    x_lag1 = 0.10

    bsa = beta / (a * (sigma**2))

    x = np.zeros(N + 2)
    n = np.full(2, 0.50)
    g = np.array([0.0, 1.2])
    b = np.array([0.0, 0.0])

    x[0] = x_lag2
    x[1] = x_lag1

    for i in range(int(len(theta) / 2)):
        g[i] = theta[i * 2]
        b[i] = theta[i * 2 + 1]

    for t in range(2, N + 1):
        x[t] = n[1] * g[1] * x[t - 1] / R
        x[t] = x[t] + np.random.uniform(low=div_eps_min, high=div_eps_max, size=1)

        n[0] = np.exp(bsa * (R * x[t - 1] * (R * x[t - 1] - x[t])) - beta * C)
        n[1] = np.exp(bsa * (x[t] - R * x[t - 1]) * (g[1] * x[t - 2] - R * x[t - 1]))
        n = n / np.sum(n)

    return np.atleast_2d(x[2:]).T


#
def BH4(theta: Sequence[float], N: int, seed: int):  # noqa: N802, N803
    """Model from Brock and Hommes 1998.

    4.3 Four belief types: Fundamentalists versus trend versus bias

    This function implements the parametrisation of the model proposed in Platt (2020)

    theta structure:
    [g1,b1, g2,b2, g3,b3, g4,b4]

    From Donovan Platt 2019: (experiments run)
    [g2, b2] and [g2, b2, g3, b3], with all parameters assumed to lie in the
    interval [0, 1], with the exception of b3, which we assume lies in the interval [−1, 0].
    [g1,b1,  g2,b2,  g3,b3,  g4,b4,  r,β] = [0,0,  0.9,0.2,  0.9,-0.2,  1.01,0,  0.01,1]

    Args:
        theta: parameters
        N: length of simulation
        seed: random seed

    Returns:
        simulated series
    """
    np.random.seed(seed=seed)

    R = 1.01  # noqa: N806
    beta = 120
    sigma = 0.04

    # BH noise:
    x_lag2 = 0.05
    x_lag1 = 0.10

    x = np.zeros(N + 2)
    n = np.full(4, 0.25)
    g = np.array([0.0, 0.9, 0.9, 1.01])
    b = np.array([0.0, 0.2, -0.2, 0.00])

    x[0] = x_lag2
    x[1] = x_lag1

    for i in range(min(int(len(theta) / 2), 4)):
        g[i] = theta[i * 2]
        b[i] = theta[i * 2 + 1]

    if len(theta) >= 9:
        R = 1.0 + theta[8]  # noqa: N806
    if len(theta) >= 10:
        beta = theta[9]

    for t in range(2, N + 1):
        expectation = np.add(g * x[t - 1], b)
        weighted_exp = np.multiply(n, expectation)
        # BH noise:
        # divEpsMin = 0  # -0.05
        # divEpsMax = 0  # 0.05
        # dividend_noise = np.random.uniform(low=divEpsMin, high=divEpsMax, size=1)
        # DP noise:
        dividend_noise = np.random.normal(0, sigma, 1)
        x[t] = (np.sum(weighted_exp) + dividend_noise) / R

        left_factor = x[t] - R * x[t - 1]
        right_factor = np.add(g * x[t - 2], b) - R * x[t - 1]

        n = softmax(beta * left_factor * right_factor)

    return np.atleast_2d(x[2:]).T
