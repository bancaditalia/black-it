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

"""Module containing simple models used for testing."""

import numpy as np
from numpy.typing import NDArray

# Just an iid normal sampling
from scipy.special import softmax


def NormalMV(theta: NDArray, N: int, seed: int) -> NDArray:
    """Sample from a normal distribution with adjustable mean and variance.

    Args:
        theta: mean and standard deviation of the distribution
        N: number of samples
        seed: random seed

    Returns:
        the sampled series
    """
    np.random.seed(seed=seed)

    y = np.random.normal(theta[0], theta[1], N)
    return np.atleast_2d(y).T


def BH4(theta: NDArray, N: int, seed: int) -> NDArray:
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

    R = 1.01
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

    for i in range(int(len(theta) / 2)):
        g[i] = theta[i * 2]
        b[i] = theta[i * 2 + 1]

    for t in range(2, N + 1):
        expectation = np.add(g * x[t - 1], b)
        weighted_exp = np.multiply(n, expectation)
        dividend_noise = np.random.normal(0, sigma, 1)
        x[t] = (np.sum(weighted_exp) + dividend_noise) / R

        left_factor = x[t] - R * x[t - 1]
        right_factor = np.add(g * x[t - 2], b) - R * x[t - 1]

        n = softmax(beta * left_factor * right_factor)

    return np.atleast_2d(x[2:]).T
