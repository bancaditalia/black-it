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

"""Module containing simple models used for testing."""

import numpy as np
from numpy.typing import NDArray


# Just an iid normal sampling
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
