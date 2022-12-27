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
Implementation the CORS sampler.

    Regis, Rommel G., and Christine A. Shoemaker. "Constrained global optimization of expensive black box functions
    using radial basis functions." Journal of Global optimization 31.1 (2005): 153-171.

"""
from math import factorial
from typing import Callable, Optional, cast

import numpy as np
import scipy.optimize as op
from numpy.linalg import LinAlgError
from numpy.typing import NDArray

from black_it.samplers.base import BaseSampler
from black_it.search_space import SearchSpace
from black_it.utils.base import digitize_data, positive_float


def volume_d_dimensional_ball_radius_1(dims: int) -> float:
    """Compute the volume of a d-dimensional ball with radius 1."""
    if dims % 2 == 0:
        return np.pi ** (dims / 2) / factorial(int(dims / 2))
    return (
        2
        * (4 * np.pi) ** ((dims - 1) / 2)
        * factorial(int((dims - 1) / 2))
        / factorial(dims)
    )


def cubetobox(X: NDArray[np.float64], space_bounds: NDArray) -> NDArray[np.float64]:
    """Go from normalized values (unit cube) to absolute values (box)."""
    box_points = space_bounds[0] + X * (space_bounds[1] - space_bounds[0])
    return box_points


def boxtocube(X: NDArray[np.float64], space_bounds: NDArray) -> NDArray[np.float64]:
    """Go from absolute values (box) to normalized values (unit cube)."""
    cube_points = (X - space_bounds[0]) / (space_bounds[1] - space_bounds[0])
    return cube_points


def rbf(points: NDArray[np.float64], losses: NDArray[np.float64]) -> Callable:
    """
    Build RBF-fit for given points (see Holmstrom, 2008 for details).

    Implementation mostly taken from:
    https://github.com/paulknysh/blackbox/blob/cd8baa0cc344f36b3fddf910ae8037f62009619c/black_box/blackbox.py#L148-L195

    Args:
        points: Array of multi-d points with corresponding values [[x1, x2, .., xd, val], ...].
        losses: array of losses for each point.

    Returns:
        callable function that returns the value of the RBF-fit at a given point.
    """
    n = len(points)
    d = len(points[0])

    def phi(r: float) -> float:
        """Compute phi."""
        return r * r * r

    Phi = [
        [phi(np.linalg.norm(np.subtract(points[i], points[j]))) for j in range(n)]  # type: ignore
        for i in range(n)
    ]

    P = np.ones((n, d + 1))
    P[:, 0:-1] = points

    F = losses

    M = np.zeros((n + d + 1, n + d + 1))
    M[0:n, 0:n] = Phi
    M[0:n, n : n + d + 1] = P
    M[n : n + d + 1, 0:n] = np.transpose(P)

    v = np.zeros(n + d + 1)
    v[0:n] = F

    try:
        sol = np.linalg.solve(M, v)
    except LinAlgError:
        # might help with singular matrices
        print(
            "Singular matrix occurred during RBF-fit construction. RBF-fit might be inaccurate!"
        )
        sol = np.linalg.lstsq(M, v)[0]

    lam, b, a = sol[0:n], sol[n : n + d], sol[n + d]

    def fit(x: float) -> float:
        return (
            sum(
                lam[i] * phi(np.linalg.norm(np.subtract(x, points[i])))  # type: ignore
                for i in range(n)
            )
            + np.dot(b, x)
            + a
        )

    return fit


class CORSSampler(BaseSampler):  # pylint: disable=too-many-instance-attributes
    """Implement the modified CORS sampler."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        batch_size: int,
        max_samples: int,
        rho0: float = 0.5,
        p: float = 1.0,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the CORS sampler.

        Args:
            batch_size: the batch size
            max_samples: the maximum number samples used to compute the density decay
            rho0: initial "balls density"
            p: rate of "balls density" decay (p=1 - linear, p>1 - faster, 0<p<1 - slower)
            random_state: the random state of the sampler
            verbose: activate verbose mode
        """
        super().__init__(
            batch_size, random_state=random_state, max_deduplication_passes=0
        )
        self._max_samples = max_samples
        self._rho0 = positive_float(rho0)
        self._p = positive_float(p)
        self._verbose = verbose

        self._batch_id = 0

    @property
    def rho0(self) -> float:
        """Get the rho0 parameter."""
        return self._rho0

    @property
    def p(self) -> float:
        """Get the p parameter."""
        return self._p

    def sample_batch(
        self,
        batch_size: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sample a batch of points."""
        nb_seed_points = len(existing_points)
        fmax = np.max(np.abs(existing_losses))
        v1 = volume_d_dimensional_ball_radius_1(search_space.dims)

        # the first time sample batch is called. Normalize seed_points
        current_points = boxtocube(existing_points, search_space.parameters_bounds)
        current_losses = existing_losses / fmax

        # fit RBF to all data
        fit = rbf(cast(NDArray[np.float64], current_points), current_losses)

        # enlarge array for new points
        current_points = np.append(
            cast(NDArray[np.float64], current_points),
            np.zeros((batch_size, search_space.dims)),
            axis=0,
        )

        for j in range(batch_size):
            r = (
                (
                    self.rho0
                    * (
                        (self._max_samples - 1.0 - (self._batch_id * batch_size + j))
                        / (self._max_samples - 1.0)
                    )
                    ** self.p
                )
                / (v1 * (nb_seed_points + self._batch_id * batch_size + j))
            ) ** (1.0 / search_space.dims)

            if self._verbose:
                print(f"Using radius {r}")
            cons = [
                {
                    "type": "ineq",
                    "fun": lambda x, localk=k: np.linalg.norm(
                        np.subtract(x, current_points[localk])
                    )
                    - r,  # noqa: B023  # pylint: disable=cell-var-from-loop
                }
                for k in range(nb_seed_points + j)
            ]
            while True:
                minfit = op.minimize(
                    fit,
                    self.random_generator.random(search_space.dims),
                    method="SLSQP",
                    bounds=[[0.0, 1.0]] * search_space.dims,
                    constraints=cons,
                )
                if not np.isnan(minfit.x)[0]:
                    break

            current_points[nb_seed_points + j] = np.copy(minfit.x)

        self._batch_id += 1
        new_cube_batch = current_points[-batch_size:]
        new_box_batch = cubetobox(new_cube_batch, search_space.parameters_bounds)
        return digitize_data(new_box_batch, search_space.param_grid)
