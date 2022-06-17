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

"""SIR models written in C and run in Docker containers."""

import numpy as np

from . import simlib


def SIR(theta, N, rndSeed):
    """SIR_docker.

    C++ SIR model run in Docker container.
    """
    sim_params = {
        "agents": 1000,
        "epochs": N - 1,
        "beta": theta[0],
        "gamma": theta[1],
        "infectious-t0": 10,
        "lattice-order": 20,
        "rewire-probability": 0.2,
    }

    res = simlib.execute_simulator("coronabm", sim_params)
    ret = np.array([(x["susceptible"], x["infectious"], x["recovered"]) for x in res])

    return ret


def SIR_w_breaks(theta, N, rndSeed=None):
    """SIR_docker_w_breaks."""
    breaktime = int(theta[0])

    beta1 = theta[1]
    beta2 = theta[2]

    gamma = theta[3]

    sim_params = {
        "agents": 1000,
        "epochs": f"{breaktime},{N - 1 - breaktime}",
        "beta": f"{beta1},{beta2}",
        "gamma": gamma,
        "infectious-t0": 1,
        "lattice-order": 20,
        "rewire-probability": 0.0,
    }

    res = simlib.execute_simulator("coronabm", sim_params)
    ret = np.array([(x["susceptible"], x["infectious"], x["recovered"]) for x in res])

    return ret
