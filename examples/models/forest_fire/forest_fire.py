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

import subprocess

import numpy as np


def forest_fire(theta, N, rndSeed=0):
    """A simple model of a wildfire on a 2D grid.

    The model is taken from this example from Agent.jl
      https://juliadynamics.github.io/Agents.jl/stable/examples/forest_fire/.

    Warning: This is a Python wrapper for an underlying Julia implementation.
        In order to run this code you need to have Julia installed on your
        computer, as well as the Agents.jl package (a possible command is, from
        inside the Julia REPL: `import Pkg; Pkg.add("Agents")`).

    Args:
        theta: the initial density of trees on the 2D grid
        N: the length of the simulation
        rndSeed: the random seed of the simulation

    Returns:
        An array containing the fraction of trees burned at each time step
    """

    density = theta[0]
    n = N - 1

    # the size of the grid is fixed
    xsize = 30
    ysize = 30
    seed = rndSeed

    command = "julia forest_fire_julia.jl {} {} {} {} {}".format(
        density, n, xsize, ysize, seed
    )

    res = subprocess.run(
        command.split(),
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    stdout = res.stdout

    # remove first lines and last line
    lines = stdout.split("\n")
    lines = lines[4:]
    lines = lines[:-1]

    # parse the result of the simulation
    results = []
    for line in lines:
        splitted_line = line.split()
        results.append(float(splitted_line[-1]))

    results = np.array([results]).T
    return results


if __name__ == "__main__":
    results = forest_fire([0.5], 10, 3)

    print(results)
    print(results.shape)
