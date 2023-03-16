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

"""
Executes the C++ simulator with chosen command line parameters, and returns the simulation output as a Python list.

The library can also be invoked directly from command line on a custom
simulator command.

USAGE AS A LIBRARY:
    >>> import simlib
    >>> simulation_result = simlib.execute_simulator(
    ...     'bancaditalia/abmsimulator', {'agents': 1000, 'epochs': 2, 'beta': 0.5, 'gamma': 0.5})
    >>> from pprint import pprint
    >>> pprint(simulation_result)
    [{'infectious': 2, 'recovered': 0, 'susceptible': 998},
     {'infectious': 4, 'recovered': 1, 'susceptible': 995},
     {'infectious': 2, 'recovered': 3, 'susceptible': 995}]


DIRECT USAGE FROM COMMAND LINE:
    $ python simlib.py coronabm --agents 1000 --epochs 2 --beta 0.5 --gamma 0.5
    Executing: coronabm --agents 1000 --epochs 2 --beta 0.5 --gamma 0.5
    [{"susceptible": 998, "infectious": 2, "recovered": 0},
     {"susceptible": 995, "infectious": 4, "recovered": 1},
     {"susceptible": 995, "infectious": 2, "recovered": 3}]
"""

import json
import re
import subprocess
import sys
from itertools import chain
from typing import Dict, List


def parse_simulator_output(stdout: str) -> List[Dict[str, int]]:
    """Filter the output of a simulator execution and convert it to a python list."""
    regex = re.compile(r"^EPOCH.*(initial status|ended with)")

    noise_filtered_out = filter(
        lambda line: regex.match(line) is not None, stdout.splitlines()
    )
    results_as_text = (line.split(":", 1)[1] for line in noise_filtered_out)
    results = (json.loads(text) for text in results_as_text)

    return list(results)


def _build_simulator_cmdline(
    docker_image_name: str, sim_params: Dict[str, str]
) -> List[str]:
    """Convert a configuration object in a list of command line parameters.

    Accepts a configuration object for the simulation and returns a long term
    form arguments list, suitable for invoking the simulator via
    subprocess.run().

    For a reference of the simulator's command line arguments, see:
    https://github.com/bancaditalia/CoronABM/blob/master/cppsim/README.md#examples

    For example:
        { 'agents': 1, 'lattice-order' : 2 }
    Becomes:
        [ '--agents' , '1', '--lattice-order', '2' ]

    REMARKS:
        "args" is in the form accepted by subprocess.run(), thus:

        - args[0] is the path to the simulator executable
        - args[n>0] are the command line arguments passed to the simulator

    Args:
        docker_image_name: the name of the Docker image to run
        sim_params: the simulation parameters

    Returns:
        the arguments for the Docker CLI
    """
    args = ["docker", "run", "--rm", docker_image_name] + list(
        chain.from_iterable(
            (
                (f"--{argname}", str(argvalue))
                for argname, argvalue in sim_params.items()
            )
        )
    )

    return args


def execute_simulator(
    path_to_simulator: str, sim_params: Dict[str, str]
) -> List[Dict[str, int]]:
    """Execute the simulator with the given parameters, and return a structured output.

    - the simulator parameters are converted via _build_simulator_cmdline()
    - the simulator is invoked and its output parsed via
      _execute_simulator_subprocess()

    Example of a return value:

      [
        {'infectious': 2, 'recovered': 0, 'susceptible': 99998}, <-- epoch 0 (initial state)
        {'infectious': 2, 'recovered': 0, 'susceptible': 99998}, <-- epoch 1
        {'infectious': 2, 'recovered': 0, 'susceptible': 99998}, <-- ...
      ]

    Args:
        path_to_simulator: path to the simulator
        sim_params: the simulation parameters

    Returns:
        the simulation output
    """
    args = _build_simulator_cmdline(path_to_simulator, sim_params)

    return _execute_simulator_subprocess(args)


def _execute_simulator_subprocess(args: List[str]) -> List[Dict[str, int]]:
    """Execute the simulator and convert its output in structured form via parse_simulator_output().

    REMARKS:
        "args" must be in the form accepted by subprocess.run(). To build it
        from a configuration object, you can invoke _build_simulator_cmdline().

    Args:
        args: the arguments for the simulator executable

    Returns:
        the simulation output
    """
    res = subprocess.run(args, text=True, capture_output=True)

    if res.returncode != 0:
        # bail out in case of error. In a mature system we should return &
        # handle the error instead
        print(f"Error executing {args[0]}")
        print(f"stdout:\n{res.stdout}")
        print(f"stderr:\n{res.stderr}")

        exit(res.returncode)

    simulation_output = parse_simulator_output(res.stdout)

    return simulation_output


def run_single_simulation():
    """Test function that runs the simulator with a fixed set of parameters."""
    # example of formally correct values for simulation parameters
    sim_params = {
        "agents": 100,
        "epochs": 3,
        "beta": 0.1,
        "gamma": 0.1,
        "infectious-t0": 10,
        "lattice-order": 20,
        "rewire-probability": 0.2,
    }

    simulation_output = execute_simulator("bancaditalia/abmsimulator", sim_params)
    print(simulation_output)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_single_simulation()
    else:
        # Executes an arbitrary simulator (passed as first command line
        # argument), taking a verbatim list of command line parameters.
        # Parses the output of the simulator into a python object, and prints a
        # json representation on stdout.
        #
        # Useful only for testing purposes, if one wants to normalize the
        # simulator output, or wants to see hot simlib internally parses the
        # simulation results.
        simulation_output = _execute_simulator_subprocess(sys.argv[1:])
        print(json.dumps(simulation_output))
