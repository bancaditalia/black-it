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

"""SIR models written in Python."""

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx
import numpy as np


def SIR(theta, N, seed):
    """SIR model.

    0 theta = [#LOC CONNECTIONS,
    1           %NON-LOC.CONNECTIONS,
    2           COND INIZIALE TOPOLOGIA,
    3           INF.RATE,
    4           REC.RATE,
    5           NETWORKX RND seed]

    Args:
        theta: parameters
        N: length of simulation
        seed: random seed

    Returns:
        simulated series
    """
    np.random.seed(seed=seed)

    NumAgents = 100000
    g = nx.watts_strogatz_graph(NumAgents, int(theta[0]), theta[1], seed=theta[5])

    model = ep.SIRModel(g)

    cfg = mc.Configuration()
    cfg.add_model_parameter("beta", theta[3])  # infection rate
    cfg.add_model_parameter("gamma", theta[4])  # recovery rate
    cfg.add_model_parameter("percentage_infected", theta[2])
    model.set_initial_status(cfg)

    iterations = model.iteration_bunch(N, node_status=True)

    outputNP = np.zeros((N, 3))

    for i in range(len(iterations)):
        entry = iterations[i]["node_count"]

        for j in range(3):
            outputNP[i, j] = entry[j]

    g.clear()

    return outputNP


def SIR_w_breaks(theta, N, seed):
    """SIR model with structural breaks.

    0  theta = [#LOC CONNECTIONS,
    1           %NON-LOC.CONNECTIONS,
    2           COND INIZIALE TOPOLOGIA,
    3           INF.RATE 1,
    4           REC.RATE 1,
    5           INF.RATE 2,
    6           INF.RATE 3,
    7           INF.RATE 4,
    8           T BREAK 1 ,
    9           T BREAK 2 ,
    10          T BREAK 3 ,
    11          NETWORKX RND seed]

    Args:
        theta: parameters
        N: length of simulation
        seed: random seed

    Returns:
        simulated series
    """
    np.random.seed(seed=seed)

    NumAgents = 100000
    g = nx.watts_strogatz_graph(NumAgents, int(theta[0]), theta[1], seed=theta[11])

    model = ep.SIRModel(g)

    cfg = mc.Configuration()
    cfg.add_model_parameter("beta", theta[3])  # infection rate
    cfg.add_model_parameter("gamma", theta[4])  # recovery rate
    cfg.add_model_parameter("percentage_infected", theta[2])
    model.set_initial_status(cfg)

    iterations0 = model.iteration_bunch(int(theta[8]), node_status=True)

    model.params["model"]["beta"] = theta[5]
    iterations1 = model.iteration_bunch(int(theta[9] - theta[8]), node_status=True)

    model.params["model"]["beta"] = theta[6]
    iterations2 = model.iteration_bunch(int(theta[10] - theta[9]), node_status=True)

    model.params["model"]["beta"] = theta[7]
    iterations3 = model.iteration_bunch(int(N - theta[10]), node_status=True)

    outputNP = np.zeros((N, 3))

    for i in range(len(iterations0)):
        entry = iterations0[i]["node_count"]

        for j in range(3):
            outputNP[i, j] = entry[j]

    for i in range(len(iterations1)):
        entry = iterations1[i]["node_count"]

        for j in range(3):
            outputNP[i + theta[8], j] = entry[j]

    for i in range(len(iterations2)):
        entry = iterations2[i]["node_count"]

        for j in range(3):
            outputNP[i + theta[9], j] = entry[j]

    for i in range(len(iterations3)):
        entry = iterations3[i]["node_count"]
        for j in range(3):
            outputNP[i + theta[10], j] = entry[j]

    return outputNP
