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

"""Mesa-based Boltzmann Wealth Model.

Taken from the 'Mesa' documentation:

    https://github.com/projectmesa/mesa/blob/main/examples/boltzmann_wealth_model/boltzmann_wealth_model/model.py

and modified to allow more flexibility.
"""

from typing import cast

import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation


class BoltzmannWealthModel(Model):
    """A simple model of an economy where agents exchange currency at random.

    All the agents begin with one unit of currency, and each time step can give
    a unit of currency to another agent. Note how, over time, this produces a
    highly skewed distribution of wealth.
    """

    def __init__(
        self,
        num_agents: int = 100,
        width: int = 10,
        height: int = 10,
        generosity_ratio: float = 100.0,
        mean_init_wealth: int = 3,
    ) -> None:
        """Initialize the model."""
        self.num_agents = num_agents
        self.grid = MultiGrid(height, width, True)
        self.generosity_ratio = generosity_ratio
        self.mean_init_wealth = mean_init_wealth
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={"Gini": compute_gini},
            agent_reporters={"Wealth": "wealth"},
        )
        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.running = True
        self.datacollector.collect(self)

    def step(self) -> None:
        """Do a step."""
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n: int) -> None:
        """Run the model."""
        for _ in range(n):
            self.step()


class MoneyAgent(Agent):
    """An agent with fixed initial wealth."""

    model: BoltzmannWealthModel

    def __init__(self, unique_id: int, model: BoltzmannWealthModel) -> None:
        """Initialize the agent."""
        super().__init__(unique_id, model)
        self.wealth = np.random.binomial(  # noqa: NPY002
            n=2 * self.model.mean_init_wealth,
            p=0.5,
        )

    def move(self) -> None:
        """Move the agent."""
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self) -> None:
        """Give the money to another agent."""
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other = cast(MoneyAgent, other)

            if other.wealth <= self.wealth * self.model.generosity_ratio:
                other.wealth += 1
                self.wealth -= 1

    def step(self) -> None:
        """Do a step in the environment."""
        self.move()
        if self.wealth > 0:
            self.give_money()


def compute_gini(model: BoltzmannWealthModel) -> float:
    """Compute the Gini index."""
    agent_wealths = [cast(MoneyAgent, agent).wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    n = model.num_agents
    b = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * sum(x))  # noqa: N806
    return 1 + (1 / n) - 2 * b
