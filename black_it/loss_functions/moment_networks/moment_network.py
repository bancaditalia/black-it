# Black-box ABM Calibration Kit (Black-it)
# Copyright for this module (C) 2023 Jonathan Chassot and Banca d'Italia
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
This module contains a Moment Network implementation.

- Chassot, J. (2023).
  Paper name

Algorithm description:

"""


from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from black_it.samplers.base import BaseSampler
from black_it.search_space import SearchSpace

MSE_LOSS = nn.MSELoss()


# TODO: Some issues with the parameters imho, the optimizer as a Callback is not
# a particularly nice way to do it
# TODO: Make trainable either directly from model or from datapath
class MomentNetwork:
    """Class for the Moment Network."""

    def __init__(
        self,
        net: nn.Module,
        model: Callable,
        search_space: SearchSpace,
        N: int,
        sampler: BaseSampler,
        criterion: nn.Module = MSE_LOSS,
        optimizer: Callable = lambda x: torch.optim.Adam(x, lr=1e-3),
        batch_dtype: torch.dtype = torch.float32,
        verbosity: int = 10,
    ):
        """Initialise a Moment Network.

        Args:
            net (nn.Module): _description_
            model (Callable): _description_
            search_space (SearchSpace): _description_
            N (int): _description_
            sampler (BaseSampler): _description_
            criterion (nn.Module, optional): _description_. Defaults to MSE_LOSS.
            optimizer (_type_, optional): _description_. Defaults to lambdax:torch.optim.Adam(x, lr=1e-3).
            batch_dtype (torch.dtype, optional): _description_. Defaults to torch.float32.
            verbosity (int, optional): _description_. Defaults to 10.
        """
        self.net = net
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer(self.net.parameters())

        # Sampler and bounds for model parameters
        self.sampler = sampler
        self.search_space = search_space

        self.N = N
        self.batch_dtype = batch_dtype
        self.verbosity = verbosity

    # TODO: Do we care about handling seeds? Any reason to do so instead of specifying the seed at start?
    # The issue is that currently the seed is on the model, so we can't really set the seed for a batch
    def train_network(self, epochs: int = 100) -> None:
        """Train the network for a number of epochs."""
        # Ensure that the network is in training mode
        self.train()
        running_loss = 0.0

        # Train the network
        for epoch in range(epochs):
            X, Y = self.make_batch(dtype=self.batch_dtype)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + backward + optimize
            loss = self.criterion(self.net(X), Y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if (epoch + 1) % self.verbosity == 0:
                print(
                    f"Epoch {1 + epoch:>{len(str(epochs))}} -",
                    f"Running loss: {running_loss / self.verbosity:.4f}",
                )
                running_loss = 0.0

    def make_batch(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make a batch of data for training the network."""
        theta = self.sampler.sample(
            self.search_space,
            np.zeros((0, self.search_space.dims)),
            np.zeros((0, self.search_space.dims)),
        )

        Y = torch.tensor(theta, dtype=dtype)
        X = torch.tensor(
            np.hstack([self.model(y, self.N, None) for y in Y]), dtype=dtype
        )
        X = torch.unsqueeze(X.transpose(1, 0), 1)

        return X, Y

    def train(self) -> None:
        """Set the network to training mode."""
        self.net.train()

    def eval(self) -> None:
        """Set the network to evaluation mode."""
        self.net.eval()

    def __call__(self, x: NDArray) -> NDArray:
        """Compute the moments from a time series."""
        assert x.shape[0] == self.N
        x_t: torch.Tensor = (
            torch.tensor(x, dtype=self.batch_dtype)
            .unsqueeze(1)
            .unsqueeze(1)
            .transpose(0, 2)
        )
        y: torch.Tensor = self.net(x_t)
        return y.flatten().detach().numpy()
