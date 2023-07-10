import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

import torch
import torch.nn as nn

from typing import Callable


# TODO: Some issues with the parameters imho, the optimizer as a Callback is not
# a particularly nice way to do it
# TODO: Make trainable either directly from model or from datapath
class MomentNetwork:
    def __init__(self, net: nn.Module, model: Callable,
                 l_bounds: NDArray[np.float64], u_bounds: NDArray[np.float64], 
                 N: int, batch_size: int = 256, epochs: int = 100, 
                 criterion = nn.MSELoss(), 
                 optimizer: Callable = lambda x: torch.optim.Adam(x, lr=1e-3),
                 batch_dtype: torch.dtype = torch.float32, verbosity: int = 10):
        
        self.net = net
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer(self.net.parameters())

        # Sampler and bounds for model parameters
        self.sampler = qmc.Sobol(d=len(l_bounds))
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds

        self.N = N
        self.batch_size = batch_size
        self.epochs = epochs
        self.batch_dtype = batch_dtype
        self.verbosity = verbosity


    # TODO: Do we care about handling seeds? Any reason to do so instead of specifying the seed at start?
    # The issue is that currently the seed is on the model, so we can't really set the seed for a batch
    def train_network(self):
        # Ensure that the network is in training mode
        self.train()
        running_loss = 0.0

        # Train the network
        for epoch in range(self.epochs):
            X, Y = self.make_batch(dtype=self.batch_dtype)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + backward + optimize
            loss = self.criterion(self.net(X), Y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if (epoch + 1) % self.verbosity == 0:
                print(f"Epoch {1 + epoch:>{len(str(self.epochs))}} -",
                      f"Running loss: {running_loss / self.verbosity:.4f}")
                running_loss = 0.0



    def make_batch(self, dtype: torch.dtype):
        """Make a batch of data for training the network."""
        theta = self.sampler.random(self.batch_size)
        Y = torch.tensor(self.l_bounds + theta * (self.u_bounds - self.l_bounds), dtype=dtype)
        X = torch.tensor(np.hstack([self.model(y, self.N, None) for y in Y]), dtype=dtype)
        X = torch.unsqueeze(X.transpose(1, 0), 1)

        return X, Y

    def train(self):
        """Sets the network to training mode."""
        self.net.train()

    def eval(self):
        """Sets the network to evaluation mode."""
        self.net.eval()

    def __call__(self, x: torch.Tensor):
        return self.net(x)