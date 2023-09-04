import matplotlib.pyplot as plt
import numpy as np
import torch

from black_it.loss_functions.moment_networks.moment_network import MomentNetwork
from black_it.loss_functions.moment_networks.tcn import *
from black_it.samplers.halton import HaltonSampler
from black_it.search_space import SearchSpace
from ..fixtures.test_models import AR1


def test_moment_network() -> None:
    # Define a search space for the network
    search_space = SearchSpace(
        parameters_bounds=[[-1 + 1e-3], [1 - 1e-3]],
        parameters_precision=[0.001],
        verbose=False,
    )

    # Define a sampler for the network.
    # Note that batch_size corresponds also to the size of the minibatches used for training
    sampler = HaltonSampler(batch_size=32)

    # Create a simple moment network for the AR1 model
    mnet = MomentNetwork(
        net=TemporalConvNet(1, 1, [10, 10], 32),
        model=AR1,
        search_space=search_space,
        sampler=sampler,
        N=100,
        verbosity=10,
    )

    # Train the network
    mnet.train_network(10)

    # Set the network to evaluation mode and generate some data at an arbitrary theta
    mnet.eval()
    theta = [0.3]

    X = np.hstack([AR1(theta, mnet.N, None) for _ in range(1000)])
    X_t: torch.Tensor = torch.tensor(X, dtype=mnet.batch_dtype).transpose(1, 0).unsqueeze(1)

    Y = mnet.net(X_t).flatten().detach().numpy()

    assert True
