from examples.models.simple_models import *
from tcn import *
from moment_network import MomentNetwork

import matplotlib.pyplot as plt
import numpy as np
import torch



# Create a simple moment network for the AR1 model
mnet = MomentNetwork(
    net = TemporalConvNet(1, 1, [10, 10, 10, 10, 10], 32),
    model = AR1,
    l_bounds = np.array([-1 + 1e-6]),
    u_bounds = np.array([1 - 1e-6]),
    N = 100, 
    verbosity = 10,
    epochs = 200
)

# Train the network
mnet.train_network()

# Set the network to evaluation mode and generate some data at an arbitrary theta
mnet.eval()
theta = [0.3]

X = np.hstack([AR1(theta, mnet.N, None) for _ in range(10000)])
X = torch.tensor(X, dtype=mnet.batch_dtype).transpose(1, 0).unsqueeze(1)

Y = mnet(X).flatten().detach().numpy()

# Plot the results of the network's point estimates
plt.hist(Y, bins=100)
plt.show()