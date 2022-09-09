
<p align="center">
<img src="https://raw.githubusercontent.com/bancaditalia/black-it/main/docs/logo/logo_1024.png" width="500">
<sup><a href="#footnote-1">*</a></sup>
</p>

<a href="https://pypi.org/project/black-it">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/black-it" />
</a>

<a href="https://github.com/bancaditalia/black-it/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/bancaditalia/black-it">
</a>

<a href="https://codecov.io/gh/bancaditalia/black-it">
  <img src="https://codecov.io/gh/bancaditalia/black-it/branch/main/graph/badge.svg" />
</a>

# Black-box abm calibration kit

Black-it is an easy-to-use toolbox designed to help you calibrate the parameters
in your agent-based models and simulations (ABMs), using state-of-the-art
techniques to sample the parameter search space, with no need to reinvent the
wheel.

Models from economics, epidemiology, biology, logistics, and more can be dealt
with. The software can be used as-is - if your main interest is the ABM model
itself. However, in case your research thing is to, e.g., devise new sampling
strategies for ginormous search spaces and highly non-linear model, then you can
deploy and test your new ideas on a solid, reusable, modular foundation, in a
matter of days, with no need to reimplement all the plumbings from scratch.

## Installation

This project requires Python v3.8 or later.

To install the latest version of the package from [PyPI](https://pypi.org/project/black-it/):
```
pip install black-it
```

Or, directly from GitHub:

```
pip install git+https://github.com/bancaditalia/black-it.git#egg=black-it
```

If you'd like to contribute to the package, please read the [CONTRIBUTING.md](./CONTRIBUTING.md) guide.

## Quick Example

The GitHub repo of Black-it contains a series ready-to-run calibration examples.

To experiment with them, simply clone the repo and enter the `examples` folder

```
git clone https://github.com/bancaditalia/black-it.git
cd black-it/examples
```

You'll find several scripts and notebooks. The following is the script named `main.py`, note that copying and pasting 
the lines below will not work in general as the script needs to be inside the "examples" folder in order to run correctly. 

```python
import models.simple_models as md

from black_it.calibrator import Calibrator
from black_it.loss_functions.msm import MethodOfMomentsLoss
from black_it.samplers.best_batch import BestBatchSampler
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.random_forest import RandomForestSampler

true_params = [0.20, 0.20, 0.75]
bounds = [
    [0.10, 0.10, 0.10],  # LOWER bounds
    [1.00, 1.00, 1.00],  # UPPER bounds
]
bounds_step = [0.01, 0.01, 0.01]  # Step size in range between bounds

batch_size = 8
halton_sampler = HaltonSampler(batch_size=batch_size)
random_forest_sampler = RandomForestSampler(batch_size=batch_size)
best_batch_sampler = BestBatchSampler(batch_size=batch_size)

# define a model to be calibrated
model = md.MarkovC_KP

# generate a synthetic dataset to test the calibrator
N = 2000
seed = 1
real_data = model(true_params, N, seed)

# define a loss
loss = MethodOfMomentsLoss()

# define the calibration seed
calibration_seed = 1

# initialize a Calibrator object
cal = Calibrator(
    samplers=[halton_sampler, random_forest_sampler, best_batch_sampler],
    real_data=real_data,
    model=model,
    parameters_bounds=bounds,
    parameters_precision=bounds_step,
    ensemble_size=3,
    loss_function=loss,
    random_state=calibration_seed,
)

# calibrate the model
params, losses = cal.calibrate(n_batches=5)

print(f"True parameters:       {true_params}")
print(f"Best parameters found: {params[0]}")
```

When the calibration terminates (~half a minute), towards the end  of the output
you should see the following messages:
```
True parameters:       [0.2, 0.2, 0.75]
Best parameters found: [0.19 0.19 0.74]
```

## Docs

Black-it calibration is initiated via the [Calibrator](https://bancaditalia.github.io/black-it/calibrator/) which,
when called, performs three main steps.

First, a [Sampler](https://bancaditalia.github.io/black-it/samplers/) is summoned to suggest a set of promising 
parameter configurations to explore.

Second, the [model](https://bancaditalia.github.io/black-it/simulator_interface/) to be calibrated is simulated for 
all the selected parameters.

Third, a specific [loss function](https://bancaditalia.github.io/black-it/losses/), measuring the goodness of fitness 
of the simulation data with respect to the real data, is evaluated.

These steps are performed in a loop, and this allows the samplers to progress towards better parameter values 
by exploiting the knowledge of previously computed loss functions.

A more detailed explanation of how Black-it works is available 
[here](https://bancaditalia.github.io/black-it/description/), while the full documentation -complete with examples 
and tutorials- is available [here](https://bancaditalia.github.io/black-it/). 

## Disclaimer

This package is an outcome of a research project. All errors are those of the authors. All views expressed are personal views, not those of Bank of Italy.

---

<p id="footnote-1">
* Credits to <a href="https://www.bankit.art/people/sara-corbo">Sara Corbo</a> for the logo.
</p>
