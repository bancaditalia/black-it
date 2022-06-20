
<p align="center">
<img src="logo/logo_1024.png" alt="black-it logo" width="300"/>
<sup><a href="#footnote-1">*</a></sup>
</p>

<h1 align="center">
  <b>Black-box ABM calibration kit</b>
</h1>

This package contains a black-box calibrator, which can be used to calibrate a specified model, using a loss function and a sequence of chosen search algorithms to estimate the wanted parameters. It comes with a set of ready-to-use example models, loss functions and search algorithms. Custom models and functions can be implemented to use with the calibrator.

## Why use it
While this tool can be used as a general optimizer for any built-in or custom model, it has been created with [agent-based models](https://en.wikipedia.org/wiki/Agent-based_model) in mind, or any situation in which a simple optimization is not enough but you have to combine techniques. That's why the package is easily customizable in terms of model, loss and algorithms, and comes with the ability to combine search algorithms sequentially.

## Installation

This project requires Python v3.8 or later and [Poetry](https://python-poetry.org).

To install the package simply run

```
pip install black-it
```

## How to run

Several calibration examples can be found in the **`examples`** folder of the GitHub repo.
To run them, you first need to clone the repo

```commandline
git clone https://github.com/bancaditalia/black-it.git
```

In the next section we will analyse in detail the `main.py` script, which you can run by

```commandline
cd black-it/examples
python main.py
```

## How to use

To write a basic script, it is enough to instantiate a `Calibrator` object and use the `calibrate(n_batches)` method.
The following example will refer to **`examples/main.py`**.

```python
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
```
The calibrator constructor accepts as inputs:

1. the real dataset (`real_data`),
2. a stochastic model (`model`),
3. a loss function (`loss_function`),
4. the parameter space (`parameters_bounds`)
5. a list of search algorithms (`samplers`)

The method used to run the calibration (`calibrate`) accepts as input the number of batches to be executed (`n_batches`).
For more information, check the `Code Reference` section of the documentation.

### Model

The model must be specified as a function and is used by the calibrator to produce simulated data (for more information about this, check [how it works](description.md)). In **`examples/main.py`**, the following is used:
```python
# define a model to be calibrated
model = md.MarkovC_KP
```
A list of simple models can be found in the `examples/models` directory. A custom model may be specified by implementing a custom function.

If an external simulator has to be used instead, check [how to interface with external simulators](furtherinfo.md#how-to-interface-with-external-simulators).

### Loss function

The loss function must be a concrete class inheriting from the abstract class `BaseLoss` and is used by the calibrator to evaluate the distance between the real dataset and the simulated one. In **`examples/main.py`**, the `MinkowskiLoss` is used.

A list of functions can be found in the `loss_functions` module or, again, a the `BaseLoss` class can be extended to implement a specific loss function.

### Search algorithms

The calibrator accepts a list of search algorithms which are used sequentially to estimate the wanted parameters.
The parameter space to be searched is defined through its bounds by specifying `parameters_bounds`.

Each search algorithm must be specified as an object and must be instantiated first. In this example,
```python
batch_size = 8
halton_sampler = HaltonSampler(batch_size=batch_size)
random_forest_sampler = RandomForestSampler(batch_size=batch_size)
best_batch_sampler = BestBatchSampler(batch_size=batch_size)
```
Each sampler has its own subclass derived from `BaseSampler` and a list of ready-to-use samplers is contained in `samplers`.
To specify a custom algorithm, one must extend the `BaseSampler` superclass and implement its abstract method `single_sample` to specify how to sample a single parameter. If this is not possible and one wants to specify the whole sampling strategy, then they must redefine the `sample` method directly. This is the case, for example, for the `RandomForestSampler`.

*Remark*: when instantiated, the sampler accepts a `batch_size` parameter.
While in this example every sampler runs on the same batch size, they can also run on different sizes, if required.

## License

Black-it is released under the GNU Affero General Public License v3 or later (AGPLv3+).

Copyright 2021-2022 Banca d'Italia.

## Original Author

- [Gennaro Catapano](https://github.com/CatapanoG) <[gennaro.catapano@bancaditalia.it](mailto:gennaro.catapano@bancaditalia.it)>

## Co-authors/Maintainers

- [Marco Benedetti](https://github.com/mabene-BI) <[marco.benedetti@bancaditalia.it](mailto:marco.benedetti@bancaditalia.it)>
- [Francesco De Sclavis](https://github.com/Francesco-De-Sclavis-BdI) <[francesco.desclavis@bancaditalia.it](mailto:francesco.desclavis@bancaditalia.it)>
- [Marco Favorito](https://github.com/marcofavoritobi) <[marco.favorito@bancaditalia.it](mailto:marco.favorito@bancaditalia.it)>
- [Aldo Glielmo](https://github.com/AldoGl) <[aldo.glielmo@bancaditalia.it](mailto:aldo.glielmo@bancaditalia.it)>
- [Davide Magnanimi](https://github.com/davidemagnanimi) <[davide.magnanimi@bancaditalia.it](mailto:davide.magnanimi@bancaditalia.it)>
- [Antonio Muci](https://github.com/muxator) <[antonio.muci@bancaditalia.it](mailto:antonio.muci@bancaditalia.it)>

---

<p id="footnote-1">
* Credits to <a href="https://www.bankit.art/people/sara-corbo">Sara Corbo</a> for the logo.
</p>
