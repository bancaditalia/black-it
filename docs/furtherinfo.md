# Further reading

## How to interface with external simulators

As already mentioned, the tool was designed with the ability to be customizable in each of its components. Being able to use a custom model is even more vital to the usability of the tool, since one may want to plug their own model in the tool and calibrate it.

A possibility is to write a custom Python function and use it as explained [here](index.md#model). But one may have already written their own model simulator somewhere else and may reasonably expect to use it with the calibrator.

To do this, essentially a Python wrapper of the simulator must be written. To illustrate this, we will take a look at the [SIR model tutorial](finding_the_parameters_of_a_SIR_model.ipynb) which makes use of a C++ simulator run on a [Docker](https://docs.docker.com/get-docker/) container. Here the wrapper is defined as follows:

```python
def SIR_docker(theta, N, rndSeed):
    sim_params = {
        "agents": 1000,
        "epochs": N - 1,
        "beta": theta[0],
        "gamma": theta[1],
        "infectious-t0": 10,
        "lattice-order": 20,
        "rewire-probability": 0.2,
    }

    res = simlib.execute_simulator("bancaditalia/abmsimulator", sim_params)
    ret = np.array([(x["susceptible"], x["infectious"], x["recovered"]) for x in res])

    return ret
```

To function correctly, the wrapper must have the interface shown in the example:

- it must accept in input three parameters `(theta, N, rndSeed)`, which are respectively the parameter vector to be optimized, the length of the time series and a random seed (this last one is not used in the example)

- it must output a $N \times D$ array where $N$ is the same as before and $D$ is the vector dimension of the time series (in the SIR example it is $D=3$)


## Save and load

For each batch execution of the calibration, a checkpoint can be created (*save*) so that it can be later restored (*load*).

The functions responsible for these features are the method `create_checkpoint(file_name)` and the class method `restore_from_checkpoint(checkpoint_path, model)` of the `Calibrator` class.

The checkpoint is created automatically at each batch execution, if a saving folder is provided. To do this, when a `Calibrator` object is instantiated, the `saving_folder` field must be specified as a string which contains the path where the checkpoint must be saved.

*Remark*: there's no need to call the save method, once the saving folder is specified.

To restore the checkpoint it suffices to call the load class method with the path name and, optionally, the model which was being calibrated. Once restored, the calibration starts from where it was interrupted.

An example of save and load is as follows:
```python
# initialize a Calibrator object
    cal = Calibrator(
        samplers=[
            random_sampler,
            halton_sampler,
        ],
        real_data=real_data,
        model=model,
        parameters_bounds=bounds,
        parameters_precision=bounds_step,
        ensemble_size=2,
        loss_function=loss,
        saving_folder="saving_folder",
        n_jobs=1,
    )

    _, _ = cal.calibrate(2)

    cal_restored = Calibrator.restore_from_checkpoint("saving_folder", model=model)
```

This code has been extracted from the following example: **`tests/test_calibrator_restore_from_checkpoint.py`**.

*Remark*: the saving folder where the checkpoint is saved can also be used as a parameter for the plotting functions, as shown in the [plotting tutorial](overview_of_plotting_functions.ipynb), to produce plots quickly.

## Parallelization

Since calibrating an agent-based model can be very intensive, by default the model simulation is parallelized. The number of parallel processes equals the number of cores in the computer. This number can be changed by specifying the optional parameter `n_jobs` in the constructor of the calibrator.
