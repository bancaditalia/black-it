# How to interface Black-it with your simulator

As already mentioned, the tool was designed with the ability to be customizable in each of its components. 
Being able to use a custom model is even more vital to the usability of the tool, since one may want to plug 
their own model in the tool and calibrate it.

A possibility is to write a custom Python function and use it as explained [here](index.md#model). 
But one may have already written their own model simulator somewhere else and may reasonably expect to use it 
with the calibrator.

To do this, essentially a Python wrapper of the simulator must be written. To illustrate this, we will take a look 
at the [SIR model tutorial](finding_the_parameters_of_a_SIR_model.ipynb) which makes use of a C++ simulator 
run on a [Docker](https://docs.docker.com/get-docker/) container. Here the wrapper is defined as follows:

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

