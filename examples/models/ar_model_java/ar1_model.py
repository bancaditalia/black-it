import os
import subprocess

import numpy as np


def ar1_model(theta, N, rndSeed=0):
    """Autoregressive model.

    A simple AR(1) model. This is a Python wrapper for an underlying
        Java implementation. In order to run this code you need to first
        compile the Java code via "javac ARModel.java".

    Args:
        theta: the two parameters of the process
        N: the length of the generated time series
        rndSeed: the random seed of the simulation

    Returns:
        the generated time series
    """

    # AR(1) constant term
    const = theta[0]
    # AR(1) multiplicative term
    mul_par = theta[1]

    # the path of the Java executable
    file_path = os.path.realpath(os.path.dirname(__file__))

    command = (
        "java -classpath "
        + file_path
        + " ARModel {} {} {} {}".format(const, mul_par, N, rndSeed)
    )

    res = subprocess.run(
        command.split(),
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    stdout = res.stdout

    # remove first lines and last line
    lines = stdout.split("\n")

    # parse the result of the simulation
    time_series = []
    for line in lines[:-1]:

        splitted_line = line.split()
        time_series.append(float(splitted_line[-1]))

    time_series = np.array([time_series]).T

    return time_series


def ar1_model_not_random(theta, N, rndSeed=0):
    """Autoregressive model.

    A simple AR(1) model. This is a Python wrapper for an underlying
        Java implementation. In order to run this code you need to first
        compile the Java code via "javac ARModel.java".

    Args:
        theta: the two parameters of the process
        N: the length of the generated time series
        rndSeed: the random seed of the simulation

    Returns:
        the generated time series
    """

    # AR(1) constant term
    const = theta[0]
    # AR(1) multiplicative term
    mul_par = theta[1]

    # the path of the Java executable
    file_path = os.path.realpath(os.path.dirname(__file__))

    # fixed seed to zero in this
    seed = 0
    command = (
        "java -classpath "
        + file_path
        + " ARModel {} {} {} {}".format(const, mul_par, N, rndSeed)
    )

    res = subprocess.run(
        command.split(),
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    stdout = res.stdout

    # remove first lines and last line
    lines = stdout.split("\n")

    # parse the result of the simulation
    time_series = []
    for line in lines[:-1]:
        splitted_line = line.split()
        time_series.append(float(splitted_line[-1]))

    time_series = np.array([time_series]).T

    return time_series


if __name__ == "__main__":
    results = ar1_model([0.0, 1.0], 10, 3)

    print(results)
    print(results.shape)
