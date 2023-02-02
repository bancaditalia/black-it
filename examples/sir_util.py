import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plotSeries(title, SIR1, SIR2=None):
    cmap = matplotlib.cm.get_cmap("tab10").colors
    fig, ax1 = plt.subplots()
    ax1.plot(SIR1[:, 0], "-", label="susceptible", color=cmap[0])
    ax1.plot(SIR1[:, 1], "-", label="infected", color=cmap[1])
    ax1.plot(SIR1[:, 2], "-", label="recovered", color=cmap[2])
    ax1.tick_params(axis="y", labelcolor=cmap[0])
    ax1.set_ylabel("susceptible per 1000 inhabitants", color=cmap[0])
    ax1.set_xlabel("weeks")
    ax1.legend(loc=5)
    if SIR2 is not None:
        ax1.plot(SIR2[:, 0], "--", label="susceptible", color=cmap[0])
        ax1.plot(SIR2[:, 1], "--", label="infected", color=cmap[1])
        ax1.plot(SIR2[:, 2], "--", label="recovered", color=cmap[2])

    ax1.title.set_text(title)
    fig.tight_layout()
    plt.show()


def printBestParams(params):
    with np.printoptions(suppress=True, formatter={"float_kind": "{:.3f}   ".format}):
        print(f"                  brktime  beta1    beta2    gamma")
        print(f"Best parameters: {params[0]}")
