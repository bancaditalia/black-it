from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from numpy.typing import NDArray


def MCS_TO_WRITE(
    params: NDArray[float], losses: NDArray[float], verbose: bool = True
) -> None:
    # this could be a wrapper of MCS_core taking either the parameters and the losses
    # as two arrays, or directly the folder where the calibration was saved
    for p, l in zip(params, losses):
        pass

    return None


def MCS_test(
    means: NDArray[float], variances: NDArray[float], N: int, A: float = 1
) -> Tuple:
    """The main statistical test of the MCS scheme.

    Args:
        means: the mean value of the losses corresponding to the same parameter
        variances: the variance of the losses corresponding to the same parameter
        N: number of losses computed for each parameter
        A: #TODO: complete description here
        verbose: the verbosity level

    Returns:
        test: the value of the test statistic
        quan: #TODO: precise description
        p_value: the p-value corresponding to the confidence set
        stop: the stopping criterion for the MCS scheme
    """
    M = len(means)

    # if M > 2 run the test in vectorised form
    if M > 2:
        A_mat = np.hstack((np.ones((M - 1, 1)), np.diag(-np.ones(M - 1))))
        var_mat = np.diag(variances[1:])
        var_mat = var_mat + variances[0]
        target = A_mat.dot(means)
        inv_var_mat = np.linalg.inv(var_mat)
        test = N * np.dot(target.T, np.dot(inv_var_mat, target))

    # if M == 2 run a simpler version of the test
    elif M == 2:
        test = N * (means[0] - means[1]) ** 2 / (variances[0] + variances[1])

    # TODO: is this functionally correct?

    # if M == 1, the test does not make sense
    else:
        test = -np.Inf

    if M > 1:
        quan = ss.chi2.ppf(1 - A, M - 1)
        p_value = 1 - ss.chi2.cdf(test, M - 1)
    # if M == 1 set the p_value to 1
    else:
        quan = 0
        p_value = 1

    if test > quan:
        stop = 1
    else:
        stop = 0

    return test, quan, p_value, stop


def MCS_elim(
    means: NDArray[float], variances: NDArray[float], indices: NDArray[int]
) -> Tuple:
    """The elimination rule in the MCS scheme.

    Args:
        means: the mean value of the losses corresponding to the same parameter
        variances: the variance of the losses corresponding to the same parameter
        indices: sequential indices corresponding to the different parameters

    Returns:
        the new means, variances and indices array, as well as the eliminated index
    """
    to_eliminate = np.argmax(means)  # elimination rule ($\arg\max(\widehat{i})$)

    means = np.delete(means, to_eliminate)
    variances = np.delete(variances, to_eliminate)
    elim_index = indices[to_eliminate]
    indices = np.delete(indices, to_eliminate)

    return means, variances, indices, elim_index


def MCS_core(
    means: NDArray[float],
    variances: NDArray[float],
    indices: NDArray[int],
    N: int,
    A: float,
    verbose: bool = True,
) -> Tuple:
    """The Model Confidence Sets statistical analysis of Seri et al. (see reference).

    The losses corresponding to different parameter values are analysed in search of the
    set of parameters that are statistically optimal with respect to their loss values.

    Args:
        means: the mean value of the losses corresponding to the same parameter
        variances: the variance of the losses corresponding to the same parameter
        indices: sequential indices corresponding to the different parameters
        N: number of losses computed for each parameter
        A: #TODO: complete description here
        verbose: the verbosity level

    Returns:
        indices: the indices of the parameters sequentially selected by the MCS elimination procedure
        p_value: the p-values corresponding to each of the eliminated parameters

    Reference:
        R. Seri, M. Martinoli, D. Secchi, S. Centorrino, Model calibration and validation via confidence sets,
        Econometrics and Statistics 20 (2021) 62–86
    """
    I = len(indices)
    _, _, _, stop = MCS_test(means, variances, N)
    mPValue = np.zeros((len(means), 2))

    for i in range(I):

        test, quan, p_value, stop = MCS_test(means, variances, N, A)

        if stop != 1:
            break  # Stopping rule

        means, variances, indices, elim_index = MCS_elim(means, variances, indices)

        if verbose == True:
            print("Eliminated configurations: ", elim_index, "; ", "p-value: ", p_value)
            print("Remaining configurations:", indices)

        mPValue[i, :] = np.array([elim_index, p_value])

    if stop != 1:
        # compute the (nonsymmetric) set difference of subsets of a probability space.
        mPValue[-1, 0] = (
            set(indices) - set(mPValue[:15, 0])
        ).pop()  # last index yet to be added
        mPValue[-1, 1] = 1
    else:
        # TODO: is this right?
        mPValue = mPValue[0 : (I - 2), :]
        mPValue[:, 1] = np.maximum.accumulate(mPValue[:, 1])

    return mPValue[:, 0].astype(int), mPValue[:, 1]


if __name__ == "__main__":

    ### START LOSS FUNCTIONS COMPUTATION ###
    target_time_series = 300 + 4.9 * np.arange(1, 302)

    import pandas as pd

    data = pd.read_csv("Data.csv")
    time_series = data["pr.solved2"]
    run = data["run"]

    # refactor to matrix 16 * 301 * 200
    time_series_mat = np.zeros((16, 301, 200))
    n = 0
    for i in range(16):
        for k in range(200):
            run_k = run[n]
            for j in range(301):
                # print(i, j, k, time_series[n], run[n])
                if run[n] == run_k:
                    time_series_mat[i, j, k] = time_series[n]
                    n += 1
                elif run[n] == run_k + 1:
                    time_series_mat[i, j, k] = time_series[n - 1]
                else:
                    print("ERROR", run[n], k + 2)
                    quit()

    diffs = abs(time_series_mat - target_time_series[None, :, None])
    losses = np.sum(diffs**2, axis=1) / (301 * 1000 * 1000)

    means = np.mean(losses, axis=1)
    variances = np.var(losses, axis=1)
    indices = np.arange(16)

    ### END LOSS FUNCTIONS COMPUTATION ###

    N = losses.shape[1]
    elim_indices, p_values = MCS_core(means, variances, indices, N, A=1)

    plt.bar(np.arange(16), p_values, width=0.1)
    plt.ylabel("MCS p-values")
    plt.xlabel("Configurations of parameters")
    plt.ylim(0, 0.07)
    plt.xticks(np.arange(16), elim_indices)
    plt.show()
