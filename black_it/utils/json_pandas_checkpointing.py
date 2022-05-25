# Black-box ABM Calibration Kit (Black-it)
# Copyright (C) 2021-2022 Banca d'Italia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""This module contains serialization and deserialization of calibration state with Pandas."""
import gzip
import json
import pickle  # nosec B403
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from black_it.loss_functions.base import BaseLoss
from black_it.samplers.base import BaseSampler
from black_it.utils.base import NumpyArrayEncoder, PathLike


def load_calibrator_state(checkpoint_path: PathLike, _code_state_version: int) -> Tuple:
    """
    Load calibrator data from a given folder.

    Args:
        checkpoint_path: the folder where the data are stored
        _code_state_version: the serialization version (unused)

    Returns:
        all the data needed to reconstruct the calibrator state
    """
    checkpoint_path = Path(checkpoint_path)
    with open(checkpoint_path / "calibration_params.json", "r") as f:
        cp = json.load(f)

    cr = pd.read_csv(checkpoint_path / "calibration_results.csv")

    params_samp_list = []

    for i in range(len(cp["parameters_precision"])):
        params_samp_list.append(cr[f"params_samp_{i}"])

    params_samp = np.vstack(params_samp_list).T

    with open(checkpoint_path / "samplers_pickled.pickle", "rb") as fb:
        samplers = pickle.load(fb)  # nosec B301

    with open(checkpoint_path / "loss_function_pickled.pickle", "rb") as fb:
        loss_function = pickle.load(fb)  # nosec B301

    with gzip.GzipFile(checkpoint_path / "series_samp.npy.gz", "rb") as gzf:
        series_samp = np.load(gzf)

    return (
        # initialization parameters
        np.asarray(cp["parameters_bounds"]),
        np.asarray(cp["parameters_precision"]),
        np.asarray(cp["real_data"]),
        cp["ensemble_size"],
        cp["N"],
        cp["D"],
        cp["convergence_precision"],
        cp["verbose"],
        cp["saving_file"],
        cp["model_seed"],
        cp["model_name"],
        samplers,
        loss_function,
        cp["current_batch_index"],
        cp["n_sampled_params"],
        cp["n_jobs"],
        # calibration results
        params_samp,
        cr["losses_samp"].to_numpy(),
        series_samp,
        cr["batch_num_samp"].to_numpy(),
        cr["method_samp"].to_numpy(),
    )


def save_calibrator_state(  # pylint: disable=too-many-arguments,too-many-locals
    checkpoint_path: PathLike,
    parameters_bounds: NDArray[np.float64],
    parameters_precision: NDArray[np.float64],
    real_data: NDArray[np.float64],
    ensemble_size: int,
    N: int,
    D: int,
    convergence_precision: Optional[float],
    verbose: bool,
    saving_file: Optional[str],
    model_seed: int,
    model_name: str,
    samplers: Sequence[BaseSampler],
    loss_function: BaseLoss,
    current_batch_index: int,
    n_sampled_params: int,
    n_jobs: int,
    params_samp: NDArray[np.float64],
    losses_samp: NDArray[np.float64],
    series_samp: NDArray[np.float64],
    batch_num_samp: NDArray[np.int64],
    method_samp: NDArray[np.int64],
) -> None:
    """
    Store the state of the calibrator in a given folder.

    Args:
        checkpoint_path: path to the checkpoint
        parameters_bounds: the parameters bounds
        parameters_precision: the parameters precision
        real_data: the real data
        ensemble_size: the ensemble size
        N: the number of samples
        D: the number of dimensions
        convergence_precision: the convergence precision
        verbose: the verbosity mode
        saving_file: the saving file
        model_seed: the model seed
        model_name: the model name
        samplers: the ordered list of samplers to use in the calibration
        loss_function: the loss function
        current_batch_index: the current batch index
        n_sampled_params: the number of sampled params
        n_jobs: the number of jobs
        params_samp: the sampled parameters
        losses_samp: the sampled losses
        series_samp: the sampled series
        batch_num_samp: the sampling batch number
        method_samp: the sampling method
    """
    checkpoint_path = Path(checkpoint_path)
    # create directory if needed
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    calibration_params = dict(
        parameters_bounds=parameters_bounds,
        parameters_precision=parameters_precision,
        real_data=real_data,
        ensemble_size=ensemble_size,
        N=N,
        D=D,
        convergence_precision=convergence_precision,
        verbose=verbose,
        saving_file=saving_file,
        model_seed=model_seed,
        model_name=model_name,
        current_batch_index=current_batch_index,
        n_sampled_params=n_sampled_params,
        n_jobs=n_jobs,
    )
    # save calibration parameters in a json dictionary
    with open(checkpoint_path / "calibration_params.json", "w") as f:
        json.dump(calibration_params, f, cls=NumpyArrayEncoder)

    # save instantiated samplers and loss functions
    with open(checkpoint_path / "samplers_pickled.pickle", "wb") as fb:
        pickle.dump(samplers, fb)

    with open(checkpoint_path / "loss_function_pickled.pickle", "wb") as fb:
        pickle.dump(loss_function, fb)

    # save calibration results into pandas dataframe
    calibration_results = dict(
        losses_samp=losses_samp.tolist(),
        batch_num_samp=batch_num_samp.tolist(),
        method_samp=method_samp.tolist(),
    )

    for d in range(params_samp.shape[1]):
        param_key_d = f"params_samp_{d}"
        calibration_results[param_key_d] = params_samp[:, d]

    df_calibration_results = pd.DataFrame.from_dict(calibration_results)
    df_calibration_results.to_csv(checkpoint_path / "calibration_results.csv")

    # all time series sampled (heavy)
    with gzip.GzipFile(checkpoint_path / "series_samp.npy.gz", mode="wb") as gzf:
        np.save(gzf, series_samp)
