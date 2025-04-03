# Black-box ABM Calibration Kit (Black-it)
# Copyright (C) 2021-2024 Banca d'Italia
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
from __future__ import annotations

import json
import pickle  # nosec B403
from pathlib import Path
from typing import TYPE_CHECKING

import h5py  # type: ignore[import]
import numpy as np
import pandas as pd  # type: ignore[import]

from black_it.utils.base import NumpyArrayEncoder, PathLike

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from black_it.loss_functions.base import BaseLoss
    from black_it.schedulers.base import BaseScheduler


def load_calibrator_state(checkpoint_path: PathLike, _code_state_version: int) -> tuple:
    """Load calibrator data from a given folder.

    Args:
        checkpoint_path: the folder where the data are stored
        _code_state_version: the serialization version (unused)

    Returns:
        all the data needed to reconstruct the calibrator state
    """
    checkpoint_path = Path(checkpoint_path)
    with (checkpoint_path / "calibration_params.json").open() as f:
        cp = json.load(f)

    cr = pd.read_csv(checkpoint_path / "calibration_results.csv")

    params_samp_list = [
        cr[f"params_samp_{i}"] for i in range(len(cp["parameters_precision"]))
    ]

    params_samp = np.vstack(params_samp_list).T

    with (checkpoint_path / "scheduler_pickled.pickle").open("rb") as fb:
        scheduler = pickle.load(fb)  # nosec B301

    with (checkpoint_path / "loss_function_pickled.pickle").open("rb") as fb:
        loss_function = pickle.load(fb)  # nosec B301

    series_filename = checkpoint_path / "series_samp.h5"
    with h5py.File(series_filename, mode="r") as series_file:
        # Read the entire dataset into memory
        series_samp = series_file["data"][:]

    return (
        # initialization parameters
        cp["parameters_bounds"],
        cp["parameters_precision"],
        np.asarray(cp["real_data"]),
        cp["ensemble_size"],
        cp["N"],
        cp["D"],
        cp["convergence_precision"],
        cp["verbose"],
        cp["saving_file"],
        cp["initial_random_seed"],
        cp["random_generator_state"],
        cp["model_name"],
        scheduler,
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


def save_calibrator_state(  # noqa: PLR0913
    checkpoint_path: PathLike,
    parameters_bounds: NDArray[np.float64],
    parameters_precision: NDArray[np.float64],
    real_data: NDArray[np.float64],
    ensemble_size: int,
    N: int,  # noqa: N803
    D: int,  # noqa: N803
    convergence_precision: float | None,
    verbose: bool,
    saving_file: str | None,
    initial_random_seed: int | None,
    random_generator_state: Mapping,
    model_name: str,
    scheduler: BaseScheduler,
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
    """Store the state of the calibrator in a given folder.

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
        initial_random_seed: the initial seed of the calibrator
        random_generator_state: the internal random state of the calibrator
        model_name: the model name
        scheduler: the scheduler to use in the calibration
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

    calibration_params = {
        "parameters_bounds": parameters_bounds,
        "parameters_precision": parameters_precision,
        "real_data": real_data,
        "ensemble_size": ensemble_size,
        "N": N,
        "D": D,
        "convergence_precision": convergence_precision,
        "verbose": verbose,
        "saving_file": saving_file,
        "initial_random_seed": initial_random_seed,
        "random_generator_state": random_generator_state,
        "model_name": model_name,
        "current_batch_index": current_batch_index,
        "n_sampled_params": n_sampled_params,
        "n_jobs": n_jobs,
    }
    # save calibration parameters in a json dictionary
    with (checkpoint_path / "calibration_params.json").open("w") as f:
        json.dump(calibration_params, f, cls=NumpyArrayEncoder)

    # save instantiated scheduler and loss functions
    with (checkpoint_path / "scheduler_pickled.pickle").open("wb") as fb:
        pickle.dump(scheduler, fb)

    with (checkpoint_path / "loss_function_pickled.pickle").open("wb") as fb:
        pickle.dump(loss_function, fb)

    # save calibration results into pandas dataframe
    calibration_results = {
        "losses_samp": losses_samp.tolist(),
        "batch_num_samp": batch_num_samp.tolist(),
        "method_samp": method_samp.tolist(),
    }

    for d in range(params_samp.shape[1]):
        param_key_d = f"params_samp_{d}"
        calibration_results[param_key_d] = params_samp[:, d]

    df_calibration_results = pd.DataFrame.from_dict(calibration_results)
    df_calibration_results.to_csv(checkpoint_path / "calibration_results.csv")

    series_filename = "series_samp.h5"
    series_filepath = checkpoint_path / series_filename

    # If the HDF5 file already exists, open in append mode and add only new rows.
    if series_filepath.exists():
        with h5py.File(series_filepath, mode="a") as series_file:
            data = series_file["data"]  # Get the existing dataset
            previous_shape = data.shape  # E.g., (num_rows, dim2, dim3, ...)
            nb_rows = previous_shape[0]
            to_append = series_samp[nb_rows:]  # Slicing out only the new part

            # Resize the first dimension so there's room for the new data
            new_num_rows = nb_rows + to_append.shape[0]
            data.resize((new_num_rows,) + previous_shape[1:])

            # Write the appended portion
            data[nb_rows:new_num_rows] = to_append

        return

    # If the file does not exist, create it and store the entire dataset in one shot.
    with h5py.File(series_filepath, mode="w") as series_file:
        # Create a resizable (maxshape=None along axis 0) dataset
        data = series_file.create_dataset(
            name="data",
            data=series_samp,
            maxshape=(
                None,
                *series_samp.shape[1:],
            ),  # Resizable along the first dimension
            dtype="float64",
        )

    return
