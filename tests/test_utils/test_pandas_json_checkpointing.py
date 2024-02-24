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
"""This module contains tests for the pandas_json_checkpointing.py module."""
import os
from pathlib import Path

import numpy as np

from black_it.utils.json_pandas_checkpointing import (
    load_calibrator_state,
    save_calibrator_state,
)


def test_save_and_load_calibrator_state(  # noqa: PLR0915
    rng: np.random.Generator,
) -> None:
    """Test the 'save_calibrator_state' and 'load_calibrator_state' functions."""
    parameters_bounds = np.array([[0, 1], [0, 1]]).T
    parameters_precision = np.array([0.01, 0.01])
    real_data = rng.standard_normal(size=(100, 5))
    ensemble_size = 5
    N = 30  # noqa: N806
    D = 2  # noqa: N806
    convergence_precision = 0.1
    verbose = True
    saving_folder = "saving_folder"
    initial_random_seed = 0
    random_generator_state = np.random.default_rng(
        initial_random_seed,
    ).bit_generator.state
    model_name = "model"
    samplers = ["method_a", "method_b"]  # list of objects
    loss_function = "loss_a"  # object
    current_batch_index = 10
    n_sampled_params = 10
    n_jobs = 1

    params_samp = rng.random(size=(10, 2))
    losses_samp = rng.random(size=10)
    series_samp = rng.random(size=(10, 100, 5))
    batch_num_samp = np.arange(10)
    method_samp = np.arange(10)

    save_calibrator_state(
        "saving_folder",
        parameters_bounds,
        parameters_precision,
        real_data,
        ensemble_size,
        N,
        D,
        convergence_precision,
        verbose,
        saving_folder,
        initial_random_seed,
        random_generator_state,
        model_name,
        samplers,  # type: ignore[arg-type]
        loss_function,  # type: ignore[arg-type]
        current_batch_index,
        n_sampled_params,
        n_jobs,
        params_samp,
        losses_samp,
        series_samp,
        batch_num_samp,
        method_samp,
    )

    loaded_state = load_calibrator_state(os.path.realpath("saving_folder"), 1)

    # initialization parameters
    assert np.allclose(loaded_state[0], parameters_bounds)
    assert np.allclose(loaded_state[1], parameters_precision)
    assert np.allclose(loaded_state[2], real_data)
    assert np.allclose(loaded_state[3], ensemble_size)
    assert np.allclose(loaded_state[4], N)
    assert np.allclose(loaded_state[5], D)
    assert np.allclose(loaded_state[6], convergence_precision)
    assert np.allclose(loaded_state[7], verbose)
    assert loaded_state[8] == saving_folder
    assert np.allclose(loaded_state[9], initial_random_seed)
    assert loaded_state[10] == random_generator_state
    assert loaded_state[11] == model_name
    assert loaded_state[12] == samplers
    assert loaded_state[13] == loss_function
    assert np.allclose(loaded_state[14], current_batch_index)
    assert np.allclose(loaded_state[15], n_sampled_params)
    assert np.allclose(loaded_state[16], n_jobs)

    # results of calibration
    assert np.allclose(loaded_state[17], params_samp)
    assert np.allclose(loaded_state[18], losses_samp)
    assert np.allclose(loaded_state[19], series_samp)
    assert np.allclose(loaded_state[20], batch_num_samp)
    assert np.allclose(loaded_state[21], method_samp)

    # remove the test folder
    saving_folder_path = Path("saving_folder")
    files = saving_folder_path.glob("./*")
    for f in files:
        f.unlink(missing_ok=False)
    saving_folder_path.rmdir()
