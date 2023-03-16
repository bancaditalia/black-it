# Black-box ABM Calibration Kit (Black-it)
# Copyright (C) 2021-2023 Banca d'Italia
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
"""This module contains tests for the SQLite3-based checkpointing."""
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import black_it.utils.sqlite3_checkpointing
from black_it.utils.sqlite3_checkpointing import (
    load_calibrator_state,
    save_calibrator_state,
)


def test_sqlite3_checkpointing() -> None:  # pylint: disable=too-many-locals
    """Test SQLite3 checkpointing."""
    parameters_bounds = np.array([1.0, 2.0, 3.0])
    parameters_precision = np.array([0.1, 0.1, 0.1])
    real_data = np.linspace(0, 30)
    ensemble_size = 5
    N = 30
    D = 3
    convergence_precision = 0.1
    verbose = True
    saving_file = "test"
    initial_random_seed = 0
    random_generator_state = np.random.default_rng(
        initial_random_seed
    ).bit_generator.state
    model_name = "model"
    samplers = ["method_a", "method_b"]  # list of objects
    loss_function = "loss_a"  # object
    current_batch_index = 0
    params_samp = np.array([1.0, 2.0, 3.0])
    losses_samp = np.array([1.0, 2.0, 3.0])
    series_samp = np.array([1.0, 2.0, 3.0])
    batch_num_samp = np.array([1.0, 2.0, 3.0])
    method_samp = np.array([1.0, 2.0, 3.0])

    with TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint"
        save_calibrator_state(
            checkpoint_dir,
            parameters_bounds,
            parameters_precision,
            real_data,
            ensemble_size,
            N,
            D,
            convergence_precision,
            verbose,
            saving_file,
            initial_random_seed,
            random_generator_state,
            model_name,
            samplers,  # type: ignore
            loss_function,  # type: ignore
            current_batch_index,
            params_samp,
            losses_samp,
            series_samp,
            batch_num_samp,
            method_samp,
        )

        loaded_state = load_calibrator_state(checkpoint_dir)

        assert np.allclose(loaded_state[15], params_samp)
        assert np.allclose(loaded_state[16], losses_samp)
        assert np.allclose(loaded_state[17], series_samp)
        assert np.allclose(loaded_state[18], batch_num_samp)
        assert np.allclose(loaded_state[19], method_samp)


@patch("sqlite3.connect")
def test_sqlite3_checkpointing_loading_when_code_state_version_different(
    sqlite_connect_mock: MagicMock,
) -> None:
    """Test that loading with different code state version raises exception."""
    fake_schema_version = black_it.utils.sqlite3_checkpointing.SCHEMA_VERSION + 42

    cursor_mock = MagicMock()
    cursor_mock.execute().fetchone.return_value = [fake_schema_version]
    connection_mock = MagicMock()
    connection_mock.cursor.return_value = cursor_mock
    sqlite_connect_mock.return_value = connection_mock

    expected_message = (
        "The checkpoint you want to load has been generated with another version of the code:\n"
        f"\tCheckpoint schema version:          {fake_schema_version}"
        f"\tSchema version of the current code: {black_it.utils.sqlite3_checkpointing.SCHEMA_VERSION}"
    )
    with pytest.raises(Exception, match=expected_message):
        load_calibrator_state("")


@patch("pickle.dumps")
@patch.object(black_it.utils.sqlite3_checkpointing, "Path")
@patch("sqlite3.connect")
def test_sqlite3_checkpointing_saving_when_error_occurs(
    connect_mock: MagicMock, *_mocks: Any
) -> None:
    """Test saving function when an error occurs."""
    error_message = "error"
    connection_mock = MagicMock()
    connection_mock.cursor.side_effect = ValueError(error_message)
    connect_mock.return_value = connection_mock
    mock_args = [MagicMock()] * 21
    with pytest.raises(ValueError, match=error_message):
        save_calibrator_state(*mock_args)
