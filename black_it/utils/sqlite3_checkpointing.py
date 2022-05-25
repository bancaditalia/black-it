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

"""This module contains serialization and deserialization of calibration state with SQLite."""
import gzip
import io
import pickle  # nosec B403
import sqlite3
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from black_it.loss_functions.base import BaseLoss
from black_it.samplers.base import BaseSampler
from black_it.utils.base import PathLike

SCHEMA_VERSION = 2
"""
Encodes the version of a checkpoint, preventing to load incompatible versions.

Each time the structure of the SQL changes, this number has to be incremented.
A calibrator whose schema version is X will not be able to load checkpoints
saved with schema version Y, unless conversion mechanisms are in place.
"""

SQL_LOAD_QUERY = """
    SELECT
        parameters_bounds,
        parameters_precision,
        real_data,
        ensemble_size,
        N,
        D,
        convergence_precision,
        verbose,
        saving_folder,
        model_seed,
        model_name,
        samplers_pickled,
        loss_function_pickled,
        current_batch_index,
        params_samp,
        losses_samp,
        series_samp,
        batch_num_samp,
        method_samp
    FROM checkpoint
"""

SQL_SAVE_QUERY = """
    INSERT INTO checkpoint (
        parameters_bounds,
        parameters_precision,
        real_data,
        ensemble_size,
        N,
        D,
        convergence_precision,
        verbose,
        saving_folder,
        model_seed,
        model_name,
        samplers_pickled,
        loss_function_pickled,
        current_batch_index,
        params_samp,
        losses_samp,
        series_samp,
        batch_num_samp,
        method_samp
    ) VALUES (
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?
    )
"""

SQL_LOAD_USER_VERSION = """
    PRAGMA user_version
"""

SQL_SAVE_USER_VERSION = f"""
    PRAGMA user_version={SCHEMA_VERSION}
"""

SQL_DDL = """
    CREATE TABLE IF NOT EXISTS checkpoint (
        -- initialization parameters
        parameters_bounds     NDARRAY,
        parameters_precision  NDARRAY,
        real_data             NDARRAY,
        ensemble_size         INTEGER,
        N                     INTEGER,
        D                     INTEGER,
        convergence_precision DOUBLE,
        verbose               INTEGER,
        saving_folder           TEXT,
        model_seed            INTEGER,
        model_name            TEXT,
        samplers_pickled      BLOB,
        loss_function_pickled BLOB,
        -- arrays resulting from calibration
        current_batch_index   INTEGER,
        params_samp           NDARRAY,
        losses_samp           NDARRAY,
        series_samp           GZ_NDARRAY,
        batch_num_samp        NDARRAY,
        method_samp           NDARRAY
    );

    DELETE FROM checkpoint;
"""


class gz_ndarray(NDArray):
    """Convenience class to tell sqlite which ndarrays are to be compressed.

    If a user wants to compress/decompress the serialized version of an NDArray
    when saving it to the database, he can cast it to gz_ndarray without copying
    any memory:

        some_ndarray: NDArray = [1.0, 2.0]
        please_compressme = some_ndarray.view(gz_ndarray)

    please_compressme will then be a view on the same memory of some_ndarray,
    but its type will be gz_ndarray. An sqlite3 adapter will then be able to
    detect at INSERT time that the variable is of type gz_ndarray and compress
    it.
    """


def load_calibrator_state(  # pylint: disable=too-many-locals
    checkpoint_path: PathLike,
) -> Tuple:
    """
    Load the calibration state.

    Checks that the schema version stored in checkpoint_path has the same value
    of SCHEMA_VERSION. Currently there are no conversion mechanisms in place, so
    loading will only succeed if the schema versions are exactly the same.

    Args:
        checkpoint_path: the path to the checkpoint directory.

    Returns:
        a tuple with all the data that make a state.
    """
    checkpoint_path = Path(checkpoint_path)
    connection = sqlite3.connect(
        checkpoint_path / "checkpoint.sqlite", detect_types=sqlite3.PARSE_DECLTYPES
    )
    try:
        cursor = connection.cursor()

        checkpoint_schema_version: int = cursor.execute(
            SQL_LOAD_USER_VERSION
        ).fetchone()[0]
        if checkpoint_schema_version != SCHEMA_VERSION:
            raise Exception(
                f"The checkpoint you want to load has been generated with another version of the code:\n"
                f"\tCheckpoint schema version:          {checkpoint_schema_version}"
                f"\tSchema version of the current code: {SCHEMA_VERSION}"
            )

        (
            parameters_bounds,
            parameters_precision,
            real_data,
            ensemble_size,
            N,
            D,
            convergence_precision,
            verbose,
            saving_file,
            model_seed,
            model_name,
            samplers_pickled,
            loss_function_pickled,
            current_batch_index,
            params_samp,
            losses_samp,
            series_samp,
            batch_num_samp,
            method_samp,
        ) = cursor.execute(SQL_LOAD_QUERY).fetchone()

        samplers = pickle.loads(samplers_pickled)  # nosec B301
        loss_function = pickle.loads(loss_function_pickled)  # nosec B301

        return (
            parameters_bounds,
            parameters_precision,
            real_data,
            ensemble_size,
            N,
            D,
            convergence_precision,
            verbose,
            saving_file,
            model_seed,
            model_name,
            samplers,
            loss_function,
            current_batch_index,
            params_samp,
            losses_samp,
            series_samp,
            batch_num_samp,
            method_samp,
        )

    except BaseException as err:  # pylint: disable=broad-except
        raise err from err
    finally:
        connection.close()


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
    params_samp: NDArray[np.float64],
    losses_samp: NDArray[np.float64],
    series_samp: NDArray[np.float64],
    batch_num_samp: NDArray[np.int64],
    method_samp: NDArray[np.int64],
) -> None:
    """
    Save a calibration state.

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

    # pickle some files
    samplers_pickled = pickle.dumps(samplers)
    loss_function_pickled = pickle.dumps(loss_function)

    connection = sqlite3.connect(
        checkpoint_path / "checkpoint.sqlite", detect_types=sqlite3.PARSE_DECLTYPES
    )

    try:
        cursor = connection.cursor()
        cursor.execute(SQL_SAVE_USER_VERSION)
        cursor.executescript(SQL_DDL)

        cursor.execute(
            SQL_SAVE_QUERY,
            (
                parameters_bounds,
                parameters_precision,
                real_data,
                ensemble_size,
                N,
                D,
                convergence_precision,
                verbose,
                saving_file,
                model_seed,
                model_name,
                samplers_pickled,
                loss_function_pickled,
                current_batch_index,
                params_samp,
                losses_samp,
                series_samp.view(gz_ndarray),  # require compression on series_samp
                batch_num_samp,
                method_samp,
            ),
        )

        connection.commit()

    except BaseException as err:  # pylint: disable=broad-except
        connection.rollback()
        raise err from err
    finally:
        connection.close()


def npndarray_to_sqlite_binary(numpy_array: NDArray) -> sqlite3.Binary:
    """
    Serialize an NumPy NDArray to SQLite binary.

    Taken from: http://stackoverflow.com/a/31312102/190597 (SoulNibbler)

    Args:
        numpy_array: the numpy array to be serialized.

    Returns:
        the serialized NumPy array.
    """
    out: io.BytesIO = io.BytesIO()
    np.save(out, numpy_array)
    out.seek(0)
    return sqlite3.Binary(out.read())


def sqlite_binary_to_npndarray(sqlite_binary: bytes) -> NDArray:
    """
    Deserialize a NumPy NDArray from SQLite binary.

    Taken from: http://stackoverflow.com/a/31312102/190597 (SoulNibbler)

    Args:
        sqlite_binary: the SQLite binary version of the array.

    Returns:
        the deserialized NumPy NDArray.
    """
    out: io.BytesIO = io.BytesIO(sqlite_binary)
    out.seek(0)
    out = io.BytesIO(out.read())
    return np.load(out)


def gz_ndarray_to_gzipped_sqlite_binary(numpy_array: gz_ndarray) -> sqlite3.Binary:
    """
    Serialize a NumPy NDArray to a gzipped SQLite binary.

    The custom gz_ndarray type is a convenience type used to trigger the
    sqlite3 adapter mechanism. If a user coerces a NDArray to gz_ndarray, it
    will be automatically gzipped after serialization.

    Args:
        numpy_array: the numpy array to be serialized.

    Returns:
        the compressed serialized NumPy array.
    """
    out: io.BytesIO = io.BytesIO()

    with gzip.GzipFile(fileobj=out, mode="wb") as gzf:
        np.save(gzf, numpy_array)
    out.seek(0)
    return sqlite3.Binary(out.read())


def gzipped_sqlite_binary_to_npndarray(sqlite_binary: bytes) -> NDArray:
    """
    Deserialize a NumPy NDArray from a gzipped SQLite binary.

    Args:
        sqlite_binary: the gzipped SQLite binary version of the array.

    Returns:
        the deserialized NumPy NDArray.
    """
    buf: io.BytesIO = io.BytesIO(sqlite_binary)
    buf.seek(0)
    buf = io.BytesIO(buf.read())
    with gzip.GzipFile(fileobj=buf, mode="rb") as gzf:
        return np.load(gzf)


# Converts an np.array to a BLOB when INSERTing
sqlite3.register_adapter(np.ndarray, npndarray_to_sqlite_binary)

# Converts a BLOB to an np.ndarray when SELECTing
sqlite3.register_converter("NDARRAY", sqlite_binary_to_npndarray)

# Converts the custom type gz_ndarray to a gzipped BLOB when INSERTing
sqlite3.register_adapter(gz_ndarray, gz_ndarray_to_gzipped_sqlite_binary)

# Converts a gzipped BLOB to an np.ndarray when SELECTing
sqlite3.register_converter("GZ_NDARRAY", gzipped_sqlite_binary_to_npndarray)
