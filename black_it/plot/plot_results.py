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

"""This module contains utilities for plotting results."""
from __future__ import annotations

import pickle  # nosec B403
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from ipywidgets import fixed, interact  # type: ignore[import]

from black_it.calibrator import Calibrator

if TYPE_CHECKING:
    import os
    from collections.abc import Collection

    from numpy.typing import NDArray


def _get_samplers_id_table(saving_folder: str | os.PathLike) -> dict[str, int]:
    """Get the id table of the samplers from the checkpoint.

    Args:
        saving_folder: the folder where the calibration results were saved

    Returns:
        the id table of the samplers
    """
    output_file = Path(saving_folder) / "scheduler_pickled.pickle"
    with output_file.open("rb") as f:
        method_list = pickle.load(f)  # nosec B301

    return Calibrator._construct_samplers_id_table(method_list)  # noqa: SLF001


def _get_samplers_names(
    saving_folder: str | os.PathLike,
    ids: list[int],
) -> list[str]:
    """Get the names of the samplers from their ids and from the checkpoint of the calibration.

    Args:
        saving_folder: the folder where the calibration results were saved
        ids: the ids of the samplers

    Returns:
        the names of the samplers
    """
    samplers_id_table = _get_samplers_id_table(saving_folder)

    inv_samplers_id_table = {v: k for k, v in samplers_id_table.items()}

    return [inv_samplers_id_table[sampler_id] for sampler_id in ids]


def plot_convergence(saving_folder: str | os.PathLike) -> None:
    """Plot the loss values sampled by the various methods as a function of the batch number.

    Args:
        saving_folder: the folder where the calibration results were saved

    """
    calibration_results_file = Path(saving_folder) / "calibration_results.csv"
    data_frame = pd.read_csv(calibration_results_file)

    losses_cummin = data_frame.groupby("batch_num_samp").min()["losses_samp"].cummin()

    plt.figure()
    g = sns.lineplot(
        data=data_frame,
        x="batch_num_samp",
        y="losses_samp",
        hue="method_samp",
        palette="tab10",
    )

    g = sns.lineplot(
        x=np.arange(max(data_frame["batch_num_samp"]) + 1),
        y=losses_cummin,
        color="black",
        ls="--",
        marker="o",
        label="min loss",
    )

    ids = data_frame["method_samp"].unique()
    sampler_names = _get_samplers_names(saving_folder, ids)

    handles, labels = g.get_legend_handles_labels()
    labels = [*sampler_names, labels[-1]]

    plt.legend(handles, labels, loc="upper right")


def plot_losses(saving_folder: str | os.PathLike) -> None:
    """Plot the parameter sampled colored according to their loss value.

    Args:
        saving_folder: the folder where the calibration results were saved
    """
    calibration_results_file = Path(saving_folder) / "calibration_results.csv"
    data_frame = pd.read_csv(calibration_results_file)

    num_params = sum("params_samp_" in c_str for c_str in data_frame.columns)

    variables = ["params_samp_" + str(i) for i in range(num_params)]

    g = sns.pairplot(
        data_frame,
        hue="losses_samp",
        vars=variables,
        diag_kind="hist",
        corner=True,
        palette="viridis",
        plot_kws={"markers": "o", "linewidth": 1, "alpha": 0.8},
        diag_kws={
            "fill": False,
            "hue": None,
        },
    )

    g.legend.set_bbox_to_anchor((0.8, 0.5))


def plot_sampling(saving_folder: str | os.PathLike) -> None:
    """Plot the parameter sampled colored according to the sampling method used to sample them.

    Args:
        saving_folder: the folder where the calibration results were saved
    """
    calibration_results_file = Path(saving_folder) / "calibration_results.csv"
    data_frame = pd.read_csv(calibration_results_file)

    num_params = sum("params_samp_" in c_str for c_str in data_frame.columns)
    variables = ["params_samp_" + str(i) for i in range(num_params)]

    g = sns.pairplot(
        data_frame,
        hue="method_samp",
        vars=variables,
        diag_kind="hist",
        corner=True,
        palette="tab10",
        plot_kws={"markers": "o", "linewidth": 1, "alpha": 0.8},
        diag_kws={
            "fill": False,
        },
    )
    ids = data_frame["method_samp"].unique()
    sampler_names = _get_samplers_names(saving_folder, ids)

    # take legend of the plot in the last row and first column, to be sure it's a scatter plot
    handles, _ = g.axes[-1][0].get_legend_handles_labels()
    g.legend.remove()

    plt.legend(loc=2, handles=handles, labels=sampler_names, bbox_to_anchor=(0.0, 1.8))


def plot_losses_method_num(
    saving_folder: str | os.PathLike,
    method_num: int,
) -> None:
    """Plot the parameter sampled by a specific sampling method, and color them according to their loss value.

    Args:
        saving_folder: the folder where the calibration results were saved
        method_num: the integer value defining a specific sampling method
    """
    calibration_results_file = Path(saving_folder) / "calibration_results.csv"
    data_frame = pd.read_csv(calibration_results_file)

    if method_num not in set(data_frame["method_samp"]):
        msg = f"Samplers with method_num = {method_num} was never used"
        raise ValueError(msg)

    data_frame = data_frame.loc[data_frame["method_samp"] == method_num]

    num_params = sum("params_samp_" in c_str for c_str in data_frame.columns)

    variables = ["params_samp_" + str(i) for i in range(num_params)]

    g = sns.pairplot(
        data_frame,
        hue="losses_samp",
        vars=variables,
        diag_kind="hist",
        corner=True,
        palette="viridis",
        plot_kws={"markers": "o", "linewidth": 1, "alpha": 0.8},
        diag_kws={
            "fill": False,
            "hue": None,
        },
    )

    g.legend.set_bbox_to_anchor((0.8, 0.5))


def plot_losses_interact(saving_folder: str | os.PathLike) -> None:
    """Plot the parameter sampled colored according to their loss value.

    This plot allows to interactively choose the sampling method.

    Args:
        saving_folder: the folder where the calibration results were saved
    """
    calibration_results_file = Path(saving_folder) / "calibration_results.csv"
    data_frame = pd.read_csv(calibration_results_file)
    method_nums = set(data_frame["method_samp"])

    samplers_id_table = _get_samplers_id_table(saving_folder)

    id_samplers_cal = {}

    for sampler, sampler_id in samplers_id_table.items():
        if sampler_id in method_nums:
            id_samplers_cal[sampler] = sampler_id  # noqa: PERF403

    interact(
        plot_losses_method_num,
        saving_folder=fixed(saving_folder),
        method_num=id_samplers_cal,
    )


def plot_sampling_batch_nums(
    saving_folder: str | os.PathLike,
    batch_nums: Collection[int],
) -> None:
    """Plot the parameter sampled in specific batches colored according to the sampling method used to sample them.

    Args:
        saving_folder: the folder where the calibration results were saved
        batch_nums: a list of batch number
    """
    calibration_results_file = Path(saving_folder) / "calibration_results.csv"
    data_frame = pd.read_csv(calibration_results_file)

    # deduplicate batch numbers
    batch_nums = set(batch_nums)

    filter_bns = [bn in batch_nums for bn in data_frame["batch_num_samp"]]

    data_frame["filter_bns"] = filter_bns

    data_frame_2 = data_frame.loc[data_frame["filter_bns"] == True]  # noqa: E712

    num_params = sum("params_samp_" in c_str for c_str in data_frame.columns)

    variables = ["params_samp_" + str(i) for i in range(num_params)]

    g = sns.pairplot(
        data_frame_2,
        hue="method_samp",
        vars=variables,
        diag_kind="hist",
        corner=True,
        plot_kws={"markers": "o", "linewidth": 1, "alpha": 0.8},
        diag_kws={"fill": False},
        palette="tab10",
    )

    ids = data_frame["method_samp"].unique()
    sampler_names = _get_samplers_names(saving_folder, ids)

    # take legend of the plot in the last row and first column, to be sure it's a scatter plot
    handles, _ = g.axes[-1][0].get_legend_handles_labels()
    g.legend.remove()

    plt.legend(loc=2, handles=handles, labels=sampler_names, bbox_to_anchor=(0.0, 1.8))


def plot_sampling_interact(saving_folder: str | os.PathLike) -> None:
    """Plot the parameter sampled colored according to the sampling method used to sample them.

    The method allows to interactively choose the batch numbers included in the plot.

    Args:
        saving_folder: the folder where the calibration results were saved
    """
    calibration_results_file = Path(saving_folder) / "calibration_results.csv"
    data_frame = pd.read_csv(calibration_results_file)

    max_bn = int(max(data_frame["batch_num_samp"]))
    all_bns: NDArray[np.int64] = np.arange(max_bn + 1, dtype=int)
    indices_bns = np.array_split(all_bns, min(max_bn, 3))

    dict_bns = {}
    for index_bns in indices_bns:
        key = "from " + str(index_bns[0]) + " to " + str(index_bns[-1])
        dict_bns[key] = index_bns.tolist()

    interact(
        plot_sampling_batch_nums,
        saving_folder=fixed(saving_folder),
        batch_nums=dict_bns,
    )
