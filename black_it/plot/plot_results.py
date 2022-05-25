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

"""This module contains utilities for plotting results."""
import os
import pickle  # nosec B403
from typing import Collection, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ipywidgets import fixed, interact

from black_it.calibrator import Calibrator


def _get_samplers_id_table(saving_folder: Union[str, os.PathLike]) -> Dict[str, int]:
    """Get the id table of the samplers from the checkpoint.

    Args:
        saving_folder: the folder where the calibration results were saved

    Returns:
        the id table of the samplers
    """
    with open(os.path.join(saving_folder, "samplers_pickled.pickle"), "rb") as f:
        method_list = pickle.load(f)  # nosec B301

    samplers_id_table = (
        Calibrator._construct_samplers_id_table(  # pylint: disable=protected-access
            method_list
        )
    )

    return samplers_id_table


def _get_samplers_names(
    saving_folder: Union[str, os.PathLike], ids: List[int]
) -> List[str]:
    """Get the names of the samplers from their ids and from the checkpoint of the calibration.

    Args:
        saving_folder: the folder where the calibration results were saved
        ids: the ids of the samplers

    Returns:
        the names of the samplers
    """
    samplers_id_table = _get_samplers_id_table(saving_folder)

    inv_samplers_id_table = {v: k for k, v in samplers_id_table.items()}

    sampler_names = []
    for sampler_id in ids:
        sampler_names.append(inv_samplers_id_table[sampler_id])

    return sampler_names


def plot_convergence(saving_folder: Union[str, os.PathLike]) -> None:
    """Plot the loss values sampled by the various methods as a function of the batch number.

    Args:
        saving_folder: the folder where the calibration results were saved

    """
    df = pd.read_csv(os.path.join(saving_folder, "calibration_results.csv"))

    losses_cummin = df.groupby("batch_num_samp").min()["losses_samp"].cummin()

    plt.figure()
    g = sns.lineplot(
        data=df, x="batch_num_samp", y="losses_samp", hue="method_samp", palette="tab10"
    )

    g = sns.lineplot(
        x=np.arange(
            max(df["batch_num_samp"]) + 1  # pylint: disable=unsubscriptable-object
        ),
        y=losses_cummin,
        color="black",
        ls="--",
        marker="o",
        label="min loss",
    )

    ids = df["method_samp"].unique()  # pylint: disable=unsubscriptable-object
    sampler_names = _get_samplers_names(saving_folder, ids)

    handles, labels = g.get_legend_handles_labels()
    labels = sampler_names + [labels[-1]]

    plt.legend(handles, labels, loc="upper right")


def plot_losses(saving_folder: Union[str, os.PathLike]) -> None:
    """Plot the parameter sampled colored according to their loss value.

    Args:
        saving_folder: the folder where the calibration results were saved
    """
    df = pd.read_csv(os.path.join(saving_folder, "calibration_results.csv"))

    num_params = sum(["params_samp_" in c_str for c_str in df.columns])

    variables = ["params_samp_" + str(i) for i in range(num_params)]

    g = sns.pairplot(
        df,
        hue="losses_samp",
        vars=variables,
        diag_kind="hist",
        corner=True,
        palette="viridis",
        plot_kws=dict(markers="o", linewidth=1, alpha=0.8),
        diag_kws=dict(
            fill=False,
            hue=None,
        ),
    )

    g._legend.set_bbox_to_anchor((0.8, 0.5))  # pylint: disable=protected-access


def plot_sampling(saving_folder: Union[str, os.PathLike]) -> None:
    """Plot the parameter sampled colored according to the sampling method used to sample them.

    Args:
        saving_folder: the folder where the calibration results were saved
    """
    df = pd.read_csv(os.path.join(saving_folder, "calibration_results.csv"))

    num_params = sum(["params_samp_" in c_str for c_str in df.columns])
    variables = ["params_samp_" + str(i) for i in range(num_params)]

    g = sns.pairplot(
        df,
        hue="method_samp",
        vars=variables,
        diag_kind="hist",
        corner=True,
        palette="tab10",
        plot_kws=dict(markers="o", linewidth=1, alpha=0.8),
        diag_kws=dict(
            fill=False,
        ),
    )
    ids = df["method_samp"].unique()  # pylint: disable=unsubscriptable-object
    sampler_names = _get_samplers_names(saving_folder, ids)

    # take legend of the plot in the last row and first column, to be sure it's a scatter plot
    handles, _ = g.axes[-1][0].get_legend_handles_labels()
    g._legend.remove()  # pylint: disable=protected-access

    plt.legend(loc=2, handles=handles, labels=sampler_names, bbox_to_anchor=(0.0, 1.8))


def plot_losses_method_num(
    saving_folder: Union[str, os.PathLike], method_num: int
) -> None:
    """Plot the parameter sampled by a specific sampling method, and color them according to their loss value.

    Args:
        saving_folder: the folder where the calibration results were saved
        method_num: the integer value defining a specific sampling method
    """
    df = pd.read_csv(os.path.join(saving_folder, "calibration_results.csv"))

    if method_num not in set(df["method_samp"]):
        raise ValueError(f"Samplers with method_num = {method_num} was never used")

    df = df.loc[
        df["method_samp"] == method_num  # pylint: disable=unsubscriptable-object
    ]

    num_params = sum(["params_samp_" in c_str for c_str in df.columns])

    variables = ["params_samp_" + str(i) for i in range(num_params)]

    g = sns.pairplot(
        df,
        hue="losses_samp",
        vars=variables,
        diag_kind="hist",
        corner=True,
        palette="viridis",
        plot_kws=dict(markers="o", linewidth=1, alpha=0.8),
        diag_kws=dict(
            fill=False,
            hue=None,
        ),
    )

    g._legend.set_bbox_to_anchor((0.8, 0.5))  # pylint: disable=protected-access


def plot_losses_interact(saving_folder: Union[str, os.PathLike]) -> None:
    """Plot the parameter sampled colored according to their loss value.

    This plot allows to interactively choose the sampling method.

    Args:
        saving_folder: the folder where the calibration results were saved
    """
    df = pd.read_csv(os.path.join(saving_folder, "calibration_results.csv"))
    method_nums = set(df["method_samp"])

    samplers_id_table = _get_samplers_id_table(saving_folder)

    id_samplers_cal = {}

    for sampler, sampler_id in samplers_id_table.items():
        if sampler_id in method_nums:
            id_samplers_cal[sampler] = sampler_id

    interact(
        plot_losses_method_num,
        saving_folder=fixed(saving_folder),
        method_num=id_samplers_cal,
    )


def plot_sampling_batch_nums(
    saving_folder: Union[str, os.PathLike], batch_nums: Collection[int]
) -> None:
    """Plot the parameter sampled in specific batches colored according to the sampling method used to sample them.

    Args:
        saving_folder: the folder where the calibration results were saved
        batch_nums: a list of batch number
    """
    # deduplicate batch numbers
    batch_nums = set(batch_nums)
    df = pd.read_csv(os.path.join(saving_folder, "calibration_results.csv"))

    filter_bns = [bn in batch_nums for bn in df["batch_num_samp"]]

    df["filter_bns"] = filter_bns

    df_ = df.loc[
        df["filter_bns"]  # pylint: disable=singleton-comparison,unsubscriptable-object
        == True  # noqa
    ]

    num_params = sum(
        ["params_samp_" in c_str for c_str in df.columns]  # pylint: disable=no-member
    )

    variables = ["params_samp_" + str(i) for i in range(num_params)]

    g = sns.pairplot(
        df_,
        hue="method_samp",
        vars=variables,
        diag_kind="hist",
        corner=True,
        plot_kws=dict(markers="o", linewidth=1, alpha=0.8),
        diag_kws=dict(fill=False),
        palette="tab10",
    )

    ids = df["method_samp"].unique()  # pylint: disable=unsubscriptable-object
    sampler_names = _get_samplers_names(saving_folder, ids)

    # take legend of the plot in the last row and first column, to be sure it's a scatter plot
    handles, _ = g.axes[-1][0].get_legend_handles_labels()
    g._legend.remove()  # pylint: disable=protected-access

    plt.legend(loc=2, handles=handles, labels=sampler_names, bbox_to_anchor=(0.0, 1.8))


def plot_sampling_interact(saving_folder: Union[str, os.PathLike]) -> None:
    """
    Plot the parameter sampled colored according to the sampling method used to sample them.

    The method allows to interactively choose the batch numbers included in the plot.

    Args:
        saving_folder: the folder where the calibration results were saved
    """
    df = pd.read_csv(os.path.join(saving_folder, "calibration_results.csv"))
    max_bn = int(max(df["batch_num_samp"]))
    all_bns = np.arange(max_bn + 1, dtype=int)
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
