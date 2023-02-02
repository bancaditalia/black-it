import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from black_it.utils.time_series import get_mom_ts_1d

# folder where data is saved
saving_folder = "CATS_model_output"

# previously found parameters
found_params = {
    "Iprob": 0.36,
    "chi": 0.069,
    "delta": 0.956,
    "inventory_depreciation": 0.764,
    "mu": 1.3769999999999587,
    "p_adj": 0.089,
    "phi": 0.016,
    "q_adj": 0.8160000000000003,
    "tax_rate": 0.002,
    "theta": 0.044,
    "xi": 0.8810000000000003,
}

target_labels = [
    "gdp gap",
    "inflation rate",
    "investment gap",
    "consumption",
    "unemployment",
]


def load_fred_data():
    return np.genfromtxt("FRED_data.txt")


def cleanup_output_dir():
    path_to_model_output = pathlib.Path("").resolve() / saving_folder
    # do not print the removal message in the demo
    # print(f"Removing {path_to_model_output}")
    shutil.rmtree(path_to_model_output, ignore_errors=True)


def plot_gdp(time_series):
    plt.plot(time_series[:, 0])
    plt.xlabel("quarters")
    plt.ylabel("real gdp")


def plot_cats_output(time_series):
    # the target time series
    fig, axes = plt.subplots(2, 3)
    i = 0
    for line in axes:
        for ax in line:
            if i < 5:
                ax.set_title(target_labels[i])
                ax.plot(time_series[:, i], alpha=0.5)
            else:
                ax.axis("off")
            i += 1
    plt.tight_layout()


def find_best_loss_idx(losses_samp):
    # find the best loss index and print the best loss
    best_idx = np.argmin(losses_samp)
    print(f"Best loss index: {best_idx} (out of {len(losses_samp)} simulations)")
    print(f"Best loss value: {losses_samp[best_idx]:.2f}")
    return best_idx


def compare_moments(real_data, best_sim, ensemble_size, coordinate_filters):
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))

    xlabels_kde = [
        [-0.1, 0.0, 0.1],
        [-0.05, 0, 0.05],
        [-0.3, 0, 0.3],
        [-0.1, 0, 0.1],
        [0.0, 0.1, 0.2],
    ]
    clips = [(-0.1, 0.1), (-0.1, 0.1), (-0.4, 0.4), (-0.1, 0.1), (0.0, 0.25)]
    cmap = plt.get_cmap("tab10")

    for i, coordinate_filter in enumerate(coordinate_filters):
        # DATA
        if coordinate_filter is None:
            coordinate_filter = lambda x: x
        filtered_data = np.array(
            [coordinate_filter(best_sim[j, :, i]) for j in range(ensemble_size)]
        )

        real_moments = get_mom_ts_1d(real_data[:, i])
        sim_moments = np.array(
            [get_mom_ts_1d(filtered_data[j, :]) for j in range(ensemble_size)]
        )
        m = np.mean(sim_moments, axis=0)
        s = np.std(sim_moments, axis=0)

        # KDE
        ax = axes[0, i]
        ax.set_title(target_labels[i])
        sns.kdeplot(
            ax=ax,
            data=real_data[:, i],
            label="target",
            bw_adjust=1.0,
            clip=clips[i],
            lw=2,
        )
        sns.kdeplot(
            ax=ax,
            data=filtered_data.flatten(),
            color=cmap(1),
            label="lowest loss",
            bw_adjust=1.0,
            clip=clips[i],
            lw=2,
        )
        ax.set_ylabel("")
        ax.set_yticklabels("")
        ax.set_xticks(xlabels_kde[i])

        # MOMENTS
        index = np.arange(len(real_moments))
        ax = axes[1, i]
        ax.set_ylim(-1, 2.2)
        ax.plot(index, real_moments, "-o", ms=3)
        ax.plot(index, m, color=cmap(1), ls="-", marker="o", ms=3)
        ax.fill_between(index, m - s, m + s, alpha=0.25, color=cmap(1))
        ax.set_xticks([0, 4, 9, 13])
        ax.set_xlabel("")

    axes[0, 0].set_ylabel("density", labelpad=10)
    axes[1, 0].set_ylabel("moments value", labelpad=-3)
    for i in range(1, 5):
        axes[1, i].set_yticklabels("")
    plt.subplots_adjust(
        left=0.1,
        bottom=0.1,
        right=0.9,
        top=0.9,
        wspace=0.3,
        hspace=0.3,
    )
    axes[0, 4].plot([0.1, 0.15], [21, 21], lw=2, color=cmap(0))
    axes[0, 4].text(0.1, 18, "real")
    axes[0, 4].plot([0.1, 0.15], [13, 13], lw=2, color=cmap(1))
    axes[0, 4].text(0.1, 10, "simulated")
    plt.show()
