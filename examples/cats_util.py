import pathlib
import shutil

import matplotlib.pyplot as plt

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


def cleanup_output_dir():
    path_to_model_output = pathlib.Path("").resolve() / saving_folder
    print(f"Removing {path_to_model_output}")
    shutil.rmtree(path_to_model_output, ignore_errors=True)


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
