#!/usr/bin/env python3
"""This is a simple example showing the main features of the library."""
import models.simple_models as md  # type: ignore

from black_it.calibrator import Calibrator
from black_it.loss_functions.msm import MethodOfMomentsLoss
from black_it.samplers.best_batch import BestBatchSampler
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.random_forest import RandomForestSampler

if __name__ == "__main__":
    true_params = [0.20, 0.20, 0.75]
    bounds = [
        [0.10, 0.10, 0.10],  # LOWER bounds
        [1.00, 1.00, 1.00],  # UPPER bounds
    ]
    bounds_step = [0.01, 0.01, 0.01]  # Step size in range between bounds

    batch_size = 8
    halton_sampler = HaltonSampler(batch_size=batch_size)
    random_forest_sampler = RandomForestSampler(batch_size=batch_size)
    best_batch_sampler = BestBatchSampler(batch_size=batch_size)

    # define a model to be calibrated
    model = md.MarkovC_KP

    # generate a synthetic dataset to test the calibrator
    N = 2000
    seed = 1
    real_data = model(true_params, N, seed)

    # define a loss
    loss = MethodOfMomentsLoss()

    # define the calibration seed
    calibration_seed = 1

    # initialize a Calibrator object
    cal = Calibrator(
        samplers=[halton_sampler, random_forest_sampler, best_batch_sampler],
        real_data=real_data,
        model=model,
        parameters_bounds=bounds,
        parameters_precision=bounds_step,
        ensemble_size=3,
        loss_function=loss,
        random_state=calibration_seed,
    )

    # calibrate the model
    params, losses = cal.calibrate(n_batches=5)

    print(f"True parameters:       {true_params}")
    print(f"Best parameters found: {params[0]}")
