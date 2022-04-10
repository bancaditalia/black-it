#!/usr/bin/env python3
"""SIR model calibration example using a Docker-based simulator."""
import numpy as np
from models.sir.sir_docker import SIR  # type: ignore

from black_it.calibrator import Calibrator
from black_it.loss_functions.minkowski import MinkowskiLoss
from black_it.samplers.best_batch import BestBatchSampler
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.r_sequence import RSequenceSampler
from black_it.samplers.random_forest import RandomForestSampler
from black_it.samplers.random_uniform import RandomUniformSampler

if __name__ == "__main__":
    true_params = [0.2, 0.15]
    bounds = [[0, 0], [1.00, 1.00]]  # UPPER bounds
    bounds_step = [0.01, 0.01]  # Step size in range between bounds

    batch_size = 8
    random_sampler = RandomUniformSampler(batch_size=batch_size, internal_seed=0)
    rseq_sampler = RSequenceSampler(batch_size=batch_size, internal_seed=0)
    halton_sampler = HaltonSampler(batch_size=batch_size, internal_seed=0)
    best_batch_sampler = BestBatchSampler(batch_size=batch_size, internal_seed=0)
    random_forest_sampler = RandomForestSampler(batch_size=batch_size, internal_seed=0)

    # generate a synthetic dataset to test the calibrator
    N = 10
    seed = 10000
    real_data = SIR(true_params, N, seed)

    # define the calibration seed
    calibration_seed = 1

    # initialize a loss function
    loss = MinkowskiLoss()

    # initialize a Calibrator object
    cal = Calibrator(
        samplers=[
            random_sampler,
            rseq_sampler,
            halton_sampler,
            random_forest_sampler,
            best_batch_sampler,
        ],
        loss_function=loss,
        real_data=real_data,
        model=SIR,
        parameters_bounds=np.asarray(bounds),
        parameters_precision=np.asarray(bounds_step),
        ensemble_size=2,
        saving_folder=None,
        random_state=calibration_seed,
    )

    # calibrate the model
    params, losses = cal.calibrate(n_batches=2)

    print(len(params))
