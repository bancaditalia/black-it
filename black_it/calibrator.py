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

"""This module contains the main class Calibrator."""
from __future__ import annotations

import multiprocessing
import textwrap
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence, cast

import numpy as np
from joblib import Parallel, delayed  # type: ignore[import]

from black_it.schedulers.base import BaseScheduler
from black_it.schedulers.round_robin import RoundRobinScheduler
from black_it.search_space import SearchSpace
from black_it.utils.base import _assert
from black_it.utils.json_pandas_checkpointing import (
    load_calibrator_state,
    save_calibrator_state,
)
from black_it.utils.seedable import BaseSeedable

if TYPE_CHECKING:
    import os

    from numpy.typing import NDArray

    from black_it.loss_functions.base import BaseLoss
    from black_it.samplers.base import BaseSampler


class Calibrator(BaseSeedable):
    """The class used to perform a calibration."""

    STATE_VERSION = 0

    def __init__(  # noqa: PLR0913
        self,
        loss_function: BaseLoss,
        real_data: NDArray[np.float64],
        model: Callable,
        parameters_bounds: NDArray[np.float64] | list[list[float]],
        parameters_precision: NDArray[np.float64] | list[float],
        ensemble_size: int,
        samplers: Sequence[BaseSampler] | None = None,
        scheduler: BaseScheduler | None = None,
        sim_length: int | None = None,
        convergence_precision: int | None = None,
        verbose: bool = True,
        saving_folder: str | None = None,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        """Initialize the Calibrator object.

        It must be initialized with details on the parameters to explore,
        on the model to calibrate, on the samplers and on the loss function to use.

        Args:
            loss_function: a loss function which evaluates the similarity between simulated and real datasets
            real_data: an array containing the real time series
            model: a model with free parameters to be calibrated
            parameters_bounds: the bounds of the parameter space
            parameters_precision: the precisions to be used for the discretization of the parameters
            ensemble_size: number of repetitions to be run for each set of parameters to decrease statistical
                fluctuations. For deterministic models this should be set to 1.
            samplers: list of methods to be used in the calibration procedure
            scheduler: the scheduler to be used in order to schedule the available samplers
            sim_length: number of periods to simulate the model for, by default this is equal to the length of the
                real time series.
            convergence_precision: number of significant digits to consider in the convergence check. The check is
                not performed if this is set to 'None'.
            verbose: whether to print calibration updates
            saving_folder: the name of the folder where data should be saved and/or retrieved
            random_state: random state of the calibrator, used for model simulations and to initialise the samplers
            n_jobs: the maximum number of concurrently running jobs. For more details, see the
                [joblib.Parallel documentation](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html).

        """
        BaseSeedable.__init__(self, random_state=random_state)
        self.loss_function = loss_function
        self.model = model
        self.random_state = random_state
        self.real_data = real_data
        self.ensemble_size = ensemble_size

        if sim_length is None:
            self.N = self.real_data.shape[0]
        else:
            if sim_length != self.real_data.shape[0]:
                warnings.warn(  # noqa: B028
                    "The length of real time series is different from the simulation length, "
                    f"got {self.real_data.shape[0]} and {sim_length}. This may or may not be a problem depending "
                    "on the loss function used.",
                    RuntimeWarning,
                )
            self.N = sim_length
        self.D = self.real_data.shape[1]
        self.verbose = verbose
        self.convergence_precision = (
            self._validate_convergence_precision(convergence_precision)
            if convergence_precision is not None
            else None
        )
        self.saving_folder = saving_folder

        # Initialize search grid
        self.param_grid = SearchSpace(parameters_bounds, parameters_precision, verbose)

        # initialize arrays
        self.params_samp = np.zeros((0, self.param_grid.dims))
        self.losses_samp = np.zeros(0)
        self.batch_num_samp = np.zeros(0, dtype=int)
        self.method_samp = np.zeros(0, dtype=int)
        self.series_samp = np.zeros((0, self.ensemble_size, self.N, self.D))

        # initialize variables before calibration
        self.n_sampled_params = 0
        self.current_batch_index = 0

        # set number of processes for parallel evaluation of model
        self.n_jobs = n_jobs if n_jobs is not None else multiprocessing.cpu_count()

        print(
            f"Selecting {self.n_jobs} processes for the parallel evaluation of the model",
        )

        self.scheduler = self.__validate_samplers_and_scheduler_constructor_args(
            samplers,
            scheduler,
        )

        self.samplers_id_table = self._construct_samplers_id_table(
            list(self.scheduler.samplers),
        )

    @classmethod
    def __validate_samplers_and_scheduler_constructor_args(
        cls,
        samplers: Sequence[BaseSampler] | None,
        scheduler: BaseScheduler | None,
    ) -> BaseScheduler:
        """Validate the 'samplers' and the 'scheduler' arguments provided to the constructor."""
        both_none = samplers is None and scheduler is None
        both_not_none = samplers is not None and scheduler is not None
        if both_none and both_not_none:
            msg = "only one between 'samplers' and 'scheduler' must be provided"
            raise ValueError(
                msg,
            )

        if samplers is not None:
            return RoundRobinScheduler(samplers)

        return cast(BaseScheduler, scheduler)

    def _set_samplers_seeds(self) -> None:
        """Set the calibration seed."""
        self.scheduler.random_state = self.random_state

        # "burn" seeds from the calibrator seed generator for backward compatibility
        for _ in self.scheduler.samplers:
            self._get_random_seed()

    @staticmethod
    def _construct_samplers_id_table(samplers: list[BaseSampler]) -> dict[str, int]:
        """Construct the samplers-by-id table.

        Given the list (built-in or user-defined) of samplers a calibration
        session is going to use, return a map from the sampler human-readable
        name to a numeric id (starting from 0).

        Different calibration sessions may result in different conversion
        tables.

        Args:
            samplers: the list of samplers of the calibrator

        Returns:
            A dict that maps from the given sampler names to unique ids.
        """
        samplers_id_table = {}
        sampler_id = 0

        for sampler in samplers:
            sampler_name = type(sampler).__name__
            if sampler_name in samplers_id_table:
                continue

            samplers_id_table[sampler_name] = sampler_id
            sampler_id = sampler_id + 1

        return samplers_id_table

    def set_samplers(self, samplers: list[BaseSampler]) -> None:
        """Set the samplers list of the calibrator.

        This method overwrites the samplers of the calibrator object with a custom list of samplers.

        Args:
            samplers: a list of samplers

        """
        # overwrite the list of samplers
        self.scheduler._samplers = tuple(samplers)  # noqa: SLF001
        self.update_samplers_id_table(samplers)

    def set_scheduler(self, scheduler: BaseScheduler) -> None:
        """Overwrite the scheduler of the calibrator object."""
        self.scheduler = scheduler
        self.update_samplers_id_table(self.scheduler.samplers)

    def update_samplers_id_table(self, samplers: Sequence[BaseSampler]) -> None:
        """Update the samplers_id_table attribute with possible new samplers."""
        # update the samplers_id_table with the new samplers, only if necessary
        sampler_id = max(self.samplers_id_table.values()) + 1

        for sampler in samplers:
            sampler_name = type(sampler).__name__
            if sampler_name in self.samplers_id_table:
                continue

            self.samplers_id_table[sampler_name] = sampler_id
            sampler_id = sampler_id + 1

    @classmethod
    def restore_from_checkpoint(
        cls,
        checkpoint_path: str,
        model: Callable,
    ) -> Calibrator:
        """Return an instantiated class from a database file and a model simulator.

        Args:
            checkpoint_path: the name of the database file to read from
            model: the model to calibrate. It must be equal to the one already calibrated

        Returns:
            An initialised Calibrator object.
        """
        (
            parameters_bounds,
            parameters_precision,
            real_data,
            ensemble_size,
            sim_length,
            _d,
            convergence_precision,
            verbose,
            saving_file,
            random_state,
            random_generator_state,
            model_name,
            scheduler,
            loss_function,
            current_batch_index,
            n_sampled_params,
            n_jobs,
            params_samp,
            losses_samp,
            series_samp,
            batch_num_samp,
            method_samp,
        ) = load_calibrator_state(checkpoint_path, cls.STATE_VERSION)

        _assert(
            model_name == model.__name__,
            (
                "Error: the model provided appears to be different from the one present "
                "in the database"
            ),
        )

        calibrator = cls(
            loss_function,
            real_data,
            model,
            parameters_bounds,
            parameters_precision,
            ensemble_size,
            scheduler=scheduler,
            sim_length=sim_length,
            convergence_precision=convergence_precision,
            verbose=verbose,
            saving_folder=saving_file,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        calibrator.current_batch_index = current_batch_index
        calibrator.n_sampled_params = n_sampled_params
        calibrator.params_samp = params_samp
        calibrator.losses_samp = losses_samp
        calibrator.series_samp = series_samp
        calibrator.batch_num_samp = batch_num_samp
        calibrator.method_samp = method_samp

        # reset the random number generator state
        calibrator.random_generator.bit_generator.state = random_generator_state

        return calibrator

    def simulate_model(self, params: NDArray) -> NDArray:
        """Simulate the model.

        This method calls the model simulator in parallel on a given set of parameter values, a number of repeated
        evaluations are performed for each parameter to average out random fluctuations.

        Args:
            params: the array of parameters for which the model should be evaluated
        # noqa
        Returns:
            simulated_data: an array of dimensions (batch_size, ensemble_size, N, D) containing all
                simulated time series
        """
        rep_params = np.repeat(params, self.ensemble_size, axis=0)

        simulated_data_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self.model)(param, self.N, self._get_random_seed())
            for i, param in enumerate(rep_params)
        )

        simulated_data = np.array(simulated_data_list)

        return np.reshape(
            simulated_data,
            (params.shape[0], self.ensemble_size, self.N, self.D),
        )

    def calibrate(self, n_batches: int) -> tuple[NDArray, NDArray]:
        """Run calibration for n batches.

        Args:
            n_batches (int): number of 'batches' to be executed. Each batch runs over all methods

        Returns:
            The sampled parameters and the corresponding sampled losses.
            Both arrays are sorted by increasing loss values
        """
        if self.current_batch_index == 0:
            # we only set the samplers' random state at the start of a calibration
            self._set_samplers_seeds()

        with self.scheduler.session():
            for _ in range(n_batches):
                print()
                print(f"BATCH NUMBER:   {self.current_batch_index + 1}")
                print(f"PARAMS SAMPLED: {self.n_sampled_params}")

                method = self.scheduler.get_next_sampler()

                t_start = time.time()
                print()
                print(f"METHOD: {type(method).__name__}")

                # get new params from a specific sampler
                new_params = method.sample(
                    self.param_grid,
                    self.params_samp,
                    self.losses_samp,
                )

                t_eval = time.time()

                # simulate an ensemble of models for different parameters

                new_simulated_data = self.simulate_model(new_params)

                new_losses = []

                for sim_data_ensemble in new_simulated_data:
                    new_loss = self.loss_function.compute_loss(
                        sim_data_ensemble,
                        self.real_data,
                    )
                    new_losses.append(new_loss)

                # update arrays
                self.params_samp = np.vstack((self.params_samp, new_params))
                self.losses_samp = np.hstack((self.losses_samp, new_losses))
                self.series_samp = np.vstack((self.series_samp, new_simulated_data))
                self.batch_num_samp = np.hstack(
                    (
                        self.batch_num_samp,
                        [self.current_batch_index] * method.batch_size,
                    ),
                )
                self.method_samp = np.hstack(
                    (
                        self.method_samp,
                        [self.samplers_id_table[type(method).__name__]]
                        * method.batch_size,
                    ),
                )

                # logging
                t_end = time.time()
                if self.verbose:
                    min_dist_new_points = np.round(np.min(new_losses), 2)
                    avg_dist_new_points = np.round(np.average(new_losses), 2)
                    avg_dist_existing_points = np.round(np.average(self.losses_samp), 2)

                    elapsed_tot = np.round(t_end - t_start, 1)
                    elapsed_eval = np.round(t_end - t_eval, 1)
                    print(
                        textwrap.dedent(
                            f"""\
                        ----> sim exec elapsed time: {elapsed_eval}s
                        ---->   min loss new params: {min_dist_new_points}
                        ---->   avg loss new params: {avg_dist_new_points}
                        ----> avg loss exist params: {avg_dist_existing_points}
                        ---->         curr min loss: {np.min(self.losses_samp)}
                        ====>    total elapsed time: {elapsed_tot}s
                        """,
                        ),
                        end="",
                    )

                # update count of number of params sampled
                self.n_sampled_params = self.n_sampled_params + len(new_params)

                self.scheduler.update(
                    self.current_batch_index,
                    new_params,
                    new_losses,  # type: ignore[arg-type]
                    new_simulated_data,
                )

                self.current_batch_index += 1

                # check convergence for early termination
                if self.convergence_precision is not None:
                    converged = self.check_convergence(
                        self.losses_samp,
                        self.n_sampled_params,
                        self.convergence_precision,
                    )
                    if converged and self.verbose:
                        print("\nCONVERGENCE CHECK:")
                        print("Achieved convergence loss, stopping search.")
                        break

                if self.saving_folder is not None:
                    self.create_checkpoint(self.saving_folder)

            idx = np.argsort(self.losses_samp)

        return self.params_samp[idx], self.losses_samp[idx]

    @staticmethod
    def check_convergence(
        losses_samp: NDArray,
        n_sampled_params: int,
        convergence_precision: int,
    ) -> bool:
        """Check convergence of the calibration.

        Args:
            losses_samp: the sampled losses
            n_sampled_params: the number of sampled params
            convergence_precision: the required convergence precision.

        Returns:
            True if the calibration converged, False otherwise.
        """
        return (
            np.round(np.min(losses_samp[:n_sampled_params]), convergence_precision) == 0
        )

    def create_checkpoint(self, file_name: str | os.PathLike) -> None:
        """Save the current state of the object.

        Args:
            file_name: the name of the folder where the data will be saved
        """
        checkpoint_path: Path = Path(file_name).resolve()

        t_start = time.time()

        model_name = self.model.__name__

        save_calibrator_state(
            checkpoint_path,
            self.param_grid.parameters_bounds,
            self.param_grid.parameters_precision,
            self.real_data,
            self.ensemble_size,
            self.N,
            self.D,
            self.convergence_precision,
            self.verbose,
            self.saving_folder,
            self.random_state,
            self.random_generator.bit_generator.state,
            model_name,
            self.scheduler,
            self.loss_function,
            self.current_batch_index,
            self.n_sampled_params,
            self.n_jobs,
            self.params_samp,
            self.losses_samp,
            self.series_samp,
            self.batch_num_samp,
            self.method_samp,
        )

        t_end = time.time()

        elapsed = np.round(t_end - t_start, 1)
        print(f"Checkpoint saved in {elapsed}s")

    @staticmethod
    def _validate_convergence_precision(convergence_precision: int) -> int:
        """Validate convergence precision input."""
        _assert(
            convergence_precision >= 0,
            f"convergence precision must be an integer greater than 0, got {convergence_precision}",
            exception_class=ValueError,
        )
        return convergence_precision
