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

"""This module contains tests for the Calibrator.calibrate method."""

import glob
import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from black_it.calibrator import Calibrator
from black_it.loss_functions.msm import MethodOfMomentsLoss
from black_it.samplers.base import BaseSampler
from black_it.samplers.best_batch import BestBatchSampler
from black_it.samplers.gaussian_process import GaussianProcessSampler
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.r_sequence import RSequenceSampler
from black_it.samplers.random_forest import RandomForestSampler
from black_it.samplers.random_uniform import RandomUniformSampler
from black_it.search_space import SearchSpace

from .fixtures.test_models import NormalMV  # type: ignore


class TestCalibrate:  # pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
    """Test the Calibrator.calibrate method."""

    def setup(self) -> None:
        """Set up the tests."""
        self.true_params = np.array([0.50, 0.50])
        self.bounds = [
            [0.01, 0.01],
            [1.00, 1.00],
        ]
        self.bounds_step = [0.01, 0.01]

        self.batch_size = 2
        self.random_sampler = RandomUniformSampler(batch_size=self.batch_size)
        self.halton_sampler = HaltonSampler(batch_size=self.batch_size)
        self.bb_sampler = BestBatchSampler(batch_size=self.batch_size)
        self.gauss_sampler = GaussianProcessSampler(batch_size=self.batch_size)
        self.rseq_sampler = RSequenceSampler(batch_size=self.batch_size)
        self.forest_sampler = RandomForestSampler(batch_size=self.batch_size)

        # model to be calibrated
        self.model = NormalMV

        # generate a synthetic dataset to test the calibrator
        self.real_data = self.model(self.true_params, N=100, seed=0)

        # set calibrator initial random seed
        self.random_state = 0

        # define a loss
        self.loss = MethodOfMomentsLoss()

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_calibrator_calibrate(self, n_jobs: int) -> None:
        """Test the Calibrator.calibrate method, positive case, with different number of jobs."""
        expected_params = np.array(
            [
                [0.53, 0.75],
                [0.68, 0.72],
                [0.18, 0.39],
                [0.53, 0.74],
                [0.8, 0.06],
                [0.56, 0.79],
                [0.56, 0.37],
                [0.82, 0.08],
                [0.08, 0.37],
                [0.99, 0.56],
                [0.99, 0.06],
                [0.07, 0.5],
                [1.0, 1.0],
                [0.16, 0.13],
                [0.68, 0.69],
                [0.71, 0.43],
                [0.65, 0.72],
                [1.0, 0.07],
                [0.06, 0.82],
                [0.98, 0.98],
                [0.03, 0.83],
                [0.32, 0.93],
                [0.43, 0.17],
                [0.46, 0.32],
            ]
        )

        expected_losses = [
            0.3379579210350664,
            0.6142406842968648,
            0.7511535247731311,
            0.7947552011919158,
            0.8438595077543548,
            0.9962563670251199,
            1.0871632317532742,
            1.1358155438012694,
            1.1931075019681772,
            1.2153376231157316,
            1.2609924409732285,
            1.5690132814746562,
            1.6982927628783613,
            1.7291748954177788,
            1.7313136729102845,
            1.8208003946536748,
            2.0235794497211184,
            2.169554479206179,
            2.3750688536096995,
            2.452161750981678,
            2.5383914038222293,
            2.555848298760814,
            3.258788971813837,
            3.500997423345573,
        ]

        cal = Calibrator(
            samplers=[
                self.random_sampler,
                self.halton_sampler,
                self.rseq_sampler,
                self.forest_sampler,
                self.bb_sampler,
                self.gauss_sampler,
            ],
            real_data=self.real_data,
            model=self.model,
            parameters_bounds=self.bounds,
            parameters_precision=self.bounds_step,
            ensemble_size=3,
            loss_function=self.loss,
            saving_folder=None,
            random_state=self.random_state,
            n_jobs=n_jobs,
        )

        params, losses = cal.calibrate(2)

        assert np.allclose(params, expected_params)
        assert np.allclose(losses, expected_losses)

    def test_calibrator_with_check_convergence(self, capsys: Any) -> None:
        """Test the Calibrator.calibrate method with convergence check."""
        cal = Calibrator(
            samplers=[
                self.random_sampler,
                self.halton_sampler,
                self.rseq_sampler,
                self.forest_sampler,
                self.bb_sampler,
                self.gauss_sampler,
            ],
            real_data=self.real_data,
            model=self.model,
            parameters_bounds=self.bounds,
            parameters_precision=self.bounds_step,
            ensemble_size=3,
            loss_function=self.loss,
            saving_folder=None,
            convergence_precision=4,
            n_jobs=None,
            verbose=True,
        )

        with patch.object(cal, "check_convergence", return_value=[False, True]):
            cal.calibrate(2)

        captured_output = capsys.readouterr()
        assert "Achieved convergence loss, stopping search." in captured_output.out


def test_calibrator_restore_from_checkpoint_and_set_sampler() -> None:
    """Test 'Calibrator.restore_from_checkpoint', positive case, and 'Calibrator.set_sampler'."""
    true_params = np.array([0.50, 0.50])
    bounds = [
        [0.01, 0.01],
        [1.00, 1.00],
    ]
    bounds_step = [0.01, 0.01]

    batch_size = 2
    random_sampler = RandomUniformSampler(batch_size=batch_size)
    halton_sampler = HaltonSampler(batch_size=batch_size)

    model = NormalMV
    real_data = model(true_params, N=100, seed=0)
    loss = MethodOfMomentsLoss()

    # initialize a Calibrator object
    cal = Calibrator(
        samplers=[
            random_sampler,
            halton_sampler,
        ],
        real_data=real_data,
        model=model,
        parameters_bounds=bounds,
        parameters_precision=bounds_step,
        ensemble_size=2,
        loss_function=loss,
        saving_folder="saving_folder",
        n_jobs=1,
    )

    _, _ = cal.calibrate(2)

    cal_restored = Calibrator.restore_from_checkpoint("saving_folder", model=model)

    # loop over all attributes of the classes
    vars_cal = vars(cal)
    for key in vars_cal:
        # if the attribute is an object just check the equality of their names
        if key == "samplers":
            for method1, method2 in zip(vars_cal["samplers"], cal_restored.samplers):
                assert type(method1).__name__ == type(method2).__name__
        elif key == "loss_function":
            assert (
                type(vars_cal["loss_function"]).__name__
                == type(cal_restored.loss_function).__name__  # noqa
            )
        elif key == "param_grid":
            assert (
                type(vars_cal["param_grid"]).__name__
                == type(cal_restored.param_grid).__name__  # noqa
            )
        elif key == "_random_generator":
            assert (
                vars_cal[key].bit_generator.state
                == cal_restored.random_generator.bit_generator.state
            )
        # otherwise check the equality of numerical values
        else:
            assert vars_cal[key] == pytest.approx(getattr(cal_restored, key))

    # testt the setting of a new sampler to the calibrator object
    best_batch_sampler = BestBatchSampler(batch_size=2)
    cal.set_samplers(
        [random_sampler, best_batch_sampler]
    )  # note: only the second sampler is new
    assert len(cal.samplers) == 2
    assert type(cal.samplers[1]).__name__ == "BestBatchSampler"
    assert len(cal.samplers_id_table) == 3
    assert cal.samplers_id_table["BestBatchSampler"] == 2

    # remove the test folder
    files = glob.glob("saving_folder/*")
    for f in files:
        os.remove(f)
    os.rmdir("saving_folder")


def test_new_sampling_method() -> None:
    """Test Calibrator instantiation using a new sampling method."""

    class MyCustomSampler(BaseSampler):
        """Custom sampler."""

        def sample_batch(
            self,
            batch_size: int,
            search_space: SearchSpace,
            existing_points: NDArray[np.float64],
            existing_losses: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            """Sample a batch of parameters."""

    cal = Calibrator(
        samplers=[MyCustomSampler(batch_size=2)],
        real_data=MagicMock(),
        model=MagicMock(),
        parameters_bounds=[
            MagicMock(),
            MagicMock(),
        ],
        parameters_precision=MagicMock(),
        ensemble_size=2,
        loss_function=MagicMock(),
        saving_folder=None,
        n_jobs=1,
    )

    assert len(cal.samplers_id_table) == 1
    assert cal.samplers_id_table[MyCustomSampler.__name__] == 0
