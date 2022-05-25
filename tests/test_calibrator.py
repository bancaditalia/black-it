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
        self.bounds = np.array([[0.01, 0.01], [1.00, 1.00]])
        self.bounds_step = np.array([0.01, 0.01])

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

        # define a loss
        self.loss = MethodOfMomentsLoss()

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_calibrator_calibrate(self, n_jobs: int) -> None:
        """Test the Calibrator.calibrate method, positive case, with different number of jobs."""
        expected_params = np.array(
            [
                [0.7, 0.61],
                [0.86, 0.75],
                [0.86, 0.64],
                [0.82, 0.76],
                [0.91, 0.68],
                [0.02, 0.64],
                [0.88, 0.65],
                [0.75, 0.12],
                [0.51, 0.34],
                [0.52, 0.78],
                [0.48, 0.52],
                [0.84, 0.27],
                [0.77, 0.22],
                [0.82, 0.09],
                [0.13, 0.45],
                [0.99, 0.99],
                [0.02, 0.02],
                [0.26, 0.67],
                [1.0, 0.98],
                [0.26, 0.08],
                [0.72, 0.96],
                [0.01, 0.02],
                [0.9, 0.76],
                [0.86, 0.7],
            ]
        )

        expected_losses = [
            0.3972552,
            0.42686962,
            0.43481878,
            0.73160366,
            0.73603323,
            0.8516071,
            0.90245126,
            0.97365625,
            1.06424397,
            1.11547266,
            1.17847713,
            1.2153038,
            1.25877588,
            1.26424663,
            1.49182603,
            1.60069072,
            1.60835184,
            1.69164426,
            1.71154007,
            2.00411593,
            2.28922901,
            2.36455155,
            2.61220382,
            3.60022745,
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
    bounds = np.array([[0.01, 0.01], [1.00, 1.00]])
    bounds_step = np.array([0.01, 0.01])

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

        def single_sample(
            self,
            seed: int,
            search_space: SearchSpace,
            existing_points: NDArray[np.float64],
            existing_losses: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            """Do a single sample."""

    cal = Calibrator(
        samplers=[MyCustomSampler(MagicMock(), MagicMock(), MagicMock())],
        real_data=MagicMock(),
        model=MagicMock(),
        parameters_bounds=np.array([MagicMock(), MagicMock()]),
        parameters_precision=MagicMock(),
        ensemble_size=2,
        loss_function=MagicMock(),
        saving_folder=None,
        n_jobs=1,
    )

    assert len(cal.samplers_id_table) == 1
    assert cal.samplers_id_table[MyCustomSampler.__name__] == 0
