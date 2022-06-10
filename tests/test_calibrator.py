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
import shutil
import tempfile
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

        # set calibrator random state
        self.random_state = 0

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_calibrator_calibrate(self, n_jobs: int) -> None:
        """Test the Calibrator.calibrate method, positive case, with different number of jobs."""
        expected_params = np.array(
            [
                [0.11, 0.36],
                [0.58, 0.64],
                [0.68, 0.72],
                [0.58, 0.59],
                [0.18, 0.39],
                [0.53, 0.74],
                [0.11, 0.31],
                [0.8, 0.06],
                [0.2, 0.22],
                [0.19, 0.37],
                [0.41, 0.49],
                [0.16, 0.13],
                [0.32, 0.93],
                [0.65, 0.36],
                [0.82, 0.08],
                [0.03, 0.01],
                [1.0, 0.98],
                [0.06, 0.41],
                [0.07, 0.5],
                [0.03, 0.83],
                [0.02, 0.03],
                [0.98, 1.0],
                [0.43, 0.17],
                [0.46, 0.32],
            ]
        )

        expected_losses = [
            0.40391037,
            0.55369057,
            0.61424068,
            0.74831741,
            0.75115352,
            0.7947552,
            0.81273763,
            0.84385951,
            1.01089835,
            1.61112201,
            1.61341081,
            1.7291749,
            1.76300457,
            1.85015661,
            1.88723244,
            1.89303475,
            1.94197351,
            1.94259241,
            2.19809781,
            2.5383914,
            2.67175361,
            2.81332498,
            3.25878897,
            3.50099742,
        ]

        cal = Calibrator(
            samplers=[
                self.random_sampler,
                self.halton_sampler,
                self.rseq_sampler,
                self.gauss_sampler,
                self.forest_sampler,
                self.bb_sampler,
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

        print(params)
        print(losses)
        assert np.allclose(params, expected_params)
        assert np.allclose(losses, expected_losses)

    def test_calibrator_with_check_convergence(self, capsys: Any) -> None:
        """Test the Calibrator.calibrate method with convergence check."""
        cal = Calibrator(
            samplers=[
                self.random_sampler,
                self.halton_sampler,
                self.rseq_sampler,
                self.gauss_sampler,
                self.forest_sampler,
                self.bb_sampler,
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


class test_calibrator_restore_from_checkpoint_and_set_sampler:
    """Test 'Calibrator.restore_from_checkpoint', positive case, and 'Calibrator.set_sampler'."""

    saving_folder: str

    @classmethod
    def setup_class(cls) -> None:
        """Set up the test class."""
        cls.saving_folder = tempfile.mktemp()

    def test_run(self) -> None:  # pylint: disable=too-many-locals
        """Run the test."""
        true_params = np.array([0.50, 0.50])
        bounds = np.array([[0.0, 0.0], [1.00, 1.00]])
        bounds_step = np.array([0.01, 0.01])

        batch_size = 2
        random_sampler = RandomUniformSampler(batch_size=batch_size)
        halton_sampler = HaltonSampler(batch_size=batch_size)
        gaussian_sampler = GaussianProcessSampler(batch_size=batch_size, max_iters=10)
        best_batch_sampler = BestBatchSampler(batch_size=batch_size)
        r_sequence_sampler = RSequenceSampler(batch_size=batch_size)
        random_forest_sampler = RandomForestSampler(
            batch_size=batch_size, n_estimators=10
        )

        model = NormalMV
        real_data = model(true_params, N=100, seed=0)
        loss = MethodOfMomentsLoss()

        # initialize a Calibrator object
        cal = Calibrator(
            samplers=[
                random_sampler,
                halton_sampler,
                gaussian_sampler,
                best_batch_sampler,
                r_sequence_sampler,
                random_forest_sampler,
            ],
            real_data=real_data,
            model=model,
            parameters_bounds=bounds,
            parameters_precision=bounds_step,
            ensemble_size=2,
            loss_function=loss,
            saving_folder=self.saving_folder,
            n_jobs=1,
        )

        _, _ = cal.calibrate(2)

        cal_restored = Calibrator.restore_from_checkpoint(
            self.saving_folder, model=model
        )

        # loop over all attributes of the classes
        vars_cal = vars(cal)
        for key in vars_cal:
            # if the attribute is an object just check the equality of their names
            if key == "samplers":
                for method1, method2 in zip(
                    vars_cal["samplers"], cal_restored.samplers
                ):
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

    @classmethod
    def teardown_class(cls) -> None:
        """Teardown the class."""
        shutil.rmtree(cls.saving_folder)


def test_new_sampling_method() -> None:
    """Test Calibrator instantiation using a new sampling method."""

    class MyCustomSampler(BaseSampler):
        """Custom sampler."""

        def sample_batch(
            self,
            nb_samples: int,
            search_space: SearchSpace,
            existing_points: NDArray[np.float64],
            existing_losses: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            """Sample a batch of points."""

    cal = Calibrator(
        samplers=[MyCustomSampler(batch_size=2, max_deduplication_passes=0)],
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
