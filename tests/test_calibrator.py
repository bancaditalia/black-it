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

"""This module contains tests for the Calibrator.calibrate method."""
import sys
from pathlib import Path
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
from black_it.samplers.xgboost import XGBoostSampler
from black_it.search_space import SearchSpace
from black_it.utils.seedable import BaseSeedable

from .fixtures.test_models import NormalMV  # type: ignore


class TestCalibrate:  # pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
    """Test the Calibrator.calibrate method."""

    expected_params = np.array(
        [
            [0.59, 0.36],
            [0.63, 0.41],
            [0.18, 0.39],
            [0.56, 0.37],
            [0.83, 0.35],
            [0.54, 0.32],
            [0.74, 0.32],
            [0.53, 0.46],
            [0.57, 0.39],
            [0.94, 0.42],
            [0.32, 0.93],
            [0.8, 0.06],
            [0.01, 0.02],
            [0.04, 0.99],
        ]
    )

    expected_losses = [
        0.33400294,
        0.55274918,
        0.55798021,
        0.61712034,
        0.91962075,
        1.31118518,
        1.51682355,
        1.55503666,
        1.65968375,
        1.78845827,
        1.79905545,
        2.07605975,
        2.28484134,
        3.01432484,
    ]

    win32_expected_params = np.array(
        [
            [0.59, 0.36],
            [0.63, 0.41],
            [0.18, 0.39],
            [0.56, 0.37],
            [0.83, 0.35],
            [0.54, 0.32],
            [0.74, 0.32],
            [0.53, 0.46],
            [0.57, 0.39],
            [0.32, 0.93],
            [0.8, 0.06],
            [0.01, 0.02],
            [1.0, 0.99],
            [0.04, 0.99],
        ]
    )

    win32_expected_losses = [
        0.33400294,
        0.55274918,
        0.55798021,
        0.61712034,
        0.91962075,
        1.31118518,
        1.51682355,
        1.55503666,
        1.65968375,
        1.79905545,
        2.07605975,
        2.28484134,
        2.60093616,
        3.01432484,
    ]

    darwin_expected_params = np.array(
        [
            [0.59, 0.36],
            [0.63, 0.41],
            [0.18, 0.39],
            [0.56, 0.37],
            [0.83, 0.35],
            [0.54, 0.32],
            [0.74, 0.32],
            [0.53, 0.46],
            [0.57, 0.39],
            [0.92, 0.39],
            [0.32, 0.93],
            [0.8, 0.06],
            [0.01, 0.02],
            [0.04, 0.99],
        ]
    )

    darwin_expected_losses = [
        0.33400294,
        0.55274918,
        0.55798021,
        0.61712034,
        0.91962075,
        1.31118518,
        1.51682355,
        1.55503666,
        1.65968375,
        1.78808894,
        1.79905545,
        2.07605975,
        2.28484134,
        3.01432484,
    ]

    def setup(self) -> None:
        """Set up the tests."""
        self.true_params = np.array([0.50, 0.50])
        self.bounds = [
            [0.01, 0.01],
            [1.00, 1.00],
        ]
        self.bounds_step = [0.01, 0.01]

        self.batch_size = 1
        self.random_sampler = RandomUniformSampler(batch_size=self.batch_size)
        self.halton_sampler = HaltonSampler(batch_size=self.batch_size)
        self.bb_sampler = BestBatchSampler(batch_size=self.batch_size)
        self.gauss_sampler = GaussianProcessSampler(batch_size=self.batch_size)
        self.rseq_sampler = RSequenceSampler(batch_size=self.batch_size)
        self.forest_sampler = RandomForestSampler(batch_size=self.batch_size)
        self.xgboost_sampler = XGBoostSampler(batch_size=self.batch_size)

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
        cal = Calibrator(
            samplers=[
                self.random_sampler,
                self.halton_sampler,
                self.rseq_sampler,
                self.forest_sampler,
                self.bb_sampler,
                self.gauss_sampler,
                self.xgboost_sampler,
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

        params, losses = cal.calibrate(14)

        # TODO: this is a temporary workaround to make tests to run also on Windows.  # pylint: disable=fixme
        #       See: https://github.com/bancaditalia/black-it/issues/49
        if sys.platform == "win32":
            assert np.allclose(params, self.win32_expected_params)
            assert np.allclose(losses, self.win32_expected_losses)
        elif sys.platform == "darwin":
            assert np.allclose(params, self.darwin_expected_params)
            assert np.allclose(losses, self.darwin_expected_losses)
        else:
            assert np.allclose(params, self.expected_params)
            assert np.allclose(losses, self.expected_losses)

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
            cal.calibrate(12)

        captured_output = capsys.readouterr()
        assert "Achieved convergence loss, stopping search." in captured_output.out


def test_calibrator_restore_from_checkpoint_and_set_sampler(tmp_path: Path) -> None:
    """Test 'Calibrator.restore_from_checkpoint', positive case, and 'Calibrator.set_sampler'."""
    saving_folder_path_str = str(tmp_path / "saving_folder")
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
        saving_folder=saving_folder_path_str,
        random_state=0,
        n_jobs=1,
    )

    _, _ = cal.calibrate(4)

    cal_restored = Calibrator.restore_from_checkpoint(
        saving_folder_path_str, model=model
    )

    # loop over all attributes of the classes
    vars_cal = vars(cal)
    for key in vars_cal:
        # if the attribute is an object just check the equality of their names
        if key == "samplers":
            for method1, method2 in zip(
                vars_cal["samplers"], cal_restored.scheduler.samplers
            ):
                assert type(method1).__name__ == type(method2).__name__  # noqa
        if key == "scheduler":
            t1 = type(vars_cal["scheduler"])
            t2 = type(cal_restored.scheduler)
            assert t1 == t2  # noqa, pylint: disable=unidiomatic-typecheck
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
        elif key == f"_{BaseSeedable.__name__}__random_generator":
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
    assert len(cal.scheduler.samplers) == 2
    assert type(cal.scheduler.samplers[1]).__name__ == "BestBatchSampler"
    assert len(cal.samplers_id_table) == 3
    assert cal.samplers_id_table["BestBatchSampler"] == 2


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
            return []  # type: ignore[return-value]

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
