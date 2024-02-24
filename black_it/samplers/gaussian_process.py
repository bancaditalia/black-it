# Black-box ABM Calibration Kit (Black-it)
# Copyright (C) 2021-2024 Banca d'Italia
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

"""This module contains the implementation of the Gaussian process-based sampling."""

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.special import erfc  # type: ignore[import]  # pylint: disable=no-name-in-module
from sklearn.gaussian_process import GaussianProcessRegressor, kernels  # type: ignore[import]

from black_it.samplers.surrogate import MLSurrogateSampler

if TYPE_CHECKING:
    from numpy.typing import NDArray


_BIG_DATASET_SIZE_WARNING_THRESHOLD = 500
_SMALL_VARIANCE_VALUES = 1e-5


class _AcquisitionTypes(Enum):
    """Enumeration of allowed acquisition types."""

    MEAN = "mean"
    EI = "expected_improvement"

    def __str__(self) -> str:
        """Get the string representation."""
        return self.value


class GaussianProcessSampler(MLSurrogateSampler):
    """This class implements the Gaussian process-based sampler.

    In particular, the sampling is based on a Gaussian Process interpolation of the loss function.

    Note: this class is a wrapper of the GaussianProcessRegressor model of the scikit-learn package.
    """

    def __init__(  # noqa: PLR0913
        self,
        batch_size: int,
        random_state: int | None = None,
        max_deduplication_passes: int = 5,
        candidate_pool_size: int | None = None,
        optimize_restarts: int = 5,
        acquisition: str = "expected_improvement",
        jitter: float = 0.1,
    ) -> None:
        """Initialize the sampler.

        Args:
            batch_size: the number of points sampled every time the sampler is called
            random_state: the random state of the sampler, fixing this number the sampler behaves deterministically
            max_deduplication_passes: the maximum number of deduplication passes that are made
            candidate_pool_size: number of randomly sampled points on which the random forest predictions are evaluated
            optimize_restarts: number of independent random trials of the optimization of the GP hyperparameters
            acquisition: type of acquisition function, it can be 'expected_improvement' of simply 'mean'
            jitter: positive value to make the "expected_improvement" acquisition more explorative.
        """
        self._validate_acquisition(acquisition)

        super().__init__(
            batch_size,
            random_state,
            max_deduplication_passes,
            candidate_pool_size,
        )
        self.optimize_restarts = optimize_restarts
        self.acquisition = acquisition
        self.jitter = jitter
        self._gpmodel: GaussianProcessRegressor | None = None
        self._fmin: np.double | float | None = None

    @staticmethod
    def _validate_acquisition(acquisition: str) -> None:
        """Check that the required acquisition is among the supported ones.

        Args:
            acquisition: the acquisition provided as input of the constructor.

        Raises:
            ValueError: if the provided acquisition type is not among the allowed ones.
        """
        try:
            _AcquisitionTypes(acquisition)
        except ValueError as e:
            msg = (
                "expected one of the following acquisition types: "
                f"[{' '.join(map(str, _AcquisitionTypes))}], got {acquisition}"
            )
            raise ValueError(
                msg,
            ) from e

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:  # noqa: N803
        """Fit a gaussian process surrogate model."""
        y = np.atleast_2d(y).T

        if X.shape[0] > _BIG_DATASET_SIZE_WARNING_THRESHOLD:
            warnings.warn(  # noqa: B028
                "Standard GP evaluations can be expensive for large datasets, consider implementing a sparse GP",
                RuntimeWarning,
            )

        # initialize GP class from scikit-learn with a Matern kernel
        kernel = kernels.Matern(length_scale=1, length_scale_bounds=(1e-5, 1e5), nu=2.5)

        noise_var = y.var() * 0.01

        self._gpmodel = GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise_var,
            n_restarts_optimizer=self.optimize_restarts,
            optimizer="fmin_l_bfgs_b",
            random_state=self._get_random_seed(),
        )

        self._gpmodel.fit(X, y)

        # store minimum
        if self.acquisition == "expected_improvement":
            m, _ = self._predict_mean_std(X)
            self._fmin = np.min(m)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: N803
        """Predict using a gaussian process surrogate model."""
        # predict mean or expected improvement on the full sample set
        if self.acquisition == _AcquisitionTypes.EI.value:
            # minus sign needed for subsequent sorting
            candidates_score = -self._predict_EI(X, self.jitter)
        else:  # acquisition is "mean"
            candidates_score = self._predict_mean_std(X)[0]

        return candidates_score

    def _predict_mean_std(
        self,
        X: NDArray[np.float64],  # noqa: N803
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict mean and standard deviation of a fitted GP.

        Args:
            X: the points on which the predictions should be performed

        Returns:
            The pair (mean, std).
        """
        gpmodel = cast(GaussianProcessRegressor, self._gpmodel)
        X = X[None, :] if X.ndim == 1 else X  # noqa: N806
        m, s = gpmodel.predict(X, return_std=True, return_cov=False)
        s = np.clip(s, 1e-5, np.inf)
        return m, s

    def _predict_EI(  # noqa: N802
        self,
        X: NDArray[np.float64],  # noqa: N803
        jitter: float = 0.1,
    ) -> NDArray[np.float64]:
        """Compute the Expected Improvement per unit of cost.

        Args:
            X:  the points on which the predictions should be performed
            jitter: positive value to make the acquisition more explorative.

        Returns:
            the expected improvement.
        """
        m, s = self._predict_mean_std(X)

        fmin = cast(float, self._fmin)

        phi, Phi, u = self.get_quantiles(jitter, fmin, m, s)  # noqa: N806

        return s * (u * Phi + phi)

    @staticmethod
    def get_quantiles(
        acquisition_par: float,
        fmin: float,
        m: NDArray[np.float64],
        s: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Quantiles of the Gaussian distribution useful to determine the acquisition function values.

        Args:
            acquisition_par: parameter of the acquisition function
            fmin: current minimum.
            m: vector of means.
            s: vector of standard deviations.

        Returns:
            the quantiles.
        """
        # remove values of variance that are too small
        s[s < _SMALL_VARIANCE_VALUES] = _SMALL_VARIANCE_VALUES

        u: NDArray[np.float64] = (fmin - m - acquisition_par) / s
        phi: NDArray[np.float64] = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        Phi: NDArray[np.float64] = 0.5 * erfc(-u / np.sqrt(2))  # noqa: N806

        return phi, Phi, u
