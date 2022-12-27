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

"""This module contains the implementation of the Gaussian process-based sampling."""
import random
import warnings
from enum import Enum
from typing import Optional, Tuple, cast

import GPy
import numpy as np
from GPy.models import GPRegression
from numpy.typing import NDArray
from scipy.special import erfc  # pylint: disable=no-name-in-module

from black_it.samplers.surrogate import MLSurrogateSampler


class _AcquisitionTypes(Enum):
    """Enumeration of allowed acquisition types."""

    MEAN = "mean"
    EI = "expected_improvement"

    def __str__(self) -> str:
        """Get the string representation."""
        return self.value


class GaussianProcessSampler(MLSurrogateSampler):
    """
    This class implements the Gaussian process-based sampler.

    In particular, the sampling is based on a Gaussian Process interpolation of the loss function.

    Note: this class is a wrapper of the GPRegression model of the GPy package.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        batch_size: int,
        random_state: Optional[int] = None,
        max_deduplication_passes: int = 5,
        candidate_pool_size: Optional[int] = None,
        max_iters: int = 1000,
        optimize_restarts: int = 5,
        acquisition: str = "expected_improvement",
    ):
        """
        Initialize the sampler.

        Args:
            batch_size: the number of points sampled every time the sampler is called
            random_state: the random state of the sampler, fixing this number the sampler behaves deterministically
            max_deduplication_passes: the maximum number of deduplication passes that are made
            candidate_pool_size: number of randomly sampled points on which the random forest predictions are evaluated
            max_iters: maximum number of iteration in the optimization of the GP hyperparameters
            optimize_restarts: number of independent random trials of the optimization of the GP hyperparameters
            acquisition: type of acquisition function, it can be 'expected_improvement' of simply 'mean'
        """
        self._validate_acquisition(acquisition)

        super().__init__(
            batch_size, random_state, max_deduplication_passes, candidate_pool_size
        )
        self.max_iters = max_iters
        self.optimize_restarts = optimize_restarts
        self.acquisition = acquisition
        self._gpmodel: Optional[GPRegression] = None

    @staticmethod
    def _validate_acquisition(acquisition: str) -> None:
        """
        Check that the required acquisition is among the supported ones.

        Args:
            acquisition: the acquisition provided as input of the constructor.

        Raises
            ValueError: if the provided acquisition type is not among the allowed ones.
        """
        try:
            _AcquisitionTypes(acquisition)
        except ValueError as e:
            raise ValueError(
                "expected one of the following acquisition types: "
                f"[{' '.join(map(str, _AcquisitionTypes))}], "
                f"got {acquisition}"
            ) from e

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit a gaussian process surrogate model."""
        y = np.atleast_2d(y).T

        if X.shape[0] > 500:
            warnings.warn(
                "Standard GP evaluations can be expensive for large datasets, consider implementing a sparse GP",
                RuntimeWarning,
            )

        # initialize GP class from GPy with a Matern kernel by default
        dims = X.shape[1]
        kern = GPy.kern.Matern52(dims, variance=1.0, ARD=False)
        noise_var = y.var() * 0.01

        self._gpmodel = GPRegression(
            X, y, kernel=kern, noise_var=noise_var, mean_function=None
        )

        # Make sure we do not get ridiculously small residual noise variance
        self._gpmodel.Gaussian_noise.constrain_bounded(
            1e-9, 1e6, warning=False
        )  # constrain_positive(warning=False)

        # we need to set the seed globally for GPy optimisations
        # to give reproducible results
        np.random.seed(self._get_random_seed())
        random.seed(self._get_random_seed())
        if self.max_iters > 0:
            # --- update the model maximizing the marginal likelihood.
            if self.optimize_restarts == 1:
                self._gpmodel.optimize(
                    optimizer="bfgs",
                    max_iters=self.max_iters,
                    messages=False,
                    ipython_notebook=False,
                )
            else:
                self._gpmodel.optimize_restarts(
                    num_restarts=self.optimize_restarts,
                    optimizer="bfgs",
                    max_iters=self.max_iters,
                    verbose=False,
                )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict using a gaussian process surrogate model."""
        # predict mean or expected improvement on the full sample set
        if self.acquisition == _AcquisitionTypes.EI.value:
            # minus sign needed for subsequent sorting
            candidates_score = -self._predict_EI(X)[:, 0]
        else:  # acquisition is "mean"
            candidates_score = self._predict_mean_std(X)[0][:, 0]

        return candidates_score

    def _predict_mean_std(
        self, X: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Predict mean and standard deviation of a fitted GP.

        Args:
            X: the points on which the predictions should be performed

        Returns:
            The pair (mean, std).
        """
        gpmodel = cast(GPRegression, self._gpmodel)
        X = X[None, :] if X.ndim == 1 else X
        m, v = gpmodel.predict(X, full_cov=False, include_likelihood=True)
        v = np.clip(v, 1e-10, np.inf)
        return m, np.sqrt(v)

    def _get_fmin(self) -> float:
        """Return the location where the posterior mean is takes its minimal value."""
        gpmodel = cast(GPRegression, self._gpmodel)
        return gpmodel.predict(gpmodel.X)[0].min()

    def _predict_EI(
        self, X: NDArray[np.float64], jitter: float = 0.1
    ) -> NDArray[np.float64]:
        """
        Compute the Expected Improvement per unit of cost.

        Args:
            X:  the points on which the predictions should be performed
            jitter: positive value to make the acquisition more explorative.

        Returns:
            the expected improvement.
        """
        m, s = self._predict_mean_std(X)

        fmin = self._get_fmin()

        phi, Phi, u = self.get_quantiles(jitter, fmin, m, s)

        f_acqu = s * (u * Phi + phi)

        return f_acqu

    @staticmethod
    def get_quantiles(
        acquisition_par: float,
        fmin: float,
        m: NDArray[np.float64],
        s: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Quantiles of the Gaussian distribution useful to determine the acquisition function values.

        Args:
            acquisition_par: parameter of the acquisition function
            fmin: current minimum.
            m: vector of means.
            s: vector of standard deviations.

        Returns:
            the quantiles.
        """
        # remove values of variance that are too small
        s[s < 1e-10] = 1e-10

        u: NDArray[np.float64] = (fmin - m - acquisition_par) / s
        phi: NDArray[np.float64] = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        Phi: NDArray[np.float64] = 0.5 * erfc(-u / np.sqrt(2))

        return phi, Phi, u
