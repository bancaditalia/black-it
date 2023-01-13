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

"""This module contains the implementation of the XGBoost sampling."""
import warnings
from typing import Optional, cast

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from black_it.samplers.surrogate import MLSurrogateSampler

MAX_FLOAT32 = np.finfo(np.float32).max
MIN_FLOAT32 = np.finfo(np.float32).min
EPS_FLOAT32 = np.finfo(np.float32).eps


class XGBoostSampler(MLSurrogateSampler):
    """This class implements xgboost sampling."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        batch_size: int,
        random_state: Optional[int] = None,
        max_deduplication_passes: int = 5,
        candidate_pool_size: Optional[int] = None,
        colsample_bytree: float = 0.3,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        alpha: float = 1.0,
        n_estimators: int = 10,
    ) -> None:
        """
        Sampler based on a xgboost surrogate model of the loss function.

        Note: this class makes use of the xgboost library, for more information on the XGBoost parameters
            visit https://xgboost.readthedocs.io/en/stable/parameter.html.

        Args:
            batch_size: the number of points sampled every time the sampler is called
            random_state: the random state of the sampler, fixing this number the sampler behaves deterministically
            max_deduplication_passes: the maximum number of deduplication passes
            candidate_pool_size: number of randomly sampled points on which the random forest predictions are evaluated
            colsample_bytree: subsample ratio of columns when constructing each tree
            learning_rate: the learning rate
            max_depth: maximum depth of XGBoost trees
            alpha: L1 regularization term on weights
            n_estimators: number of estimators

        References:
            Lamperti, Roventini, and Sani, "Agent-based model calibration using machine learning surrogates"
        """
        super().__init__(
            batch_size, random_state, max_deduplication_passes, candidate_pool_size
        )

        self._colsample_bytree = colsample_bytree
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._alpha = alpha
        self._n_estimators = n_estimators
        self._xg_regressor: Optional[xgb.XGBRegressor] = None

    @property
    def colsample_bytree(self) -> float:
        """Get the colsample_bytree parameter."""
        return self._colsample_bytree

    @property
    def learning_rate(self) -> float:
        """Get the learning rate."""
        return self._learning_rate

    @property
    def max_depth(self) -> int:
        """Get the maximum tree depth."""
        return self._max_depth

    @property
    def alpha(self) -> float:
        """Get the alpha regularisation parameter."""
        return self._alpha

    @property
    def n_estimators(self) -> int:
        """Get the number of estimators."""
        return self._n_estimators

    @staticmethod
    def _clip_losses(y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Check that loss values fall within the float32 limits needed for XGBoost to work."""
        large_floats = np.where(y >= MAX_FLOAT32)
        small_floats = np.where(y <= MIN_FLOAT32)

        if len(large_floats) == 0 and len(small_floats) == 0:
            return y

        warnings.warn(
            "Found loss values out of float32 limits, clipping them for XGBoost.",
            RuntimeWarning,
        )
        if len(large_floats) > 0:
            y[large_floats] = MAX_FLOAT32 - EPS_FLOAT32

        if len(small_floats) > 0:
            y[small_floats] = MIN_FLOAT32 + EPS_FLOAT32

        return y

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit a xgboost surrogate model."""
        # prepare data
        y = self._clip_losses(y)  # pylint: disable=W0212
        _ = xgb.DMatrix(data=X, label=y)

        # train surrogate
        self._xg_regressor = xgb.XGBRegressor(
            objective="reg:squarederror",  # original: objective ='reg:linear',
            colsample_bytree=self.colsample_bytree,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,  # original: 5
            alpha=self.alpha,
            n_estimators=self.n_estimators,
        )  # original: 10

        self._xg_regressor.fit(X, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict using a xgboost surrogate model."""
        # predict over large pool of candidates
        _ = xgb.DMatrix(data=X)

        self._xg_regressor = cast(xgb.XGBRegressor, self._xg_regressor)
        predictions = self._xg_regressor.predict(X)

        return predictions
