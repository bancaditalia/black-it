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
from typing import Optional

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from black_it.samplers.base import BaseSampler
from black_it.samplers.random_uniform import RandomUniformSampler
from black_it.search_space import SearchSpace
from black_it.utils.base import digitize_data


class XGBoostSampler(BaseSampler):
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
        alpha: float = 10.0,
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
        super().__init__(batch_size, random_state, max_deduplication_passes)

        self._colsample_bytree = colsample_bytree
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._alpha = alpha
        self._n_estimators = n_estimators

        if candidate_pool_size is not None:
            self._candidate_pool_size = candidate_pool_size
        else:
            self._candidate_pool_size = 1000 * batch_size

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

    @property
    def candidate_pool_size(self) -> int:
        """Get the candidate pool size."""
        return self._candidate_pool_size

    def sample_batch(
        self,
        batch_size: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Sample from the search space.

        Args:
            batch_size: the number of points to sample
            search_space: an object containing the details of the parameter search space
            existing_points: the parameters already sampled
            existing_losses: the loss corresponding to the sampled parameters

        Returns:
            the sampled parameters
        """
        # get large candidate pool
        candidates = RandomUniformSampler(
            self.candidate_pool_size, random_state=self._get_random_seed()
        ).sample_batch(
            self.candidate_pool_size, search_space, existing_points, existing_losses
        )

        # prepare data
        X = existing_points
        y = existing_losses

        _ = xgb.DMatrix(data=X, label=y)

        # train surrogate
        xg_reg = xgb.XGBRegressor(
            objective="reg:squarederror",  # original: objective ='reg:linear',
            colsample_bytree=self.colsample_bytree,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,  # original: 5
            alpha=self.alpha,
            n_estimators=self.n_estimators,
        )  # original: 10

        xg_reg.fit(X, y)

        # predict over large pool of candidates
        _ = xgb.DMatrix(data=candidates)

        predicted_points = xg_reg.predict(candidates)

        # sort params by predicted loss value
        sorting_indices: NDArray[np.int64] = np.argsort(predicted_points)
        sampled_points: NDArray[np.float64] = candidates[sorting_indices][:batch_size]

        return digitize_data(sampled_points, search_space.param_grid)
