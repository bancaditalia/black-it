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

"""This module contains the implementation of the random forest sampling."""
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from black_it.samplers.base import BaseSampler
from black_it.samplers.random_uniform import RandomUniformSampler
from black_it.search_space import SearchSpace
from black_it.utils.base import digitize_data


class RandomForestSampler(BaseSampler):
    """This class implements random forest sampling."""

    def __init__(
        self,
        batch_size: int,
        random_state: Optional[int] = None,
        max_deduplication_passes: int = 5,
        candidate_pool_size: Optional[int] = None,
        n_estimators: int = 500,
        criterion: str = "gini",
    ) -> None:
        """
        Random forest sampling.

        Note: this class is a wrapper of sklearn.ensemble.RandomForestClassifier.

        Args:
            batch_size: the number of points sampled every time the sampler is called
            random_state: the random state of the sampler, fixing this number the sampler behaves deterministically
            max_deduplication_passes: the maximum number of deduplication passes
            candidate_pool_size: number of randomly sampled points on which the random forest predictions are evaluated
            n_estimators: number of trees in the forest
            criterion: The function to measure the quality of a split.
        """
        super().__init__(batch_size, random_state, max_deduplication_passes)

        self._n_estimators = n_estimators
        self._criterion = criterion

        if candidate_pool_size is not None:
            self._candidate_pool_size = candidate_pool_size
        else:
            self._candidate_pool_size = 1000 * batch_size

    @property
    def n_estimators(self) -> int:
        """Get the number of estimators."""
        return self._n_estimators

    @property
    def criterion(self) -> str:
        """Get the criterion."""
        return self._criterion

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
        # Get large candidate pool
        candidates = RandomUniformSampler(
            self.candidate_pool_size, random_state=self._get_random_seed()
        ).sample_batch(
            self.candidate_pool_size, search_space, existing_points, existing_losses
        )

        # Train surrogate
        x: NDArray[np.float64]
        y: NDArray[np.int64]
        x, y, _existing_points_quantiles = self.prepare_data_for_classifier(
            existing_points, existing_losses
        )

        classifier: RandomForestClassifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            n_jobs=-1,
            random_state=self._get_random_seed(),
        )
        classifier.fit(x, y)
        # Predict quantiles
        predicted_points_quantiles: NDArray[np.float64] = classifier.predict(candidates)
        # Sort params by predicted quantile
        sorting_indices: NDArray[np.int64] = np.argsort(predicted_points_quantiles)
        sampled_points: NDArray[np.float64] = candidates[sorting_indices][:batch_size]

        return digitize_data(sampled_points, search_space.param_grid)

    @staticmethod
    def prepare_data_for_classifier(
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
        num_bins: int = 10,
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
        """
        Prepare data for the classifier.

        Args:
            existing_points: the parameters already sampled
            existing_losses: the loss corresponding to the sampled parameters
            num_bins: the number of bins

        Returns:
            A triple (x, y, quantiles), where
                - x is the vector of training data
                - y is the vector of targets
                - the quantiles
        """
        x: NDArray[np.float64] = existing_points
        y: NDArray[np.float64] = existing_losses

        cutoffs: NDArray[np.float64] = np.linspace(0, 1, num_bins + 1)
        quantiles: NDArray[np.float64] = np.zeros(num_bins + 1)

        for i in range(num_bins - 1):
            quantiles[i + 1] = np.quantile(y, cutoffs[i + 1])

        quantiles[-1] = np.max(y)

        y_cat: NDArray[np.int64] = np.digitize(y, quantiles, right=True)
        y_cat = y_cat - 1

        return x, y_cat, quantiles
