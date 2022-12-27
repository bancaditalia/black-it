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
from typing import Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from black_it.samplers.surrogate import MLSurrogateSampler
from black_it.utils.base import _assert


class RandomForestSampler(MLSurrogateSampler):
    """This class implements random forest sampling."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        batch_size: int,
        random_state: Optional[int] = None,
        max_deduplication_passes: int = 5,
        candidate_pool_size: Optional[int] = None,
        n_estimators: int = 500,
        criterion: str = "gini",
        n_classes: int = 10,
    ) -> None:
        """
        Random forest sampling.

        Note: this class makes use of sklearn.ensemble.RandomForestClassifier.

        Args:
            batch_size: the number of points sampled every time the sampler is called
            random_state: the random state of the sampler, fixing this number the sampler behaves deterministically
            max_deduplication_passes: the maximum number of deduplication passes
            candidate_pool_size: number of randomly sampled points on which the random forest predictions are evaluated
            n_estimators: number of trees in the forest
            criterion: the function to measure the quality of a split.
            n_classes: the number of classes used in the random forest. The classes are selected as the quantiles
                of the distribution of loss values.
        """
        _assert(
            n_classes > 2,
            "'n_classes' should be at least 2 to provide meaningful results",
        )

        super().__init__(
            batch_size, random_state, max_deduplication_passes, candidate_pool_size
        )

        self._n_estimators = n_estimators
        self._criterion = criterion
        self._n_classes = n_classes
        self._classifier: Optional[RandomForestClassifier] = None

    @property
    def n_estimators(self) -> int:
        """Get the number of estimators."""
        return self._n_estimators

    @property
    def criterion(self) -> str:
        """Get the criterion."""
        return self._criterion

    @property
    def n_classes(self) -> int:
        """Get the number of classes."""
        return self._n_classes

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit a random forest surrogate model."""
        # Train surrogate

        X, y_cat, _existing_points_quantiles = self.prepare_data_for_classifier(
            X, y, self.n_classes
        )

        self._classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            n_jobs=-1,
            random_state=self._get_random_seed(),
        )
        self._classifier.fit(X, y_cat)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict using a random forest surrogate model."""
        # Predict quantiles
        self._classifier = cast(RandomForestClassifier, self._classifier)
        predicted_points_quantiles: NDArray[np.float64] = self._classifier.predict(X)

        return predicted_points_quantiles

    @staticmethod
    def prepare_data_for_classifier(
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
        num_bins: int,
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
