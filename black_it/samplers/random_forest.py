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
from numpy.random import default_rng
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from black_it.samplers.base import BaseSampler
from black_it.search_space import SearchSpace
from black_it.utils.base import digitize_data


class RandomForestSampler(BaseSampler):
    """This class implements random forest sampling."""

    def __init__(
        self,
        batch_size: int,
        internal_seed: int = 0,
        candidate_pool_size: Optional[int] = None,
        n_estimators: int = 500,
        criterion: str = "gini",
    ) -> None:
        """
        Random forest sampling.

        Note: this class is a wrapper of sklearn.ensemble.RandomForestClassifier.

        Args:
            batch_size: the number of points sampled every time the sampler is called
            internal_seed: the internal state of the sampler, fixing this numbers the sampler behaves deterministically
            candidate_pool_size: number of randomly sampled points on which the random forest predictions are evaluated
            n_estimators: number of trees in the forest
            criterion: The function to measure the quality of a split.
        """
        super().__init__(batch_size, internal_seed)

        self.n_estimators = n_estimators
        self.criterion = criterion

        if candidate_pool_size is not None:
            self.candidate_pool_size = candidate_pool_size
        else:
            self.candidate_pool_size = 1000 * batch_size

    def single_sample(  # noqa
        self,
        seed: int,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        raise NotImplementedError(
            "for RandomForestSampler the parallelization is hard coded in sample"
        )

    def sample(
        self,
        search_space: SearchSpace,
        existing_points: NDArray[np.float64],
        existing_losses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Sample from the search space.

        Args:
            search_space: an object containing the details of the parameter search space
            existing_points: the parameters already sampled
            existing_losses: the loss corresponding to the sampled parameters

        Returns:
            the sampled parameters
        """
        # Get large candidate pool
        candidates = self._get_candidates(existing_points, search_space)

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
            random_state=self.internal_seed,
        )
        classifier.fit(x, y)
        # Predict quantiles
        predicted_points_quantiles: NDArray[np.float64] = classifier.predict(candidates)
        # Sort params by predicted quantile
        sorting_indices: NDArray[np.float64] = np.argsort(predicted_points_quantiles)
        sampled_points: NDArray[np.float64] = candidates[sorting_indices][
            : self.batch_size
        ]

        self.internal_seed += self.batch_size

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

    def _get_candidates(
        self, existing_points: NDArray[np.float64], search_space: SearchSpace
    ) -> NDArray[np.float64]:
        """
        Get the set of candidate parameters.

        Args:
            existing_points: the parameters already sampled
            search_space: the search space

        Returns:
            the list of candidates, without duplicates.
        """

        def sample(nb_samples: int) -> NDArray[np.float64]:
            random_generator = default_rng(self.internal_seed)
            candidates = np.zeros((nb_samples, search_space.dims))
            for i, params in enumerate(search_space.param_grid):
                candidates[:, i] = random_generator.choice(params, size=(nb_samples,))
            return candidates

        samples = sample(self.candidate_pool_size)

        for n in range(self.max_duplication_passes):

            duplicates = self.find_and_get_duplicates(samples, existing_points)

            num_duplicates = len(duplicates)

            if num_duplicates == 0:
                break
            new_samples = sample(num_duplicates)
            samples[duplicates] = new_samples

            if n == self.max_duplication_passes - 1:
                print(
                    f"Warning: Repeated samples still found after {self.max_duplication_passes} duplication passes."
                    " This is probably due to a small search space."
                )

        return samples
