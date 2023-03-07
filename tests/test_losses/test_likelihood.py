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
"""This module contains tests for the Likelihood loss."""
import numpy as np

from black_it.loss_functions.likelihood import LikelihoodLoss


def test_likelihood_1d() -> None:
    """Test the computation of the Likelihood in the Likelihood loss in 1d."""
    # sample from a Gaussian distribution.
    np.random.seed(11)
    real_data = np.random.normal(0, 1, size=(7, 1))

    expected_neg_log_likelihood = -np.sum(
        -0.5 * np.sum(real_data**2, axis=1) - 0.5 * np.log(2.0 * np.pi), axis=0
    )
    expected_likelihood = np.exp(-expected_neg_log_likelihood)

    sim_data_ensemble = np.random.normal(0, 1, size=(3, 100000, 1))
    loss = LikelihoodLoss(h="silverman")
    neg_log_lik = loss.compute_loss(sim_data_ensemble, real_data)
    lik = np.exp(-neg_log_lik)
    assert np.isclose(lik, expected_likelihood, rtol=0.1)


def test_likelihood_2d() -> None:
    """Test the computation of the Likelihood in the Likelihood loss in 2d."""
    # sample from a Gaussian distribution.
    np.random.seed(11)
    real_data = np.random.normal(0, 1, size=(10, 2))

    expected_neg_log_likelihood = -np.sum(
        -0.5 * np.sum(real_data**2, axis=1) - 2.0 / 2.0 * np.log(2.0 * np.pi), axis=0
    )
    expected_likelihood = np.exp(-expected_neg_log_likelihood)
    sim_data_ensemble = np.random.normal(0, 1, size=(1, 1000000, 2))
    loss = LikelihoodLoss(h=1.0)
    neg_log_lik = loss.compute_loss(sim_data_ensemble, real_data)
    lik = np.exp(-neg_log_lik)
    assert np.isclose(lik, expected_likelihood, rtol=0.1)


def test_likelihood_2d_wsigma() -> None:
    """Test the computation of the Likelihood in the Likelihood loss in 2d."""
    # sample from a Gaussian distribution.
    np.random.seed(11)
    sigma, D = 3.0, 2
    real_data = np.random.normal(0, sigma, size=(10, D))

    expected_neg_log_likelihood = -np.sum(
        -0.5 / sigma**2 * np.sum(real_data**2, axis=1)
        - D / 2.0 * np.log(2.0 * np.pi * sigma**2),
        axis=0,
    )
    expected_likelihood = np.exp(-expected_neg_log_likelihood)

    sim_data_ensemble = np.random.normal(0, sigma, size=(1, 1000000, D))
    loss = LikelihoodLoss(h=sigma)
    neg_log_lik = loss.compute_loss(sim_data_ensemble, real_data)
    lik = np.exp(-neg_log_lik)
    assert np.isclose(lik, expected_likelihood, rtol=0.1)
