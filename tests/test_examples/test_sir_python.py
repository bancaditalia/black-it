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

"""Test the SIR model implementation in Python."""

import numpy as np
import pytest

try:
    from examples.models.sir.sir_python import SIR, SIR_w_breaks
except ModuleNotFoundError as e:
    pytest.skip(
        f"skipping tests for SIR python models, reason: {str(e)}",
        allow_module_level=True,
    )

from tests.conftest import TEST_DIR


def test_sir() -> None:
    """Test the 'SIR' function in examples/models/sir/sir_python.py."""
    expected_output = np.load(TEST_DIR / "fixtures" / "data" / "test_sir_python.npy")
    model_seed = 0

    lattice_order = 20
    rewire_probability = 0.2
    percentage_infected = 0.05
    beta = 0.2
    gamma = 0.15
    networkx_seed = 0
    theta = [
        lattice_order,
        rewire_probability,
        percentage_infected,
        beta,
        gamma,
        networkx_seed,
    ]

    n = 100
    output = SIR(theta, n, seed=model_seed)

    assert np.isclose(output, expected_output).all()


def test_sir_w_breaks() -> None:
    """Test the 'SIR_w_breaks' function in examples/models/sir/sir_python.py."""
    expected_output = np.load(
        TEST_DIR / "fixtures" / "data" / "test_sir_w_breaks_python.npy",
    )
    model_seed = 0

    lattice_order = 20
    rewire_probability = 0.2
    percentage_infected = 0.05
    beta_1 = 0.2
    gamma_1 = 0.15
    beta_2 = 0.3
    beta_3 = 0.1
    beta_4 = 0.01
    t_break_1 = 10
    t_break_2 = 20
    t_break_3 = 30
    networkx_seed = 0
    theta = [
        lattice_order,
        rewire_probability,
        percentage_infected,
        beta_1,
        gamma_1,
        beta_2,
        beta_3,
        beta_4,
        t_break_1,
        t_break_2,
        t_break_3,
        networkx_seed,
    ]

    n = 100
    output = SIR_w_breaks(theta, n, seed=model_seed)

    assert np.isclose(output, expected_output).all()
