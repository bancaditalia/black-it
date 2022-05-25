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
"""This module contains tests for the samplers.base module."""
import numpy as np

from black_it.samplers.r_sequence import RSequenceSampler
from black_it.search_space import SearchSpace

expected_params = np.array(
    [
        [0.25, 0.07],
        [0.01, 0.64],
        [0.76, 0.21],
        [0.52, 0.78],
        [0.27, 0.35],
        [0.03, 0.92],
        [0.78, 0.49],
        [0.54, 0.06],
    ]
)


def test_rsequence_2d() -> None:
    """Test the r-sequence sampler, 2d."""
    sampler = RSequenceSampler(batch_size=8, internal_seed=0)
    param_grid = SearchSpace(
        parameters_bounds=np.array([[0, 1], [0, 1]]).T,
        parameters_precision=np.array([0.01, 0.01]),
        verbose=False,
    )
    new_params = sampler.sample(param_grid, np.zeros((0, 2)), np.zeros((0, 2)))
    print(new_params)
    assert np.allclose(expected_params, new_params)
