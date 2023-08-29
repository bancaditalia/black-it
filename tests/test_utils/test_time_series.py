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

"""This module contains tests for the utils.time_series module."""
import numpy as np

from black_it.utils.time_series import get_mom_ts


def test_get_mom_ts() -> None:
    """Test 'get_mom_ts'."""
    expected_mom_ts = np.asarray(
        [
            [0.50471071, 0.52523233, 0.48769797, 0.52125415, 0.45391341],
            [0.29597164, 0.29840167, 0.29334348, 0.29175012, 0.30658573],
            [-0.39247097, -0.53521998, -0.48164217, -0.41514771, 0.64604918],
            [-1.04321643, -1.04806895, -1.05729681, -1.04166884, -1.07053431],
            [-0.00826626, -0.1310761, -0.01714527, 0.05582763, -0.07196251],
            [0.08511583, 0.05062704, -0.10239579, 0.05814917, -0.15244803],
            [-0.1573025, -0.03237303, 0.14330391, 0.03944553, 0.03682953],
            [-0.13205127, 0.02166404, -0.06635989, 0.01231925, 0.07872452],
            [-0.0594593, -0.16828977, -0.08318478, 0.05319515, 0.00255055],
            [0.3472227, 0.36920251, 0.33991259, 0.32411733, 0.36455635],
            [0.23971816, 0.25552126, 0.24285517, 0.2348814, 0.25863802],
            [0.90238066, 0.74258728, 0.80671623, 0.89465546, 0.59934486],
            [-0.65135278, -0.96315459, -0.93646035, -0.78143557, -1.02558061],
            [0.15463938, 0.19370484, -0.03971417, 0.04101357, -0.05957849],
            [0.00551848, 0.05268421, -0.16417644, 0.11792084, -0.03113898],
            [-0.11193418, 0.11907523, -0.04560956, 0.03323841, -0.11316762],
            [-0.02379168, -0.00181507, -0.00476993, 0.13203911, -0.06712482],
            [0.00665585, -0.06428739, -0.11528277, 0.02027778, -0.07112011],
        ],
    )
    np.random.seed(42)
    time_series = np.random.rand(100, 5)
    mom_ts = get_mom_ts(time_series)
    assert np.allclose(mom_ts, expected_mom_ts)
