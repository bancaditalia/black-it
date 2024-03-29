# Black-box ABM Calibration Kit (Black-it)
# Copyright (C) 2021-2024 Banca d'Italia
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


def test_get_mom_ts(rng: np.random.Generator) -> None:
    """Test 'get_mom_ts'."""
    expected_mom_ts = np.asarray(
        [
            [
                0.5047481878242738,
                0.4577343452959496,
                0.45960556439046707,
                0.4939260177141695,
                0.49227447674825114,
            ],
            [
                0.3043545480614458,
                0.2669628026226539,
                0.28423903693391567,
                0.30245153381716033,
                0.2819981640130968,
            ],
            [
                -0.4482110315558403,
                0.6542053297744806,
                0.5308172556250059,
                -0.3849423235845297,
                0.31557366158556094,
            ],
            [
                -1.0657862342927136,
                -1.0000694770963234,
                -1.0101207985128777,
                -1.0782685600527042,
                -1.0462395326269132,
            ],
            [
                -0.09320904577520875,
                0.0686169052226043,
                0.1355394841358424,
                0.09653107323762894,
                0.05311636958390115,
            ],
            [
                0.12169737118483734,
                0.0810227210130983,
                0.061958779523484865,
                0.08490375108820097,
                0.006039369703180401,
            ],
            [
                -0.2025899142653773,
                -0.001258750058814992,
                0.12789551725231704,
                -0.005223658816300541,
                -0.051171362779900316,
            ],
            [
                0.0895667296886336,
                -0.1207015312253511,
                0.0636018350025852,
                -0.028626794182293015,
                -0.033236042041738946,
            ],
            [
                -0.03860009031959613,
                -0.10752955274899662,
                -0.13297311479661147,
                0.1362727715117892,
                -0.20066288961998208,
            ],
            [
                0.3705043372180348,
                0.2908837829651126,
                0.29957236943742727,
                0.32868245323966144,
                0.3219905265411198,
            ],
            [
                0.2532444717435027,
                0.22136038120813056,
                0.22607782802456683,
                0.23746687234565775,
                0.21518362830554183,
            ],
            [
                0.6878867186665477,
                0.8516810049012855,
                0.9249968652455888,
                0.8402319231441494,
                0.8433453672087052,
            ],
            [
                -0.9980698948119241,
                -0.8313118851534257,
                -0.7294913128812629,
                -0.781988185137505,
                -0.8992290494498107,
            ],
            [
                0.15460527502011878,
                0.07324077575840597,
                0.129091118386242,
                0.12525686451619064,
                0.04116717905886786,
            ],
            [
                0.132780107882644,
                -0.0652821068259806,
                -0.012893435354472009,
                -0.038207396068454,
                -0.05256884500600915,
            ],
            [
                0.1309359433071446,
                0.09360205934372054,
                0.04414471989530231,
                -0.0632467584785815,
                0.002402854248802498,
            ],
            [
                0.016550810028943094,
                0.03214763642101067,
                0.0070285228164037265,
                -0.03352305029619496,
                -0.019493316543544734,
            ],
            [
                -0.08443798877034842,
                -0.04227111216992749,
                -0.053302703788843504,
                0.10373275400311846,
                -0.06054093058906396,
            ],
        ],
    )
    time_series = rng.random(size=(100, 5))
    mom_ts = get_mom_ts(time_series)
    assert np.allclose(mom_ts, expected_mom_ts)
