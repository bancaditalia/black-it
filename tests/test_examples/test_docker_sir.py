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

"""Test the main example."""
import pytest

from tests.conftest import ROOT_DIR
from tests.test_examples.base import BaseMainExampleTestClass
from tests.utils.base import requires_docker, skip_on_windows

EXAMPLE_DOCKER_SIR_SCRIPT_PATH = ROOT_DIR / "examples" / "docker-sir.py"


@pytest.mark.e2e
@requires_docker
@skip_on_windows
class TestDockerSirMainExample(BaseMainExampleTestClass):
    """Test that example docker-sir can be run successfully."""

    script_path = EXAMPLE_DOCKER_SIR_SCRIPT_PATH
    timeout = 300
