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

"""Base module for example tests."""
import sys
from pathlib import Path
from typing import Sequence

import pytest

from tests.utils.base import PopenResult, run_process


class BaseMainExampleTestClass:
    """Base test class for the main example."""

    script_path: Path
    timeout: float
    expected_lines: Sequence[str] = ()

    def test_run(self) -> None:
        """Run the test."""
        if not self.script_path.exists():
            pytest.fail(f"script path {self.script_path} does not exist.")

        try:
            process_output: PopenResult = run_process(
                [sys.executable, str(self.script_path)],
                timeout=self.timeout,
            )
        except RuntimeError as exc:
            pytest.fail(str(exc))
            return

        assert (
            process_output.returncode == 0
        ), f"{str(self.script_path)} exited with error code {process_output.returncode}"

        assert all(
            line in process_output.stdout for line in self.expected_lines
        ), f"could not find {self.expected_lines} in stdout={process_output.stdout}"
