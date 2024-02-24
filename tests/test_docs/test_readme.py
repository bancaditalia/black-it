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

"""Test the README content."""
from textwrap import indent

from tests.conftest import ROOT_DIR
from tests.test_docs.base import BaseMainExampleDocs
from tests.test_examples.test_main import BEST_PARAMETERS_STR, TRUE_PARAMETERS_STR


class TestReadme(BaseMainExampleDocs):
    """Test the README Python code snippets."""

    DOC_PATH = ROOT_DIR / "README.md"

    def test_quickstart_same_code_of_example(self) -> None:
        """Test that the Python code snippet is the same of the example."""
        example_code = self.extract_example_code()

        quickstart_code_snippet = self.python_blocks[0]
        # test import lines are equal
        nb_import_lines = 8
        quickstart_import_lines = quickstart_code_snippet.splitlines()[:nb_import_lines]
        example_import_lines = example_code.splitlines()[:nb_import_lines]
        assert quickstart_import_lines == example_import_lines

        # test quickstart body is contained in example script
        quickstart_body = "\n".join(
            quickstart_code_snippet.splitlines()[nb_import_lines:],
        )
        assert indent(quickstart_body, " " * 4) in example_code

        # test calibration output
        actual_calibration_output = self.code_blocks[4]
        expected_calibration_output = f"{TRUE_PARAMETERS_STR}\n{BEST_PARAMETERS_STR}\n"
        assert actual_calibration_output == expected_calibration_output
