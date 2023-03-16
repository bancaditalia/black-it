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

"""Test the README content."""
from textwrap import indent

from tests.conftest import DOCS_DIR
from tests.test_docs.base import BaseMainExampleDocs


class TestDocsIndex(BaseMainExampleDocs):
    """Test the index page of the documentation."""

    DOC_PATH = DOCS_DIR / "index.md"

    def test_quickstart_same_code_of_example(self) -> None:
        """Test that the Python code snippet is the same of the example."""
        example_code = self.extract_example_code()

        calibrator_code_snippet = self.python_blocks[0]
        model_code_snippet = self.python_blocks[1]
        samplers_code_snippet = self.python_blocks[2]

        # test calibrator code snippet is contained in example script
        assert indent(calibrator_code_snippet, " " * 4) in example_code
        assert indent(model_code_snippet, " " * 4) in example_code
        assert indent(samplers_code_snippet, " " * 4) in example_code
