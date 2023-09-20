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

"""Base module for docs tests."""
import re
from pathlib import Path

from tests.conftest import ROOT_DIR
from tests.utils.docs import BaseTestMarkdownDocs


class BaseMainExampleDocs(BaseTestMarkdownDocs):
    """Test consistency of main example documentation."""

    DOC_PATH: Path
    EXAMPLE_PATH = Path(ROOT_DIR, "examples", "main.py")

    def extract_example_code(self) -> str:
        """Extract example code form the script."""
        example_code = self.EXAMPLE_PATH.read_text(encoding="utf-8")

        # remove shebang
        example_code = example_code.replace("#!/usr/bin/env python3\n", "")

        # remove copyright notice and license
        example_code = re.sub("^#.*\n(\n?)", "", example_code, flags=re.MULTILINE)

        # remove docstring
        example_code = re.sub('""".*"""\n', "", example_code)

        # remove if __name__ == "__main__":
        example_code = example_code.replace('if __name__ == "__main__":\n', "")

        return example_code
