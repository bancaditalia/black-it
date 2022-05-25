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
"""This module contains helper function to extract code from the .md files."""

import json
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import mistletoe
from mistletoe.ast_renderer import ASTRenderer

MISTLETOE_CODE_BLOCK_ID = "CodeFence"


def code_block_filter(block_dict: Dict, language: Optional[str] = None) -> bool:
    """
    Check Mistletoe block is a code block.

    Args:
        block_dict: the block dictionary describing a Mistletoe document node
        language: optionally check also the language field

    Returns:
        True if the block satistifes the conditions, False otherwise
    """
    return block_dict["type"] == MISTLETOE_CODE_BLOCK_ID and (
        language is None or block_dict["language"] == language
    )


def python_code_block_filter(block_dict: Dict) -> bool:
    """Filter Python code blocks."""
    return code_block_filter(block_dict, language="python")


def code_block_extractor(child_dict: Dict) -> str:
    """Extract Mistletoe code block from Mistletoe CodeFence child."""
    # we assume that 'children' of CodeFence child has only one child (may be wrong)
    assert len(child_dict["children"]) == 1
    return child_dict["children"][0]["content"]


class BaseTestMarkdownDocs:  # pylint: disable=too-few-public-methods
    """Base test class for testing Markdown documents."""

    DOC_PATH: Path
    blocks: List[Dict]
    code_blocks: List[str]
    python_blocks: List[str]

    @classmethod
    def setup_class(cls) -> None:
        """Set up the test."""
        doc_content = cls.DOC_PATH.read_text()
        doc_file_descriptor = StringIO(doc_content)
        markdown_parsed = mistletoe.markdown(doc_file_descriptor, renderer=ASTRenderer)
        markdown_json = json.loads(markdown_parsed)
        cls.blocks = markdown_json["children"]
        cls.code_blocks = list(
            map(code_block_extractor, filter(code_block_filter, cls.blocks))
        )
        cls.python_blocks = list(
            map(code_block_extractor, filter(python_code_block_filter, cls.blocks))
        )


class BasePythonMarkdownDocs(BaseTestMarkdownDocs):
    """Test Markdown documentation by running Python snippets in sequence."""

    locals: Dict
    globals: Dict

    @classmethod
    def setup_class(cls) -> None:
        """
        Set up class.

        It sets the initial value of locals and globals.
        """
        super().setup_class()
        cls.locals = {}
        cls.globals = {}

    def _assert(self, locals_: Dict, *mocks: Any) -> None:
        """Do assertions after Python code execution."""

    def test_python_blocks(self, *mocks: Any) -> None:
        """Run Python code block in sequence."""
        python_blocks = self.python_blocks

        globals_, locals_ = self.globals, self.locals
        for python_code_block in python_blocks:
            exec(  # nosec # pylint: disable=exec-used
                python_code_block, globals_, locals_
            )
        self._assert(locals_, *mocks)
