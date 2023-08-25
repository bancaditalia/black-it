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

"""Generic utility functions."""
import dataclasses
import shutil
import signal
import subprocess  # nosec B404
import sys
from functools import wraps
from typing import Callable, List, Type, Union

import pytest

from tests.conftest import DEFAULT_SUBPROCESS_TIMEOUT


@dataclasses.dataclass(frozen=True)
class PopenResult:
    """Dataclass to represent a Popen result."""

    returncode: int
    stdout: str
    stderr: str


def run_process(
    command: List[str], timeout: float = DEFAULT_SUBPROCESS_TIMEOUT
) -> PopenResult:
    """Run a process, and wait for it to stop.

    Args:
        command: the command to run.
        timeout: the timeout

    Returns:
        the Popen object

    Raises:
        RuntimeError: if the process cannot be stopped.
    """
    process = subprocess.Popen(  # nosec B603
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
    except Exception as exc:
        # clean up process
        # first, try to stop it gracefully
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)
        poll = process.poll()
        if poll is None:
            # if graceful stop failed, kill the process
            process.terminate()
        raise RuntimeError(f"command {command} failed with error: {exc}") from exc

    returncode = process.returncode
    stdout = stdout_bytes.decode() if stdout_bytes else ""
    stderr = stderr_bytes.decode() if stderr_bytes else ""
    return PopenResult(returncode=returncode, stdout=stdout, stderr=stderr)


def requires_docker(pytest_func_or_cls: Callable) -> Callable:
    """Wrap a Pytest function that requires docker.

    Args:
        pytest_func_or_cls: the pytest function/class to wrap.

    Returns:
        the wrapped function/class
    """
    return requires_binary("docker")(pytest_func_or_cls)


def requires_binary(binary_name: str) -> Callable:
    """Decorate a pytest class or method to skip test if a binary is not installed locally.

    Args:
        binary_name: the binary name

    Returns:
        a wrapper for functions/classes
    """

    def action() -> None:
        binary_path = shutil.which(binary_name)
        if binary_path is None:
            pytest.skip(f"Binary {binary_name} not found")

    return pytest_decorator_factory(action)


def pytest_decorator_factory(action: Callable) -> Callable:
    """Wrap a Pytest function/method/class to include the running of an action.

    For methods/functions 'f', the wrapper function looks like:

    def wrapper(*args, **kwargs):
        action()
        return f(*args, **kwargs)

    For classes 'c', the wrapper class is:
     - a subclass of 'c' (inherits all test methods)
     - same name of the parent
     - but wrapped setup_class

     If 'c' does not have setup_class, setup_class = action. Else,
     new_setup_class is equal to wrapper above (with f = setup_class)

    Args:
         action: the preamble function to call before the test or the setup_class method

    Returns:
        the Pytest wrapper for functions or classes.
    """

    def decorator(pytest_func_or_cls: Union[Callable, Type]) -> Callable:
        """Implement the decorator.

        Args:
            pytest_func_or_cls: a Pytest function or class.

        Returns:
            the decorator.
        """

        @wraps(pytest_func_or_cls)
        def wrapper(*args, **kwargs):  # type: ignore
            action()
            return pytest_func_or_cls(*args, **kwargs)

        if isinstance(pytest_func_or_cls, type):
            # here if the wrappee is a test class
            # wrap setup_class with 'action' if setup_class is defined
            if hasattr(pytest_func_or_cls, "setup_class"):
                new_setup_class = decorator(pytest_func_or_cls.setup_class)  # type: ignore
            else:
                new_setup_class = action
            # return a new subclass with same name, parent test methods,
            # but setup_class method wrapped.
            return type(
                pytest_func_or_cls.__name__,
                (pytest_func_or_cls,),
                {
                    "setup_class": new_setup_class,
                    "_skipped": True,
                },
            )
        # here if the wrappee is a test function (not a class)
        return wrapper

    return decorator


skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32", reason="This test cannot be ran on Windows"
)
