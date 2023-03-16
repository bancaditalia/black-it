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

"""Python module to handle extras dependencies loading and import errors.

This is a private module of the library. There should be no point in using it directly from client code.
"""
from __future__ import annotations

import sys

# known extras and their dependencies
_GPY_PACKAGE_NAME = "GPy"
_GP_SAMPLER_EXTRA_NAME = "gp-sampler"

_XGBOOST_PACKAGE_NAME = "xgboost"
_XGBOOST_SAMPLER_EXTRA_NAME = "xgboost-sampler"


class DependencyNotInstalledError(Exception):
    """Library exception for when a required dependency is not installed."""

    def __init__(self, component_name: str, package_name: str, extra_name: str) -> None:
        """Initialize the exception object."""
        message = (
            f"Cannot import package '{package_name}', required by component {component_name}. "
            f"To solve the issue, you can install the extra '{extra_name}': pip install black-it[{extra_name}]"
        )
        super().__init__(message)


class GPyNotSupportedOnPy311Error(Exception):
    """Specific exception class for import error of GPy on Python 3.11."""

    __ERROR_MSG = (
        f"The GaussianProcessSampler depends on '{_GPY_PACKAGE_NAME}', which is not supported on Python 3.11; "
        f"see https://github.com/bancaditalia/black-it/issues/36"
    )

    def __init__(self) -> None:
        """Initialize the exception object."""
        super().__init__(self.__ERROR_MSG)


def _check_import_error_else_raise_exception(
    import_error: ImportError | None,
    component_name: str,
    package_name: str,
    black_it_extra_name: str,
) -> None:
    """Check an import error; raise the DependencyNotInstalledError exception with a useful message.

    Args:
        import_error: the ImportError object generated by the failed attempt. If None, then no error occurred.
        component_name: the component for which the dependency is needed
        package_name: the Python package name of the dependency
        black_it_extra_name: the name of the black-it extra to install to solve the issue.
    """
    if import_error is None:
        # nothing to do.
        return

    # an import error happened; we need to raise error to the caller
    raise DependencyNotInstalledError(component_name, package_name, black_it_extra_name)


def _check_gpy_import_error_else_raise_exception(
    import_error: ImportError | None,
    component_name: str,
    package_name: str,
    black_it_extra_name: str,
) -> None:
    """Check GPy import error and if an error occurred, raise erorr with a useful error message.

    We need to handle two cases:

    - the user is using Python 3.11: the GPy package cannot be installed there;
        see https://github.com/SheffieldML/GPy/issues/998
    - the user did not install the 'gp-sampler' extra.

    Args:
        import_error: the ImportError object generated by the failed attempt. If None, then no error occurred.
        component_name: the component for which the dependency is needed
        package_name: the Python package name of the dependency
        black_it_extra_name: the name of the black-it extra to install to solve the issue.
    """
    if import_error is None:
        # nothing to do.
        return

    if sys.version_info == (3, 11):
        raise GPyNotSupportedOnPy311Error

    _check_import_error_else_raise_exception(
        import_error,
        component_name,
        package_name,
        black_it_extra_name,
    )
