# ------------------------------------------------------------------------------
# Copyright (C) 2019 Maximilian Stahlberg
#
# This file is part of PICOS.
#
# PICOS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# PICOS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------

"""Functions to support the automatic API documentation.

The environment variable ``PICOS_WARN_MISSING_DOCSTRINGS`` can be set to warn
about missing docstrings for top level objects of a module.
"""

import warnings
from types import ModuleType

_API_START = set(globals())  # api_start is not yet defined.
# -------------------------


def api_start(theGlobals):
    """Mark the start of a module's content.

    Store the returned value and pass it to :func:`api_end`.
    """
    return set(theGlobals)


def api_end(startGlobals, endGlobals):
    """Mark the end of a module's content.

    Store the returned value in __all__.
    """
    names = sorted(set(name for name, obj in endGlobals.items()
        if not name.startswith("_") and not isinstance(obj, ModuleType))
        .difference(startGlobals))

    from . import settings

    if settings.WARN_MISSING_DOCSTRINGS:
        for name in names:
            if not endGlobals[name].__doc__:
                warnings.warn(
                    "Top level object {}.{} has empty docstring."
                    .format(endGlobals["__name__"], name), stacklevel=3)

    return names


# --------------------------------------
__all__ = api_end(_API_START, globals())
