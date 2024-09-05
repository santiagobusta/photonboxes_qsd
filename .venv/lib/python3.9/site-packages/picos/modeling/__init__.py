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

"""Optimization problem modeling toolbox."""

from ..apidoc import api_end, api_start

_API_START = api_start(globals())
# -------------------------------


from .footprint  import Footprint, Specification  # noqa
from .objective  import Objective  # noqa
from .options    import Options  # noqa
from .problem    import Problem, SolutionFailure  # noqa
from .solution   import Solution  # noqa
from .strategy   import NoStrategyFound, Strategy  # noqa
from .quicksolve import find_assignment, maximize, minimize  # noqa


# --------------------------------------
__all__ = api_end(_API_START, globals())
