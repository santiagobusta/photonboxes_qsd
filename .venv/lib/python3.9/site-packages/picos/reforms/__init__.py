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

"""Optimization problem reformulation recipes."""

from ..apidoc import api_end, api_start

_API_START = api_start(globals())
# -------------------------------


from .reformulation import Reformulation  # noqa

# Reformulations that appear in SORTED_REFORMS.
from .reform_epigraph import EpigraphReformulation  # noqa
from .reform_constraint import *  # noqa
from .reform_constraint import TOPOSORTED_REFORMS as _SORTED_CONREFORMS  # noqa
from .reform_dualize import Dualization  # noqa

# Mandatory reformulations that do not appear in SORTED_REFORMS.
from .reform_options import ExtraOptions  # noqa

#: A sequence of reformulations in topological order.
SORTED_REFORMS = [EpigraphReformulation] + _SORTED_CONREFORMS + [Dualization]


# --------------------------------------
__all__ = api_end(_API_START, globals())
