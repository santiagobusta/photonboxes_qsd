# ------------------------------------------------------------------------------
# Copyright (C) 2020 Maximilian Stahlberg
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

"""Constraint types with an explicit representation of data uncertainty."""

from ...apidoc import api_end, api_start

_API_START = api_start(globals())
# -------------------------------


# TODO: UncertainConstraint base class?
from .ucon_ball_norm import BallUncertainNormConstraint  # noqa
from .ucon_conic_aff import ConicallyUncertainAffineConstraint  # noqa
from .ucon_mom_pwl import MomentAmbiguousExtremumAffineConstraint  # noqa
from .ucon_mom_sqnorm import MomentAmbiguousSquaredNormConstraint  # noqa
from .ucon_scen_conic import ScenarioUncertainConicConstraint  # noqa
from .ucon_ws_pwl import WassersteinAmbiguousExtremumAffineConstraint  # noqa
from .ucon_ws_sqnorm import WassersteinAmbiguousSquaredNormConstraint  # noqa


# --------------------------------------
__all__ = api_end(_API_START, globals())
