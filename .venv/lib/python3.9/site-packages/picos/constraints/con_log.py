# ------------------------------------------------------------------------------
# Copyright (C) 2018-2019 Maximilian Stahlberg
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

"""Implementation of :class:`LogConstraint`."""

from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class LogConstraint(Constraint):
    """Lower bound on a logarithm."""

    class ExpConeConversion(ConstraintConversion):
        """Bound on a logarithm to exponential cone constraint conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from . import AffineConstraint, ExpConeConstraint

            yield ("con", AffineConstraint.make_type(dim=1, eq=False), 1)
            yield ("con", ExpConeConstraint.make_type(), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..expressions import ExponentialCone
            from ..modeling import Problem

            x = con.log.x
            t = con.lb

            P = Problem()
            P.add_constraint(x >= 0)
            P.add_constraint((x // 1 // t) << ExponentialCone())
            return P

    def __init__(self, log, lowerBound):
        """Construct a :class:`LogConstraint`.

        :param ~picos.expressions.Logarithm log:
            Constrained expression.
        :param ~picos.expressions.AffineExpression lowerBound:
            Lower bound on the expression.
        """
        from ..expressions import AffineExpression, Logarithm

        assert isinstance(log, Logarithm)
        assert isinstance(lowerBound, AffineExpression)
        assert len(lowerBound) == 1

        self.log = log
        self.lb  = lowerBound

        super(LogConstraint, self).__init__("Logarithmic")

    Subtype = namedtuple("Subtype", ())

    def _subtype(self):
        return self.Subtype()

    @classmethod
    def _cost(cls, subtype):
        return 2

    def _expression_names(self):
        yield "log"
        yield "lb"

    def _str(self):
        return glyphs.ge(self.log.string, self.lb.string)

    def _get_slack(self):
        return self.log.safe_value - self.lb.safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
