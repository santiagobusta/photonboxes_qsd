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

"""Implementation of :class:`ExtremumConstraint`."""

import operator
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class ExtremumConstraint(Constraint):
    """Bound on a maximum (minimum) over convex (concave) functions."""

    class Conversion(ConstraintConversion):
        """Bound on a maximum or minimum of functions conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            if subtype.relation == Constraint.LE:
                relation = operator.__le__
            else:
                relation = operator.__ge__

            for lhs_type in subtype.lhs_types:
                yield ("con", lhs_type.predict(relation, subtype.rhs_type), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..modeling import Problem

            P = Problem()

            if con.relation == Constraint.LE:
                for x in con.extremum.expressions:
                    P.add_constraint(x <= con.rhs)
            else:
                for x in con.extremum.expressions:
                    P.add_constraint(x >= con.rhs)

            return P

    def __init__(self, extremum, relation, rhs):
        """Construct a :class:`ExtremumConstraint`.

        :param ~picos.expressions.Extremum extremum:
            Left hand side expression.
        :param str relation:
            Constraint relation symbol.
        :param ~picos.expressions.AffineExpression rhs:
            Right hand side expression.
        """
        from ..expressions import (AffineExpression, Extremum, MaximumConvex,
                                   MinimumConcave)

        assert isinstance(extremum, Extremum)
        assert isinstance(rhs, AffineExpression)
        assert relation in self.LE + self.GE
        if relation == self.LE:
            assert isinstance(extremum, MaximumConvex)
        else:
            assert isinstance(extremum, MinimumConcave)
        assert len(rhs) == 1

        self.extremum = extremum
        self.relation = relation
        self.rhs = rhs

        super(ExtremumConstraint, self).__init__(extremum._typeStr)

    Subtype = namedtuple("Subtype", ("lhs_types", "relation", "rhs_type"))

    def _subtype(self):
        return self.Subtype(
            lhs_types=self.extremum.subtype.types,
            relation=self.relation,
            rhs_type=self.rhs.type)

    @classmethod
    def _cost(cls, subtype):
        return len(subtype.lhs_types)

    def _expression_names(self):
        yield "extremum"
        yield "rhs"

    def _str(self):
        if self.relation == self.LE:
            return glyphs.le(self.extremum.string, self.rhs.string)
        else:
            return glyphs.ge(self.extremum.string, self.rhs.string)

    def _get_slack(self):
        if self.relation == self.LE:
            return self.rhs.safe_value - self.extremum.safe_value
        else:
            return self.extremum.safe_value - self.rhs.safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
