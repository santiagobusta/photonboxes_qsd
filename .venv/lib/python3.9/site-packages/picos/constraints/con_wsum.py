# ------------------------------------------------------------------------------
# Copyright (C) 2021 Maximilian Stahlberg
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

"""Implementation of :class:`WeightedSumConstraint`."""

import operator
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class WeightedSumConstraint(Constraint):
    """Bound on a convex or concave weighted sum of expressions."""

    class Conversion(ConstraintConversion):
        """Bound on a weighted sum of expressions conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ..expressions import AffineExpression, RealVariable
            from . import AffineConstraint

            if subtype.relation == Constraint.LE:
                fwdrel, bckrel = operator.__le__, operator.__ge__
            else:
                fwdrel, bckrel = operator.__ge__, operator.__le__

            n = len(subtype.lhs_types)

            assert n > 0

            yield ("var", RealVariable.make_var_type(dim=n, bnd=0), 1)
            yield ("con", AffineConstraint.make_type(dim=1, eq=False), 1)

            rhs_type = AffineExpression.make_type(  # Element of aux. variable.
                shape=(1, 1), constant=False, nonneg=False
            )

            for lhs_type, nnw in zip(subtype.lhs_types, subtype.nonneg_weights):
                if nnw:
                    yield ("con", lhs_type.predict(fwdrel, rhs_type), 1)
                else:
                    yield ("con", lhs_type.predict(bckrel, rhs_type), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..expressions import RealVariable
            from ..modeling import Problem

            n = len(con.wsum.expressions)
            w = con.wsum.weights

            assert n > 0

            t = RealVariable("__t", len(con.wsum.expressions))

            P = Problem()

            if con.relation == Constraint.LE:
                P += w.T * t <= con.rhs

                for i, x in enumerate(con.wsum.expressions):
                    if w[i].value >= 0:
                        P.add_constraint(x <= t[i])
                    else:
                        P.add_constraint(x >= t[i])
            else:
                P += w.T * t >= con.rhs

                for i, x in enumerate(con.wsum.expressions):
                    if w[i].value >= 0:
                        P.add_constraint(x >= t[i])
                    else:
                        P.add_constraint(x <= t[i])

            return P

    def __init__(self, wsum, relation, rhs):
        """Construct a :class:`WeightedSumConstraint`.

        :param ~picos.expressions.WeightedSum wsum:
            Left hand side expression.
        :param str relation:
            Constraint relation symbol.
        :param ~picos.expressions.AffineExpression rhs:
            Right hand side expression.
        """
        from ..expressions import AffineExpression, WeightedSum

        assert isinstance(wsum, WeightedSum)
        assert isinstance(rhs, AffineExpression)
        assert relation in self.LE + self.GE
        if relation == self.LE:
            assert wsum.convex
        else:
            assert wsum.concave
        assert len(rhs) == 1

        self.wsum = wsum
        self.relation = relation
        self.rhs = rhs

        super(WeightedSumConstraint, self).__init__(wsum._typeStr)

    Subtype = namedtuple("Subtype", (
        "lhs_types", "relation", "rhs_type", "nonneg_weights"))

    def _subtype(self):
        return self.Subtype(
            lhs_types=self.wsum.subtype.types,
            relation=self.relation,
            rhs_type=self.rhs.type,
            nonneg_weights=tuple(self.wsum.weights.np >= 0))

    @classmethod
    def _cost(cls, subtype):
        return len(subtype.lhs_types)

    def _expression_names(self):
        yield "wsum"
        yield "rhs"

    def _str(self):
        if self.relation == self.LE:
            return glyphs.le(self.wsum.string, self.rhs.string)
        else:
            return glyphs.ge(self.wsum.string, self.rhs.string)

    def _get_slack(self):
        if self.relation == self.LE:
            return self.rhs.safe_value - self.wsum.safe_value
        else:
            return self.wsum.safe_value - self.rhs.safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
