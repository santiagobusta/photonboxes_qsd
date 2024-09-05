# ------------------------------------------------------------------------------
# Copyright (C) 2012-2017 Guillaume Sagnol
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

"""Implementation of :class:`DetRootNConstraint`."""

from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class DetRootNConstraint(Constraint):
    """Lower bound on the :math:`n`-th root of a matrix determinant."""

    class Conversion(ConstraintConversion):
        """:math:`n`-th root of a matrix determinant constraint conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ..expressions import LowerTriangularVariable
            from . import (ComplexLMIConstraint, GeometricMeanConstraint,
                           LMIConstraint)

            n = subtype.diag
            r = (n * (n + 1)) // 2

            yield (
                "var", LowerTriangularVariable.make_var_type(dim=r, bnd=0), 1)

            if subtype.complex:
                yield ("con", ComplexLMIConstraint.make_type(diag=2*n), 1)
            else:
                yield ("con", LMIConstraint.make_type(diag=2*n), 1)

            yield ("con", GeometricMeanConstraint.make_type(argdim=n), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..modeling import Problem
            from ..expressions import GeometricMean, LowerTriangularVariable
            from ..expressions.algebra import block

            n = con.detRootN.n

            P = Problem()
            L = LowerTriangularVariable("__L", n)
            d = L.maindiag
            D = d.diag
            P.add_constraint(block([[con.detRootN.x, L], [L.T, D]]) >> 0)
            P.add_constraint(GeometricMean(d) >= con.lowerBound)
            return P

    def __init__(self, detRootN, lowerBound):
        """Construct a :class:`DetRootNConstraint`.

        :param ~picos.expressions.DetRootN detRootN:
            Constrained expression.
        :param ~picos.expressions.AffineExpression lowerBound:
            Lower bound on the expression.
        """
        from ..expressions import AffineExpression, DetRootN

        assert isinstance(detRootN, DetRootN)
        assert isinstance(lowerBound, AffineExpression)
        assert len(lowerBound) == 1

        self.detRootN   = detRootN
        self.lowerBound = lowerBound

        super(DetRootNConstraint, self).__init__(detRootN._typeStr)

    Subtype = namedtuple("Subtype", ("diag", "complex"))

    def _subtype(self):
        return self.Subtype(self.detRootN.n, self.detRootN.x.complex)

    @classmethod
    def _cost(cls, subtype):
        n = subtype.diag

        if subtype.complex:
            return n**2 + 1
        else:
            return n*(n + 1)//2 + 1

    def _expression_names(self):
        yield "detRootN"
        yield "lowerBound"

    def _str(self):
        return glyphs.ge(self.detRootN.string, self.lowerBound.string)

    def _get_slack(self):
        return self.detRootN.safe_value - self.lowerBound.safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
