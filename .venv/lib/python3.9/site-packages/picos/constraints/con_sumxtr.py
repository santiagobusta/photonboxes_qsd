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

"""Implementation of :class:`SumExtremesConstraint`."""

from collections import namedtuple

import cvxopt as cvx

from .. import glyphs
from ..apidoc import api_end, api_start
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class SumExtremesConstraint(Constraint):
    """Bound on a sum over extreme (eigen)values."""

    class Conversion(ConstraintConversion):
        """Sum over extremes to LMI/affine constraint conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ..expressions import (HermitianVariable, RealVariable,
                                       SymmetricVariable)
            from . import AffineConstraint, ComplexLMIConstraint, LMIConstraint

            nm, k, eigenvalues, complex = subtype

            # Determine matrix variable dimension.
            if eigenvalues:
                nFloat = nm**0.5
                n = int(nFloat)
                assert n == nFloat
                d = n**2 if complex else n*(n + 1) // 2
            else:
                n = nm

            # Validate k.
            assert k > 0 and k <= n

            # Define shorthands for better formatting below.
            RV, AC = RealVariable, AffineConstraint
            LMI = ComplexLMIConstraint if complex else LMIConstraint
            MV = HermitianVariable if complex else SymmetricVariable

            if eigenvalues:
                if k == 1:
                    yield ("con", LMI.make_type(diag=n), 1)
                elif k == n:
                    # NOTE: Refinement prevents this case from happening.
                    yield ("con", AC.make_type(dim=1, eq=False), 1)
                else:
                    yield ("var", RV.make_var_type(dim=1, bnd=0), 1)
                    yield ("var", MV.make_var_type(dim=d, bnd=0), 1)
                    yield ("con", LMI.make_type(diag=n), 2)
                    yield ("con", AC.make_type(dim=1, eq=False), 1)
            else:
                if k == 1:
                    yield ("con", AC.make_type(dim=n, eq=False), 1)
                elif k == n:
                    # NOTE: Refinement prevents this case from happening.
                    yield ("con", AC.make_type(dim=1, eq=False), 1)
                else:
                    yield ("var", RV.make_var_type(dim=1, bnd=0), 1)
                    yield ("var", RV.make_var_type(dim=n, bnd=n), 1)
                    yield ("con", AC.make_type(dim=n, eq=False), 1)
                    yield ("con", AC.make_type(dim=1, eq=False), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..expressions import (Constant, HermitianVariable,
                                       RealVariable, SymmetricVariable)
            from ..modeling import Problem

            theSum   = con.theSum
            relation = con.relation
            rhs      = con.rhs

            x = theSum.x
            k = theSum.k

            if theSum.eigenvalues:
                n = x.shape[0]
                I = Constant('I', cvx.spdiag([1.] * n))
            else:
                n = len(x)

            if x.complex:
                MatrixVariable = HermitianVariable
            else:
                MatrixVariable = SymmetricVariable

            P = Problem()

            if relation == Constraint.LE:
                if theSum.eigenvalues:
                    if k == 1:
                        P.add_constraint(x << rhs * I)
                    elif k == n:
                        # NOTE: Refinement prevents this case from happening.
                        P.add_constraint(("I" | x) <= rhs)
                    else:
                        s = RealVariable('s')
                        Z = MatrixVariable('Z', n)
                        P.add_constraint(Z >> 0)
                        P.add_constraint(x << Z + s * I)
                        P.add_constraint(rhs >= (I | Z) + (k * s))
                else:
                    if k == 1:
                        P.add_constraint(x <= rhs)
                    elif k == n:
                        P.add_constraint((1 | x) <= rhs)
                    else:
                        lbda = RealVariable('lambda')
                        mu = RealVariable('mu', x.shape, lower=0)
                        P.add_constraint(x <= lbda + mu)
                        P.add_constraint(k * lbda + (1 | mu) <= rhs)
            else:
                if theSum.eigenvalues:
                    if k == 1:
                        P.add_constraint(x >> rhs * I)
                    elif k == n:
                        # NOTE: Refinement prevents this case from happening.
                        P.add_constraint((I | x) <= rhs)
                    else:
                        s = RealVariable('s')
                        Z = MatrixVariable('Z', n)
                        P.add_constraint(Z >> 0)
                        P.add_constraint(-x << Z + s * I)
                        P.add_constraint(-rhs >= (I | Z) + (k * s))
                else:
                    if k == 1:
                        P.add_constraint(x >= rhs)
                    elif k == n:
                        P.add_constraint((1 | x) >= rhs)
                    else:
                        lbda = RealVariable('lambda')
                        mu = RealVariable('mu', x.shape, lower=0)
                        P.add_constraint(-x <= lbda + mu)
                        P.add_constraint(k * lbda + (1 | mu) <= -rhs)

            return P

    def __init__(self, theSum, relation, rhs):
        """Construct a :class:`SumExtremesConstraint`.

        :param ~picos.expressions.SumExtremes theSum:
            Left hand side expression.
        :param str relation:
            Constraint relation symbol.
        :param ~picos.expressions.AffineExpression rhs:
            Right hand side expression.
        """
        from ..expressions import AffineExpression, SumExtremes

        assert isinstance(theSum, SumExtremes)
        assert isinstance(rhs, AffineExpression)
        assert relation in self.LE + self.GE
        assert len(rhs) == 1

        self.theSum   = theSum
        self.relation = relation
        self.rhs      = rhs

        super(SumExtremesConstraint, self).__init__(theSum._typeStr)

    Subtype = namedtuple("Subtype", ("argdim", "k", "eigenvalues", "complex"))

    def _subtype(self):
        return self.Subtype(len(self.theSum.x), self.theSum.k,
            self.theSum.eigenvalues, self.theSum.x.complex)

    @classmethod
    def _cost(cls, subtype):
        nm, _, eigenvalues, _ = subtype

        if eigenvalues:
            nFloat = nm**0.5
            n = int(nFloat)
            assert n == nFloat
        else:
            n = nm

        return n + 1

    def _expression_names(self):
        yield "theSum"
        yield "rhs"

    def _str(self):
        if self.relation == self.LE:
            return glyphs.le(self.theSum.string, self.rhs.string)
        else:
            return glyphs.ge(self.theSum.string, self.rhs.string)

    def _get_slack(self):
        if self.relation == self.LE:
            return self.rhs.safe_value - self.theSum.safe_value
        else:
            return self.theSum.safe_value - self.rhs.safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
