# ------------------------------------------------------------------------------
# Copyright (C) 2012-2017, 2020 Guillaume Sagnol
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

"""Implementation of matrix norm constraints."""

import operator
from collections import namedtuple

import cvxopt as cvx

from .. import glyphs
from ..apidoc import api_end, api_start
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class MatrixNormConstraint(Constraint):
    """Upper bound on a matrix :math:`(p,q)`-norm."""

    class VectorNormConversion(ConstraintConversion):
        """Upper bound on a :math:`(p,q)`-norm constraint conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ..expressions import AffineExpression, RealVariable, Norm

            shape, pNum, pDen, qNum, qDen = subtype
            rows, cols = shape

            # HACK: Whether the bound is constant is irrelevant.
            bound = AffineExpression.make_type((1, 1), None, None)
            pNorm = Norm.make_type((rows, 1), pNum, pDen, pNum, pDen)
            qNorm = Norm.make_type((cols, 1), qNum, qDen, qNum, qDen)

            yield ("var", RealVariable.make_var_type(dim=cols, bnd=0), 1)
            yield ("con", pNorm.predict(operator.__le__, bound), cols)
            yield ("con", qNorm.predict(operator.__le__, bound), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..expressions import no_refinement, Norm, RealVariable
            from ..modeling import Problem

            norm = con.norm
            p, q = norm.p, norm.q
            cols = norm.x.size[1]

            P = Problem()

            u = RealVariable("__u", cols)

            # Bound the p-norm of every column.
            with no_refinement():
                for j in range(cols):
                    P.add_constraint(Norm(norm.x[:, j], p) <= u[j])

            # Bound the q-norm of the column norms.
            P.add_constraint(Norm(u, q) <= con.upperBound)

            return P

    def __init__(self, norm, upperBound):
        """Construct a :class:`MatrixNormConstraint`.

        :param ~picos.expressions.Norm norm:
            The norm.
        :param ~picos.expressions.AffineExpression upperBound:
            The scalar upper bound.
        """
        from ..expressions import AffineExpression, Norm

        assert isinstance(norm, Norm)
        assert isinstance(upperBound, AffineExpression)
        assert len(upperBound) == 1

        assert norm.qnum is not None or norm.pnum != norm.qnum, \
            "Won't create a (p,q)-norm constraint for a p-norm."

        self.norm       = norm
        self.upperBound = upperBound

        super(MatrixNormConstraint, self).__init__(norm._typeStr)

    Subtype = namedtuple("Subtype", ("shape", "pNum", "pDen", "qNum", "qDen"))

    @classmethod
    def _cost(cls, subtype):
        return subtype.shape[0] * subtype.shape[1] + 1

    def _subtype(self):
        return self.Subtype(self.norm.x.size, self.norm.pnum, self.norm.pden,
            self.norm.qnum, self.norm.qden)

    def _expression_names(self):
        yield "norm"
        yield "upperBound"

    def _str(self):
        return glyphs.le(self.norm.string, self.upperBound.string)

    def _get_slack(self):
        return self.upperBound.safe_value - self.norm.safe_value


class SpectralNormConstraint(Constraint):
    """Spectral norm of a matrix."""

    class Conversion(ConstraintConversion):
        """Spectral norm constraint conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from . import (ComplexLMIConstraint, LMIConstraint)
            m, n = subtype.shape

            if subtype.hermitian:
                if subtype.complex:
                    yield ("con", ComplexLMIConstraint.make_type(diag=n), 2)
                else:
                    yield ("con", LMIConstraint.make_type(diag=n), 2)
            else:
                if subtype.complex:
                    yield ("con", ComplexLMIConstraint.make_type(diag=n+m), 1)
                else:
                    yield ("con", LMIConstraint.make_type(diag=n+m), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..modeling import Problem
            from ..expressions import block, Constant

            x = con.norm.x
            m, n = x.shape
            t = con.upperBound

            In = Constant('I', cvx.spdiag([1.] * n))
            P = Problem()
            if x.hermitian:
                P.add_constraint(x << t*In)
                P.add_constraint(x >> -t * In)
            else:
                Im = Constant('I', cvx.spdiag([1.] * m))
                P.add_constraint(block([[t*Im, x], [x.H, t*In]]) >> 0)
            return P

    def __init__(self, norm, upperBound):
        """Construct a :class:`SpectralNormConstraint`.

        :param ~picos.expressions.SpectralNorm norm:
            Constrained spectral norm
        :param ~picos.expressions.AffineExpression upperBound:
            Upper bound on the expression.
        """
        from ..expressions import AffineExpression, SpectralNorm

        assert isinstance(norm, SpectralNorm)
        assert isinstance(upperBound, AffineExpression)
        assert len(upperBound) == 1

        self.norm       = norm
        self.upperBound = upperBound

        super(SpectralNormConstraint, self).__init__(norm._typeStr)

    Subtype = namedtuple("Subtype", ("shape", "complex", "hermitian"))

    def _subtype(self):
        x = self.norm.x
        return self.Subtype(x.shape, x.complex, x.hermitian)

    @classmethod
    def _cost(cls, subtype):
        m = subtype.shape[0]
        n = subtype.shape[1]

        if subtype.hermitian:
            if subtype.complex:
                return 2 * n**2 + 1
            else:
                return 2 * n*(n+1) // 2 + 1
        else:
            if subtype.complex:
                return (n + m) ** 2 + 1
            else:
                return (n + m) * (n + m + 1) // 2 + 1

    def _expression_names(self):
        yield "norm"
        yield "upperBound"

    def _str(self):
        return glyphs.le(self.norm.string, self.upperBound.string)

    def _get_slack(self):
        return self.upperBound.safe_value - self.norm.safe_value


class NuclearNormConstraint(Constraint):
    """Nuclear norm of a matrix."""

    class Conversion(ConstraintConversion):
        """Nuclear norm constraint conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ..expressions import SymmetricVariable, HermitianVariable
            from . import (ComplexLMIConstraint, LMIConstraint,
                           AffineConstraint, ComplexAffineConstraint)

            m, n = subtype.shape

            if subtype.complex:
                MatrixVariable = HermitianVariable
                dm, dn = m**2, n**2
            else:
                MatrixVariable = SymmetricVariable
                dm = (m * (m + 1)) // 2
                dn = (n * (n + 1)) // 2

            yield ("var", MatrixVariable.make_var_type(dim=dm, bnd=0), 1)
            yield ("var", MatrixVariable.make_var_type(dim=dn, bnd=0), 1)

            if subtype.hermitian:
                l = (n * (n + 1)) // 2  # Number of lower triangular elements.
                if subtype.complex:
                    yield ("con", ComplexAffineConstraint.make_type(dim=l), 1)
                    yield ("con", ComplexLMIConstraint.make_type(diag=n), 2)
                else:
                    yield ("con", AffineConstraint.make_type(dim=l, eq=True), 1)
                    yield ("con", LMIConstraint.make_type(diag=n), 2)
            else:
                if subtype.complex:
                    yield ("con", ComplexLMIConstraint.make_type(diag=n+m), 1)
                else:
                    yield ("con", LMIConstraint.make_type(diag=n+m), 1)

            yield ("con", AffineConstraint.make_type(dim=1, eq=False), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..modeling import Problem
            from ..expressions import SymmetricVariable, HermitianVariable
            from ..expressions.algebra import block, trace

            x = con.norm.x
            m, n = x.shape
            t = con.upperBound

            if con.norm._complex:
                Y = HermitianVariable("__Y", (m, m))
                Z = HermitianVariable("__Z", (n, n))
            else:
                Y = SymmetricVariable("__Y", (m, m))
                Z = SymmetricVariable("__Z", (n, n))

            P = Problem()
            if x.hermitian:
                P.add_constraint(x.trilvec == (Y - Z).trilvec)
                P.add_constraint(Y >> 0)
                P.add_constraint(Z >> 0)
                P.add_constraint(trace(Y) + trace(Z) <= t)
            else:
                P.add_constraint(block([[Y, x], [x.H, Z]]) >> 0)
                P.add_constraint(trace(Y) + trace(Z) <= 2 * t)
            return P

    def __init__(self, norm, upperBound):
        """Construct a :class:`NuclearNormConstraint`.

        :param ~picos.expressions.NuclearNorm norm:
            Constrained nuclear norm
        :param ~picos.expressions.AffineExpression upperBound:
            Upper bound on the expression.
        """
        from ..expressions import AffineExpression, NuclearNorm

        assert isinstance(norm, NuclearNorm)
        assert isinstance(upperBound, AffineExpression)
        assert len(upperBound) == 1

        self.norm       = norm
        self.upperBound = upperBound

        super(NuclearNormConstraint, self).__init__(norm._typeStr)

    Subtype = namedtuple("Subtype", ("shape", "complex", "hermitian"))

    def _subtype(self):
        x = self.norm.x
        return self.Subtype(x.shape, x.complex, x.hermitian)

    @classmethod
    def _cost(cls, subtype):
        m, n = subtype.shape

        if subtype.hermitian:
            if subtype.complex:
                return 2 * n ** 2 + 1
            else:
                return 2 * n * (n+1) // 2 + 1
        else:
            if subtype.complex:
                return (n + m) ** 2 + 1
            else:
                return (n + m) * (n + m + 1) // 2 + 1

    def _expression_names(self):
        yield "norm"
        yield "upperBound"

    def _str(self):
        return glyphs.le(self.norm.string, self.upperBound.string)

    def _get_slack(self):
        return self.upperBound.safe_value - self.norm.safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
