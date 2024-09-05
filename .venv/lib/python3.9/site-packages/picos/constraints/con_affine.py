# ------------------------------------------------------------------------------
# Copyright (C) 2018-2022 Maximilian Stahlberg
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

"""Affine constraint types."""

from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from .constraint import ConicConstraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class AffineConstraint(ConicConstraint):
    """An equality or inequality between two affine expressions."""

    def __init__(self, lhs, relation, rhs, customString=None):
        """Construct an :class:`AffineConstraint`.

        :param ~picos.expressions.AffineExpression lhs:
            Left hand side expression.
        :param str relation:
            Constraint relation symbol.
        :param ~picos.expressions.AffineExpression rhs:
            Right hand side expression.
        :param str customString:
            Optional string description.
        """
        from ..expressions import AffineExpression

        assert isinstance(lhs, AffineExpression)
        assert isinstance(rhs, AffineExpression)
        assert relation in self.LE + self.GE + self.EQ
        assert lhs.size == rhs.size

        self.lhs      = lhs
        self.rhs      = rhs
        self.relation = relation

        super(AffineConstraint, self).__init__(
            "Affine", customString, printSize=True)

    @cached_property
    def conic_membership_form(self):
        """Implement for :class:`~.constraint.ConicConstraint`."""
        from ..expressions import NonnegativeOrthant, ZeroSpace

        element = self.ge0.vec
        dim = len(element)

        if self.relation == self.EQ:
            return element, ZeroSpace(dim=dim)
        else:
            return element, NonnegativeOrthant(dim=dim)

    Subtype = namedtuple("Subtype", ("dim", "eq"))

    def _subtype(self):
        return self.Subtype(len(self.lhs), self.relation == self.EQ)

    @property
    def smaller(self):
        """Smaller-or-equal side of the constraint.

        The smaller-or-equal side expression in case of an inequality, otherwise
        the left hand side.
        """
        return self.rhs if self.relation == self.GE else self.lhs

    @property
    def greater(self):
        """Greater-or-equal side of the constraint.

        The greater-or-equal side expression in case of an inequality, otherwise
        the right hand side.
        """
        return self.lhs if self.relation == self.GE else self.rhs

    @cached_property
    def lmr(self):
        """Left hand side minus right hand side."""
        return self.lhs - self.rhs

    @cached_property
    def rml(self):
        """Right hand side minus left hand side."""
        return self.rhs - self.lhs

    @property
    def le0(self):
        """Expression constrained to be lower than or equal to zero.

        The expression posed to be less than or equal to zero in case of an
        inequality, otherwise the left hand side minus the right hand side.
        """
        if self.relation == self.GE:
            return self.rml
        else:
            return self.lmr

    @property
    def ge0(self):
        """Expression constrained to be greater than or equal to zero.

        The expression posed to be greater than or equal to zero in case of an
        inequality, otherwise the left hand side minus the right hand side.
        """
        if self.relation == self.LE:
            return self.rml
        else:
            return self.lmr

    @classmethod
    def _cost(cls, subtype):
        return subtype.dim

    def _expression_names(self):
        yield "lhs"
        yield "rhs"

    def _str(self):
        if self.relation == self.LE:
            return glyphs.le(self.lhs.string, self.rhs.string)
        elif self.relation == self.GE:
            return glyphs.ge(self.lhs.string, self.rhs.string)
        else:
            return glyphs.eq(self.lhs.string, self.rhs.string)

    def _get_size(self):
        return self.lhs.size

    def _get_slack(self):
        if self.relation == self.LE:
            delta = self.rml.safe_value
        else:
            delta = self.lmr.safe_value

        return -abs(delta) if self.relation == self.EQ else delta

    def bounded_linear_form(self):
        """Bounded linear form of the constraint.

        Separates the constraint into a linear function on the left hand side
        and a constant bound on the right hand side.

        :returns: A pair ``(linear, bound)`` where ``linear`` is a pure linear
            expression and ``bound`` is a constant expression.
        """
        linear = self.lmr
        bound  = -linear.cst
        linear = linear + bound

        return (linear, bound)


class ComplexAffineConstraint(ConicConstraint):
    """An equality between affine expressions, at least one being complex."""

    class RealConversion(ConstraintConversion):
        """Complex affine equality to real affine equality conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            yield ("con",
                AffineConstraint.make_type(dim=2*subtype.dim, eq=True), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..modeling import Problem

            P = Problem()
            P.add_constraint((con.lhs.real // con.lhs.imag)
                == (con.rhs.real // con.rhs.imag))

            return P

        @classmethod
        def dual(cls, auxVarPrimals, auxConDuals, options):
            """Implement :meth:`~.constraint.ConstraintConversion.dual`."""
            assert len(auxConDuals) == 1

            auxConDual = auxConDuals[0]
            if auxConDual is None:
                return None
            else:
                n = auxConDual.size[0] // 2
                return auxConDual[:n, :] + 1j*auxConDual[n:, :]

    def __init__(self, lhs, rhs, customString=None):
        """Construct a :class:`ComplexAffineConstraint`.

        :param ~picos.expressions.AffineExpression lhs:
            Left hand side expression.
        :param ~picos.expressions.AffineExpression rhs:
            Right hand side expression.
        :param str customString:
            Optional string description.
        """
        from ..expressions import ComplexAffineExpression

        assert isinstance(lhs, ComplexAffineExpression)
        assert isinstance(rhs, ComplexAffineExpression)
        assert lhs.size == rhs.size

        self.lhs = lhs
        self.rhs = rhs

        super(ComplexAffineConstraint, self).__init__(
            "Complex Equality", customString, printSize=True)

    @cached_property
    def conic_membership_form(self):
        """Implement for :class:`~.constraint.ConicConstraint`."""
        from ..expressions import ZeroSpace
        return self.lhs - self.rhs, ZeroSpace()

    Subtype = namedtuple("Subtype", ("dim",))

    def _subtype(self):
        return self.Subtype(len(self.lhs))

    @classmethod
    def _cost(cls, subtype):
        return 2*subtype.dim

    def _expression_names(self):
        yield "lhs"
        yield "rhs"

    def _str(self):
        return glyphs.eq(self.lhs.string, self.rhs.string)

    def _get_size(self):
        return self.lhs.size

    def _get_slack(self):
        return -abs(self.lhs.safe_value - self.rhs.safe_value)


# --------------------------------------
__all__ = api_end(_API_START, globals())
