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

"""Rotated second order cone constraints."""

from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from .constraint import ConicConstraint

_API_START = api_start(globals())
# -------------------------------


class RSOCConstraint(ConicConstraint):
    """Rotated second order cone membership constraint."""

    def __init__(self, normedExpression, upperBoundFactor1,
            upperBoundFactor2=None, customString=None):
        """Construct a :class:`RSOCConstraint`.

        :param ~picos.expressions.AffineExpression normedExpression:
            Expression under the norm.
        :param ~picos.expressions.AffineExpression upperBoundFactor1:
            First of the two scalar factors that make the upper bound on the
            normed expression.
        :param ~picos.expressions.AffineExpression upperBoundFactor2:
            Second of the two scalar factors that make the upper bound on the
            normed expression.
        :param str customString:
            Optional string description.
        """
        from ..expressions import AffineExpression

        if upperBoundFactor2 is None:
            upperBoundFactor2 = AffineExpression.from_constant(1)

        assert isinstance(normedExpression,  AffineExpression)
        assert isinstance(upperBoundFactor1, AffineExpression)
        assert isinstance(upperBoundFactor2, AffineExpression)
        assert len(upperBoundFactor1) == 1
        assert len(upperBoundFactor2) == 1

        self.ne  = normedExpression
        self.ub1 = upperBoundFactor1
        self.ub2 = upperBoundFactor2

        super(RSOCConstraint, self).__init__(
            self._get_type_term(), customString, printSize=True)

    def _get_type_term(self):
        return "RSOC"

    @cached_property
    def conic_membership_form(self):
        """Implement for :class:`~.constraint.ConicConstraint`."""
        from ..expressions import RotatedSecondOrderCone
        return (self.ub1 // self.ub2 // self.ne.vec), \
            RotatedSecondOrderCone(dim=(len(self.ne) + 2))

    Subtype = namedtuple("Subtype", ("argdim",))

    def _subtype(self):
        return self.Subtype(len(self.ne))

    @classmethod
    def _cost(cls, subtype):
        return subtype.argdim + 2

    def _expression_names(self):
        yield "ne"
        yield "ub1"
        yield "ub2"

    def _str(self):
        a = glyphs.le(glyphs.squared(glyphs.norm(self.ne.string)),
            glyphs.clever_mul(self.ub1.string, self.ub2.string))

        if self.ub1.is1:
            b = glyphs.ge(self.ub2.string, 0)
        elif self.ub2.is1:
            b = glyphs.ge(self.ub1.string, 0)
        else:
            b = glyphs.ge(glyphs.comma(self.ub1.string, self.ub2.string), 0)

        return glyphs.and_(a, b)

    def _get_size(self):
        return (len(self.ne) + 2, 1)

    def _get_slack(self):
        return self.ub1.safe_value * self.ub2.safe_value \
            - (abs(self.ne)**2).safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
