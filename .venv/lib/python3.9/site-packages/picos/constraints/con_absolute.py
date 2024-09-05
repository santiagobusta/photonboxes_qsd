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

"""Implementation of :class:`AbsoluteValueConstraint`."""

from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class AbsoluteValueConstraint(Constraint):
    """Upper bound on an absolute value."""

    # TODO: Improve performance: Add two scalar constraints.
    class AffineConversion(ConstraintConversion):
        """Upper bound on an absolute value to affine inequality conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from . import AffineConstraint

            yield ("con", AffineConstraint.make_type(dim=2, eq=False), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..modeling import Problem

            P = Problem()
            P.add_constraint((con.signedScalar // -con.signedScalar)
                <= (con.upperBound // con.upperBound))

            return P

        @classmethod
        def dual(cls, auxVarPrimals, auxConDuals, options):
            """Implement :meth:`~.constraint.ConstraintConversion.dual`."""
            assert len(auxConDuals) == 1

            if auxConDuals[0] is None:
                return None
            else:
                return auxConDuals[0][1] - auxConDuals[0][0]

    def __init__(self, signedScalar, upperBound):
        """Construct an :class:`AbsoluteValueConstraint`.

        :param ~picos.expressions.AffineExpression signedScalar:
            A scalar expression.
        :param ~picos.expressions.AffineExpression upperBound:
            Upper bound on the expression.
        """
        from ..expressions import AffineExpression

        assert isinstance(signedScalar, AffineExpression)
        assert isinstance(upperBound, AffineExpression)
        assert len(signedScalar) == 1
        assert len(upperBound) == 1

        self.signedScalar = signedScalar
        self.upperBound   = upperBound

        super(AbsoluteValueConstraint, self).__init__("Absolute Value")

    Subtype = namedtuple("Subtype", ())

    def _subtype(self):
        return self.Subtype()

    @classmethod
    def _cost(cls, subtype):
        return 2

    def _expression_names(self):
        yield "signedScalar"
        yield "upperBound"

    def _str(self):
        return glyphs.le(
            glyphs.abs(self.signedScalar.string), self.upperBound.string)

    def _get_size(self):
        return (1, 1)

    def _get_slack(self):
        return self.upperBound.safe_value - abs(self.signedScalar.safe_value)


# --------------------------------------
__all__ = api_end(_API_START, globals())
