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

"""Implementation of :class:`SquaredNormConstraint`."""

from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class SquaredNormConstraint(Constraint):
    """Upper bound on a squared Euclidean or Frobenius norm."""

    class ConicConversion(ConstraintConversion):
        """Upper bound on a squared norm to conic conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from . import (AbsoluteValueConstraint, RSOCConstraint,
                           SOCConstraint)

            if subtype.constant_bound:
                if subtype.argdim == 1:
                    yield ("con", AbsoluteValueConstraint.make_type(), 1)
                else:
                    yield ("con", SOCConstraint.make_type(subtype.argdim), 1)
            else:
                yield ("con", RSOCConstraint.make_type(subtype.argdim), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..expressions import AffineExpression
            from ..modeling import Problem
            from . import (AbsoluteValueConstraint, RSOCConstraint,
                           SOCConstraint)

            x = con.squaredNorm.fullroot
            y = con.upperBound

            P = Problem()

            if y.constant:
                value = y.value

                if value < 0:
                    # TODO: Reconsider whether infeasible constraints should
                    #       raise an exception during conversion.
                    raise ValueError("The constraint {} is infeasible as it "
                        "upper-bounds a squared norm by a negative constant."
                        .format(con))

                root = AffineExpression.from_constant(
                    value**0.5, (1, 1), glyphs.sqrt(y.string))

                if len(x) == 1:
                    P.add_constraint(AbsoluteValueConstraint(x, root))
                else:
                    P.add_constraint(SOCConstraint(x, root))
            else:
                one = AffineExpression.from_constant(1)
                P.add_constraint(RSOCConstraint(x, y, one))

            return P

    def __init__(self, squaredNorm, upperBound):
        """Construct a :class:`SquaredNormConstraint`.

        :param ~picos.expressions.SquaredNorm squaredNorm:
            The squared norm to bound from above.
        :param ~picos.expressions.AffineExpression upperBound:
            Upper bound on the squared norm.
        """
        from ..expressions import AffineExpression, SquaredNorm

        assert isinstance(squaredNorm, SquaredNorm)
        assert isinstance(upperBound, AffineExpression)
        assert len(upperBound) == 1

        self.squaredNorm = squaredNorm
        self.upperBound = upperBound

        super(SquaredNormConstraint, self).__init__(self.squaredNorm._typeStr)

    Subtype = namedtuple("Subtype", ("argdim", "constant_bound"))

    def _subtype(self):
        return self.Subtype(self.squaredNorm.argdim, self.upperBound.constant)

    @classmethod
    def _cost(cls, subtype):
        return subtype.argdim + 2  # RSOCC case.

    def _expression_names(self):
        yield "squaredNorm"
        yield "upperBound"

    def _str(self):
        return glyphs.le(self.squaredNorm.string, self.upperBound.string)

    def _get_size(self):
        return (1, 1)

    def _get_slack(self):
        return self.upperBound.value - self.squaredNorm.value


# --------------------------------------
__all__ = api_end(_API_START, globals())
