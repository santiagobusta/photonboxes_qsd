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

"""Implementation of :class:`SumExponentialsConstraint`."""

import math
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class SumExponentialsConstraint(Constraint):
    """Upper bound on a sum of exponentials."""

    class ConicConversion(ConstraintConversion):
        """Sum of exponentials to exponential cone constraint conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ..expressions import RealVariable
            from . import AffineConstraint, ExpConeConstraint

            n = subtype.argdim

            yield ("var", RealVariable.make_var_type(dim=n, bnd=0), 1)
            yield ("con", AffineConstraint.make_type(dim=1, eq=False), 1)
            yield ("con", ExpConeConstraint.make_type(), n)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..expressions import ExponentialCone
            from ..modeling import Problem

            x = con.numerator
            y = con.denominator
            n = con.theSum.n
            b = con.upperBound

            P = Problem()

            u = P.add_variable("__u", n)
            P.add_constraint((u | 1) <= b)

            for i in range(n):
                P.add_constraint((u[i] // y[i] // x[i]) << ExponentialCone())

            return P

    class LogSumExpConversion(ConstraintConversion):
        """Sum of exponentials to logarithm of the sum constraint conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from . import LogSumExpConstraint

            n = subtype.argdim

            if subtype.lse_representable:
                yield ("con", LogSumExpConstraint.make_type(argdim=n), 1)
            else:
                # HACK: Return the input constraint type.
                # TODO: Handle partial subtype support differently, e.g. by
                #       introducing ConstraintConversion.supports.
                yield ("con", SumExponentialsConstraint.make_type(*subtype), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..expressions import LogSumExp
            from ..modeling import Problem

            x = con.numerator
            b = con.upperBound

            P = Problem()

            if con.lse_representable:
                P.add_constraint(LogSumExp(x) <= math.log(b.value))
            else:
                # HACK: See predict.
                P.add_constraint(con)

            return P

    def __init__(self, theSum, upperBound):
        """Construct a :class:`SumExponentialsConstraint`.

        :param ~picos.expressions.SumExponentials theSum:
            Constrained expression.
        :param ~picos.expressions.AffineExpression upperBound:
            Upper bound on the expression.
        """
        from ..expressions import AffineExpression, SumExponentials

        assert isinstance(theSum, SumExponentials)
        assert isinstance(upperBound, AffineExpression)
        assert len(upperBound) == 1

        self.theSum     = theSum
        self.upperBound = upperBound

        super(SumExponentialsConstraint, self).__init__(theSum._typeStr)

    @property
    def numerator(self):
        """The :math:`x` of the sum."""
        return self.theSum.x

    @cached_property
    def denominator(self):
        """The :math:`y` of the sum, or :math:`1`."""
        if self.theSum.y is None:
            from ..expressions import AffineExpression
            return AffineExpression.from_constant(1, self.theSum.x.shape)
        else:
            return self.theSum.y

    @property
    def lse_representable(self):
        """Whether this can be converted to a logarithmic constraint."""
        if self.theSum.y is not None:
            return False

        if not self.upperBound.constant:
            return False

        if self.upperBound.value < 0:
            return False

        return True

    Subtype = namedtuple("Subtype", ("argdim", "lse_representable"))

    def _subtype(self):
        return self.Subtype(self.theSum.n, self.lse_representable)

    @classmethod
    def _cost(cls, subtype):
        # NOTE: Twice the argument dimension due to the denominator.
        return 2*subtype.argdim + 1

    def _expression_names(self):
        yield "theSum"
        yield "upperBound"

    def _str(self):
        return glyphs.le(self.theSum.string, self.upperBound.string)

    def _get_slack(self):
        return self.upperBound.safe_value - self.theSum.safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
