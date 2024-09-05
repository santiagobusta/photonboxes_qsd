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

"""Implementation of :class:`KullbackLeiblerConstraint`."""

from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class KullbackLeiblerConstraint(Constraint):
    """Upper bound on a Kullback-Leibler divergence.

    This is the upper bound on a negative or relative entropy, both represented
    by :class:`~picos.expressions.NegativeEntropy`.
    """

    class ExpConeConversion(ConstraintConversion):
        """Kullback-Leibler to exponential cone constraint conversion."""

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
            n = con.divergence.n
            b = con.upperBound

            P = Problem()

            u = P.add_variable("__u", n)
            P.add_constraint((u | 1) >= -b)

            for i in range(n):
                P.add_constraint((y[i] // x[i] // u[i]) << ExponentialCone())

            return P

    def __init__(self, divergence, upperBound):
        """Construct a :class:`KullbackLeiblerConstraint`.

        :param ~picos.expressions.NegativeEntropy divergence:
            Constrained expression.
        :param ~picos.expressions.AffineExpression upperBound:
            Upper bound on the expression.
        """
        from ..expressions import AffineExpression, NegativeEntropy

        assert isinstance(divergence, NegativeEntropy)
        assert isinstance(upperBound, AffineExpression)
        assert len(upperBound) == 1

        self.divergence = divergence
        self.upperBound = upperBound

        super(KullbackLeiblerConstraint, self).__init__(divergence._typeStr)

    @property
    def numerator(self):
        """The :math:`x` of the divergence."""
        return self.divergence.x

    @cached_property
    def denominator(self):
        """The :math:`y` of the divergence, or :math:`1`."""
        from ..expressions import AffineExpression

        if self.divergence.y is None:
            return AffineExpression.from_constant(1, self.divergence.x.shape)
        else:
            return self.divergence.y

    Subtype = namedtuple("Subtype", ("argdim",))

    def _subtype(self):
        return self.Subtype(len(self.numerator))

    @classmethod
    def _cost(cls, subtype):
        # NOTE: Twice the argument dimension due to the denominator.
        return 2*subtype.argdim + 1

    def _expression_names(self):
        yield "divergence"
        yield "upperBound"

    def _str(self):
        return glyphs.le(self.divergence.string, self.upperBound.string)

    def _get_slack(self):
        return self.upperBound.safe_value - self.divergence.safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
