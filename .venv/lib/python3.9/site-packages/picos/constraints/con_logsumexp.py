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

"""Implementation of :class:`LogSumExpConstraint`."""

from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class LogSumExpConstraint(Constraint):
    """Upper bound on a logarithm of a sum of exponentials."""

    class ExpConeConversion(ConstraintConversion):
        """Bound on a log-sum-exp to exponential cone constraint conversion."""

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

            x = con.lse.x
            n = con.lse.n
            b = con.ub

            P = Problem()

            u = P.add_variable("__u", n)
            P.add_constraint((u | 1) <= 1)

            for i in range(n):
                P.add_constraint((u[i] // 1 // (x[i] - b)) << ExponentialCone())

            return P

        @classmethod
        def dual(cls, auxVarPrimals, auxConDuals, options):
            """Implement :meth:`~.constraint.ConstraintConversion.dual`."""
            # TODO: Verify that this is the dual.
            return auxConDuals[0]

    def __init__(self, lse, upperBound):
        """Construct a :class:`LogSumExpConstraint`.

        :param ~picos.expressions.LogSumExp lse:
            Constrained expression.
        :param ~picos.expressions.AffineExpression upperBound:
            Upper bound on the expression.
        """
        from ..expressions import AffineExpression, LogSumExp

        assert isinstance(lse, LogSumExp)
        assert isinstance(upperBound, AffineExpression)
        assert len(upperBound) == 1

        self.lse = lse
        self.ub  = upperBound

        super(LogSumExpConstraint, self).__init__(
            lse._typeStr if isinstance(lse, LogSumExp) else "LSE")

    @property
    def exponents(self):
        """The affine exponents of the bounded log-sum-exp expression."""
        return self.lse.x

    @cached_property
    def le0(self):
        """The :class:`~.exp_logsumexp.LogSumExp` posed to be at most zero."""
        from ..expressions import LogSumExp

        if self.ub.is0:
            return self.lse
        else:
            return LogSumExp(self.lse.x - self.ub)

    Subtype = namedtuple("Subtype", ("argdim",))

    def _subtype(self):
        return self.Subtype(self.lse.n)

    @classmethod
    def _cost(cls, subtype):
        return subtype.argdim + 1

    def _expression_names(self):
        yield "lse"
        yield "ub"

    def _str(self):
        return glyphs.le(self.lse.string, self.ub.string)

    def _get_size(self):
        return (1, 1)

    def _get_slack(self):
        return self.ub.safe_value - self.lse.safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
