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

"""Implements :class:`WassersteinAmbiguousExtremumAffineConstraint`."""

from collections import namedtuple

from ... import glyphs
from ...apidoc import api_end, api_start
from ...caching import cached_property
from ..constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class WassersteinAmbiguousExtremumAffineConstraint(Constraint):
    """A bound on a W_1-ambiguous expected value of a piecewise function."""

    class DistributionallyRobustConversion(ConstraintConversion):
        """Distributionally robust counterpart conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ...expressions import RealVariable
            from .. import AffineConstraint, SOCConstraint

            k = subtype.extremum_argnum
            m = subtype.universe_subtype.sample_dim
            N = subtype.universe_subtype.sample_num

            yield ("var", RealVariable.make_var_type(dim=1, bnd=0), 1)  # gamma
            yield ("var", RealVariable.make_var_type(dim=N, bnd=0), 1)  # s

            yield ("con", AffineConstraint.make_type(dim=1, eq=False), 1)
            yield ("con", AffineConstraint.make_type(dim=k, eq=False), N)
            yield ("con", SOCConstraint.make_type(argdim=m), k)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            # The conversion recipe is found in "Robust conic optimization in
            # Python" (Stahlberg 2020) and is an application of a result in
            # "Data-driven distributionally robust optimization using the
            # Wasserstein metric: Performance guarantees and tractable
            # reformulations" (Esfahani Mohajerin and Kuhn 2018).
            from ...expressions import (
                Constant, RandomMaximumAffine, RealVariable, SecondOrderCone)
            from ...expressions.algebra import block
            from ...modeling import Problem

            # Load the constraint in maximum form.
            con = con.maximum_form
            assert isinstance(con.extremum, RandomMaximumAffine)
            assert con.relation == con.LE
            k = con.extremum.argnum
            K = range(k)

            # Load the ambiguity set.
            WAS = con.extremum.universe
            xi = WAS.parameter
            S = WAS.samples
            m = S.dim
            N = S.num
            w = WAS.weights
            eps = WAS.eps

            # Load the uncertain extremum as max(h[i].T*xi + eta[i] for i in K).
            zero = Constant(0, shape=(1, m))
            hT, eta = zip(*(
                (zero, a) if a.certain else a.factor_out(xi)
                for a in con.extremum.expressions))

            # Stack the h[i].T and eta[i] vertically and to allow N many
            # k-dimensional constraints as opposed to N*k scalar constraints.
            # TODO: Consider adding picos.algebra.hcat and picos.algebra.vcat.
            hTs = block([[hT[i]] for i in K])
            etas = block([[eta[i]] for i in K])

            # Load the upper bound.
            omega = con.rhs

            P = Problem()

            gamma = RealVariable("__gamma")
            s = RealVariable("__s", N)

            P.add_constraint(gamma*eps + w.T*s <= omega)

            for j in range(N):
                P.add_constraint(hTs*S[j] + etas <= s[j].dupvec(k))

            for i in K:
                P.add_constraint((gamma & hT[i]) << SecondOrderCone())

            return P

    def __init__(self, extremum, relation, rhs):
        """Construct a :class:`WassersteinAmbiguousExtremumAffineConstraint`.

        :param ~picos.expressions.RandomExtremumAffine extremum:
            Left hand side expression.
        :param str relation:
            Constraint relation symbol.
        :param ~picos.expressions.AffineExpression rhs:
            Right hand side expression.
        """
        from ...expressions import (
            AffineExpression, RandomMaximumAffine, RandomMinimumAffine)

        if relation == self.LE:
            assert isinstance(extremum, RandomMaximumAffine)
        else:
            assert isinstance(extremum, RandomMinimumAffine)
        assert isinstance(rhs, AffineExpression)
        assert rhs.scalar

        self.extremum = extremum
        self.relation = relation
        self.rhs = rhs

        super(WassersteinAmbiguousExtremumAffineConstraint, self).__init__(
            "Wasserstein-ambiguous Piecewise Linear Expectation",
            printSize=True)

    @cached_property
    def maximum_form(self):
        """The constraint posed as an upper bound on an expected maximum."""
        if self.relation == self.LE:
            return self
        else:
            return self.__class__(-self.extremum, self.LE, -self.rhs)

    Subtype = namedtuple("Subtype", (
        "extremum_argnum",
        "universe_subtype"))

    def _subtype(self):
        return self.Subtype(
            self.extremum.argnum,
            self.extremum.universe.subtype)

    @classmethod
    def _cost(cls, subtype):
        return float("inf")

    def _expression_names(self):
        yield "extremum"
        yield "rhs"

    def _str(self):
        if self.relation == self.LE:
            return glyphs.le(
                self.extremum.worst_case_string("max"), self.rhs.string)
        else:
            return glyphs.ge(
                self.extremum.worst_case_string("min"), self.rhs.string)

    def _get_size(self):
        return (1, 1)

    def _get_slack(self):
        if self.relation == self.LE:
            return self.rhs.safe_value - self.extremum.worst_case_value("max")
        else:
            return self.extremum.worst_case_value("min") - self.rhs.safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
