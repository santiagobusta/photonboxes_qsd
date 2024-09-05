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

"""Implements :class:`WassersteinAmbiguousSquaredNormConstraint`."""

from collections import namedtuple

from ... import glyphs
from ...apidoc import api_end, api_start
from ..constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class WassersteinAmbiguousSquaredNormConstraint(Constraint):
    """A bound on a Wasserstein-ambiguous expected value of a squared norm."""

    class DistributionallyRobustConversion(ConstraintConversion):
        """Distributionally robust counterpart conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ...expressions import RealVariable, SymmetricVariable
            from .. import AffineConstraint, LMIConstraint

            k = subtype.sqnorm_argdim
            m = subtype.universe_subtype.sample_dim
            N = subtype.universe_subtype.sample_num

            yield ("var", RealVariable.make_var_type(dim=1, bnd=1), 1)  # gamma
            yield ("var", RealVariable.make_var_type(dim=N, bnd=0), 1)  # s
            yield ("var", SymmetricVariable.make_var_type(              # U
                dim=(m * (m + 1)) // 2, bnd=0), 1)
            yield ("var", RealVariable.make_var_type(dim=m, bnd=0), 1)  # u
            yield ("var", RealVariable.make_var_type(dim=1, bnd=0), 1)  # mu

            yield ("con", AffineConstraint.make_type(dim=1, eq=False), 1)
            yield ("con", LMIConstraint.make_type(diag=(k + m + 1)), 1)
            yield ("con", LMIConstraint.make_type(diag=(m + 1)), N)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            # The recipe is found in "Robust conic optimization in
            # Python" (Stahlberg 2020) and extends a result in "Wasserstein
            # distributionally robust optimization: Theory and applications in
            # machine learning" (Kuhn, Esfahani, Nguyen and Shafieezadeh-Abadeh
            # 2019).
            from ...expressions import RealVariable, SymmetricVariable
            from ...expressions.algebra import block
            from ...modeling import Problem

            problem = Problem()

            # Load the uncertain suqared norm.
            a = con.sqnorm.x
            B, b = a.factor_out(a.perturbation)

            # Load the ambiguity set.
            WAS = con.sqnorm.universe
            S = WAS.samples
            m = S.dim
            N = S.num
            w = WAS.weights
            eps = WAS.eps

            # Load the upper bound.
            omega = con.ub

            # Introduce auxiliary variables.
            gamma = RealVariable("__gamma", lower=0)
            s = RealVariable("__s", N)
            U = SymmetricVariable("__U", m)
            u = RealVariable("__u", m)
            mu = RealVariable("__mu")

            # Compute redundant terms that appear in constraints.
            h1 = gamma.dupdiag(m) - U
            h2 = tuple(gamma*S[i] + u for i in range(N))
            h3 = tuple(gamma*abs(S[i])**2 + s[i] - mu for i in range(N))

            # Add constraints.
            problem.add_constraint(
                gamma*eps**2 + w.T*s <= omega)
            problem.add_constraint(
                block([["I", B,   b],
                       [B.T, U,   u],
                       [b.T, u.T, mu]]) >> 0)
            problem.add_list_of_constraints([
                block([[h1,      h2[i]],
                       [h2[i].T, h3[i]]]) >> 0
                for i in range(N)])

            return problem

    def __init__(self, sqnorm, upper_bound):
        """Construct a :class:`WassersteinAmbiguousSquaredNormConstraint`.

        :param ~picos.expressions.UncertainSquaredNorm sqnorm:
            Uncertain squared norm to upper bound the expectation of.

        :param ~picos.expressions.AffineExpression upper_bound:
            Upper bound on the expected value.
        """
        from ...expressions import AffineExpression, UncertainSquaredNorm
        from ...expressions.uncertain.pert_wasserstein import (
            WassersteinAmbiguitySet)

        assert isinstance(sqnorm, UncertainSquaredNorm)
        assert isinstance(sqnorm.universe, WassersteinAmbiguitySet)
        assert sqnorm.universe.p == 2
        assert isinstance(upper_bound, AffineExpression)
        assert upper_bound.scalar

        self.sqnorm = sqnorm
        self.ub = upper_bound

        super(WassersteinAmbiguousSquaredNormConstraint, self).__init__(
            "Wasserstein-ambiguous Expected Squared Norm", printSize=True)

    Subtype = namedtuple("Subtype", ("sqnorm_argdim", "universe_subtype"))

    def _subtype(self):
        return self.Subtype(len(self.sqnorm.x), self.sqnorm.universe.subtype)

    @classmethod
    def _cost(cls, subtype):
        return float("inf")

    def _expression_names(self):
        yield "sqnorm"
        yield "ub"

    def _str(self):
        return glyphs.le(self.sqnorm.worst_case_string("max"), self.ub.string)

    def _get_size(self):
        return (1, 1)

    def _get_slack(self):
        return self.ub.value - self.sqnorm.worst_case_value(direction="max")


# --------------------------------------
__all__ = api_end(_API_START, globals())
