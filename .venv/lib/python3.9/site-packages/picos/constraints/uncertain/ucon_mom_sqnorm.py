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

"""Implements :class:`MomentAmbiguousSquaredNormConstraint`."""

from collections import namedtuple

from ... import glyphs
from ...apidoc import api_end, api_start
from ..constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class MomentAmbiguousSquaredNormConstraint(Constraint):
    """A bound on a moment-ambiguous expected value of a squared norm."""

    class DistributionallyRobustConversion(ConstraintConversion):
        """Distributionally robust counterpart conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ...expressions import RealVariable, SymmetricVariable
            from .. import AffineConstraint, LMIConstraint, SOCConstraint

            k = subtype.sqnorm_argdim
            m = subtype.universe_subtype.dim
            bounded_mean = subtype.universe_subtype.bounded_mean
            bounded_covariance = subtype.universe_subtype.bounded_covariance
            bounded_support = subtype.universe_subtype.bounded_support

            if bounded_mean or bounded_covariance:
                yield ("var", RealVariable.make_var_type(dim=1, bnd=0), 1)  # r

            if bounded_mean:
                yield ("var", RealVariable.make_var_type(dim=m, bnd=0), 1)  # q
                yield ("var", RealVariable.make_var_type(dim=1, bnd=0), 1)  # t

            if bounded_covariance:
                # Q >> 0
                yield ("var", SymmetricVariable.make_var_type(
                    dim=(m * (m + 1)) // 2, bnd=0), 1)
                yield ("con", LMIConstraint.make_type(diag=m), 1)

            if bounded_support:
                # l >= 0
                yield ("var", RealVariable.make_var_type(dim=1, bnd=1), 1)

            if bounded_mean and bounded_covariance and bounded_support:
                yield ("con", AffineConstraint.make_type(dim=1, eq=False), 1)
                yield ("con", SOCConstraint.make_type(argdim=m), 1)
            elif not bounded_mean and bounded_covariance and bounded_support:
                yield ("con", AffineConstraint.make_type(dim=1, eq=False), 1)
            elif bounded_mean and not bounded_covariance and bounded_support:
                yield ("con", AffineConstraint.make_type(dim=1, eq=False), 1)
                yield ("con", SOCConstraint.make_type(argdim=m), 1)
            elif bounded_mean and bounded_covariance and not bounded_support:
                yield ("con", AffineConstraint.make_type(dim=1, eq=False), 1)
                yield ("con", SOCConstraint.make_type(argdim=m), 1)
            elif not (bounded_mean or bounded_covariance) and bounded_support:
                pass
            else:
                assert False, "Unexpected unboundedness pattern."

            yield ("con", LMIConstraint.make_type(diag=(k + m + 1)), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            # The conversion recipe is found in "Robust conic optimization in
            # Python" (Stahlberg 2020) and extends a result in "Models and
            # algorithms for distributionally robust least squares problems"
            # (Mehrotra and Zhang 2014).
            from ...expressions import (Constant, RealVariable, SecondOrderCone,
                                        SymmetricVariable)
            from ...expressions.algebra import block
            from ...expressions.data import cvxopt_principal_root
            from ...modeling import Problem

            # Load the uncertain suqared norm.
            a = con.sqnorm.x
            B, b = a.factor_out(a.perturbation)

            # Load the ambiguity set.
            MAS = con.sqnorm.universe
            m = MAS.dim
            mu = MAS.nominal_mean
            Sigma = MAS.nominal_covariance
            alpha = MAS.alpha
            beta = MAS.beta
            S = MAS.sample_space

            # Determime boundedness pattern.
            bounded_mean = alpha is not None
            bounded_covariance = beta is not None
            bounded_support = S is not None

            # Load the upper bound.
            omega = con.ub

            problem = Problem()

            if bounded_mean or bounded_covariance:
                r = RealVariable("__r")

            if bounded_mean:
                q = RealVariable("__q", m)
                t = RealVariable("__t")
                sqrt_alpha = alpha**0.5
                sqrt_Sigma = Constant(glyphs.sqrt(Sigma.string),
                    cvxopt_principal_root(Sigma.value_as_matrix))

            if bounded_covariance:
                Q = SymmetricVariable("__Q", m)
                problem.add_constraint(Q >> 0)

            if bounded_support:
                l = RealVariable("__lambda", lower=0)
                inv_D = S.Ainv
                G = inv_D.T*inv_D
                d = S.c

            if bounded_mean and bounded_covariance and bounded_support:
                # Default case.
                U = l*G + Q
                V = 0.5*q - l*G*d
                W = l*(d.T*G*d - 1) + r

                problem.add_constraint(
                    ((beta*Sigma + mu*mu.T) | Q) + mu.T*q + r + t <= omega)
                problem.add_constraint(
                    (t // (sqrt_alpha*sqrt_Sigma*(2*Q*mu + q)))
                    << SecondOrderCone())
            elif not bounded_mean and bounded_covariance and bounded_support:
                # Unbounded mean.
                U = l*G + Q
                V = -Q*mu - l*G*d
                W = l*(d.T*G*d - 1) + r

                problem.add_constraint(
                    ((beta*Sigma - mu*mu.T) | Q) + r <= omega)
            elif bounded_mean and not bounded_covariance and bounded_support:
                # Unbounded covariance.
                U = l*G
                V = 0.5*q - l*G*d
                W = l*(d.T*G*d - 1) + r

                problem.add_constraint(mu.T*q + r + t <= omega)
                problem.add_constraint(
                    (t // (sqrt_alpha*sqrt_Sigma*q)) << SecondOrderCone())
            elif bounded_mean and bounded_covariance and not bounded_support:
                # Unbounded support.
                U = Q
                V = 0.5*q
                W = r

                problem.add_constraint(
                    ((beta*Sigma + mu*mu.T) | Q) + mu.T*q + r + t <= omega)
                problem.add_constraint(
                    (t // (sqrt_alpha*sqrt_Sigma*(2*Q*mu + q)))
                    << SecondOrderCone())
            elif not (bounded_mean or bounded_covariance) and bounded_support:
                # Unbounded mean and covariance.
                U = l*G
                V = -l*G*d
                W = l*(d.T*G*d - 1) + omega
            else:
                assert False, "Unexpected unboundedness pattern."

            problem.add_constraint(
                block([["I", B,   b],
                       [B.T, U,   V],
                       [b.T, V.T, W]]) >> 0
            )

            return problem

    def __init__(self, sqnorm, upper_bound):
        """Construct a :class:`MomentAmbiguousSquaredNormConstraint`.

        :param ~picos.expressions.UncertainSquaredNorm sqnorm:
            Uncertain squared norm to upper bound the expectation of.

        :param ~picos.expressions.AffineExpression upper_bound:
            Upper bound on the expected value.
        """
        from ...expressions import AffineExpression, UncertainSquaredNorm
        from ...expressions.uncertain.pert_moment import MomentAmbiguitySet

        assert isinstance(sqnorm, UncertainSquaredNorm)
        assert isinstance(sqnorm.universe, MomentAmbiguitySet)
        assert isinstance(upper_bound, AffineExpression)
        assert upper_bound.scalar

        self.sqnorm = sqnorm
        self.ub = upper_bound

        super(MomentAmbiguousSquaredNormConstraint, self).__init__(
            "Moment-ambiguous Expected Squared Norm", printSize=True)

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
