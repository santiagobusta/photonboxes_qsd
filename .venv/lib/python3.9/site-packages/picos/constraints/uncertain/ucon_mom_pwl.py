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

"""Implements :class:`MomentAmbiguousExtremumAffineConstraint`."""

from collections import namedtuple

from ... import glyphs
from ...apidoc import api_end, api_start
from ...caching import cached_property
from ..constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class MomentAmbiguousExtremumAffineConstraint(Constraint):
    """A bound on a moment-ambiguous expected value of a piecewise function."""

    class DistributionallyRobustConversion(ConstraintConversion):
        """Distributionally robust counterpart conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ...expressions import RealVariable, SymmetricVariable
            from .. import AffineConstraint, LMIConstraint, SOCConstraint

            k = subtype.extremum_argnum
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
                yield ("var", RealVariable.make_var_type(dim=k, bnd=k), 1)

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

            yield ("con", LMIConstraint.make_type(diag=(m + 1)), k)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            # The conversion recipe is found in "Robust conic optimization in
            # Python" (Stahlberg 2020) and extends a result in "Distributionally
            # robust optimization under moment uncertainty with application to
            # data-driven problems" (Delage and Ye 2010).
            from ...expressions import (
                Constant, RealVariable, SecondOrderCone, SymmetricVariable)
            from ...expressions.algebra import block
            from ...expressions.data import cvxopt_principal_root
            from ...modeling import Problem

            # Load the constraint in maximum form.
            con = con.maximum_form
            assert isinstance(con, MomentAmbiguousExtremumAffineConstraint)
            assert con.relation == con.LE
            k = con.extremum.argnum
            K = range(k)

            # Load the ambiguity set.
            MAS = con.extremum.universe
            xi = MAS.parameter
            m = MAS.dim
            mu = MAS.nominal_mean
            Sigma = MAS.nominal_covariance
            alpha = MAS.alpha
            beta = MAS.beta
            S = MAS.sample_space

            # Load the uncertain extremum as max(h[i].T*xi + eta[i] for i in K).
            zero = Constant(0, shape=(1, m))
            hT, eta = zip(*(
                (zero, a) if a.certain else a.factor_out(xi)
                for a in con.extremum.expressions))
            h = tuple(hiT.T for hiT in hT)

            # Determime boundedness pattern.
            bounded_mean = alpha is not None
            bounded_covariance = beta is not None
            bounded_support = S is not None

            # Load the upper bound.
            omega = con.rhs

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
                l = RealVariable("__lambda", k, lower=0)
                l = [l[i] for i in K]  # Cache element access.
                inv_D = S.Ainv
                G = inv_D.T*inv_D
                d = S.c
                Gd = G*d
                dTGd_minus_one = d.T*Gd - 1

            if bounded_mean and bounded_covariance and bounded_support:
                # Default case.
                U = [l[i]*G + Q for i in K]
                V = [0.5*(q - h[i]) - l[i]*Gd for i in K]
                W = [l[i]*dTGd_minus_one - eta[i] + r for i in K]

                problem.add_constraint(
                    ((beta*Sigma + mu*mu.T) | Q) + mu.T*q + r + t <= omega)
                problem.add_constraint(
                    (t // (sqrt_alpha*sqrt_Sigma*(2*Q*mu + q)))
                    << SecondOrderCone())
            elif not bounded_mean and bounded_covariance and bounded_support:
                # Unbounded mean.
                U = [l[i]*G + Q for i in K]
                V = [-(Q*mu + 0.5*h[i] + l[i]*Gd) for i in K]
                W = [l[i]*dTGd_minus_one - eta[i] + r for i in K]

                problem.add_constraint(
                    ((beta*Sigma - mu*mu.T) | Q) + r <= omega)
            elif bounded_mean and not bounded_covariance and bounded_support:
                # Unbounded covariance.
                U = [l[i]*G for i in K]
                V = [0.5*(q - h[i]) - l[i]*Gd for i in K]
                W = [l[i]*dTGd_minus_one - eta[i] + r for i in K]

                problem.add_constraint(mu.T*q + r + t <= omega)
                problem.add_constraint(
                    (t // (sqrt_alpha*sqrt_Sigma*q)) << SecondOrderCone())
            elif bounded_mean and bounded_covariance and not bounded_support:
                # Unbounded support.
                U = [Q for i in K]
                V = [0.5*(q - h[i]) for i in K]
                W = [r - eta[i] for i in K]

                problem.add_constraint(
                    ((beta*Sigma + mu*mu.T) | Q) + mu.T*q + r + t <= omega)
                problem.add_constraint(
                    (t // (sqrt_alpha*sqrt_Sigma*(2*Q*mu + q)))
                    << SecondOrderCone())
            elif not (bounded_mean or bounded_covariance) and bounded_support:
                # Unbounded mean and covariance.
                U = [l[i]*G for i in K]
                V = [-(0.5*h[i] + l[i]*Gd) for i in K]
                W = [l[i]*dTGd_minus_one - eta[i] + omega for i in K]

                # TODO: There is also an SOCP representation (sketched below);
                #       use it once there is sufficient test coverage to support
                #       the special treatment.
                # z = [RealVariable("__z_{}".format(i), m) for i in K]
                # zeta = [RealVariable("__z_{}".format(i), 1) for i in K]
                # problem.add_list_of_constraints(
                #     [zeta[i] - d.T*inv_D.T*z[i] + eta[i] <= omega for i in K])
                # problem.add_list_of_constraints(
                #     [inv_D.T*z[i] + h[i] == 0 for i in K])
                # problem.add_list_of_constraints(
                #     [abs(z[i]) <= zeta[i] for i in K])
                # return problem
            else:
                assert False, "Unexpected unboundedness pattern."

            problem.add_list_of_constraints([
                block([[U[i],   V[i]],
                       [V[i].T, W[i]]]) >> 0
                for i in K])

            return problem

    def __init__(self, extremum, relation, rhs):
        """Construct a :class:`MomentAmbiguousExtremumAffineConstraint`.

        :param ~picos.expressions.RandomExtremumAffine extremum:
            Left hand side expression.
        :param str relation:
            Constraint relation symbol.
        :param ~picos.expressions.AffineExpression rhs:
            Right hand side expression.
        """
        from ...expressions import (AffineExpression, MomentAmbiguitySet,
            RandomMaximumAffine, RandomMinimumAffine)

        if relation == self.LE:
            assert isinstance(extremum, RandomMaximumAffine)
        else:
            assert isinstance(extremum, RandomMinimumAffine)
        assert isinstance(extremum.universe, MomentAmbiguitySet)
        assert isinstance(rhs, AffineExpression)
        assert rhs.scalar

        self.extremum = extremum
        self.relation = relation
        self.rhs = rhs

        super(MomentAmbiguousExtremumAffineConstraint, self).__init__(
            "Moment-ambiguous Piecewise Linear Expectation", printSize=True)

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
