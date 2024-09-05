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

"""Implements :class:`BallUncertainNormConstraint`."""

import operator
from collections import namedtuple

from ... import glyphs
from ...apidoc import api_end, api_start
from ..constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class BallUncertainNormConstraint(Constraint):
    """An (uncertain) upper bound on a norm with unit ball uncertainty."""

    class RobustConversion(ConstraintConversion):
        """Robust counterpart conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ...expressions import AffineExpression, RealVariable
            from .. import AffineConstraint, LMIConstraint

            if subtype.bound_universe_subtype:  # uncertain bound
                X = subtype.bound_universe_subtype
                x_dim = X.param_dim

                K = X.cone_type
                D = X.dual_cone_type
                K_dim = K.subtype.dim

                v = AffineExpression.make_type(
                    shape=(K_dim, 1), constant=False, nonneg=False)

                yield ("var", RealVariable.make_var_type(dim=1, bnd=0), 1)
                yield ("var", RealVariable.make_var_type(dim=K_dim, bnd=0), 1)
                yield ("con", AffineConstraint.make_type(dim=1, eq=False), 1)
                yield ("con", AffineConstraint.make_type(dim=x_dim, eq=True),
                    2 if X.has_B else 1)
                yield ("con", v.predict(operator.__lshift__, D), 1)

            k = subtype.dim
            p = subtype.norm_universe_subtype.param_dim

            yield ("var", RealVariable.make_var_type(dim=1, bnd=1), 1)
            yield ("con", LMIConstraint.make_type(diag=(k + p + 1)), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`.

            Conversion recipe and variable names based on the book
            *Robust Optimization* (Ben-Tal, El Ghaoui, Nemirovski, 2009).
            """
            from ...expressions import (
                block, Constant, ConicPerturbationSet, RealVariable)
            from ...modeling import Problem

            problem = Problem()

            # a = A(η)y + b(η) is the normed expression, e = η its perturbation.
            a = con.ne.vec
            e = con.ne.perturbation

            # Lz is the uncertain and g = Aⁿy + bⁿ the certain part of a.
            L, g = a.factor_out(e)

            # Define t = τ depending on whether the upper bound is uncertain.
            if con.ub.certain:
                t = con.ub
            else:
                # b = cᵀ(χ)y + d(χ) is the upper bound, x = χ its perturbation.
                b = con.ub
                x = con.ub.perturbation

                # sx = σᵀ(y)χ is the uncertain, d = δ(y) the certain part of b.
                s, d = b.factor_out(x)

                X = x.universe
                assert isinstance(X, ConicPerturbationSet)
                P, Q, p, K = X.A, X.B, X.c, X.K

                t = RealVariable("__t")
                v = RealVariable("__v", K.subtype.dim)

                problem.add_constraint(t + p.T*v <= d)
                problem.add_constraint(P.T*v == s.T)
                if Q is not None:
                    problem.add_constraint(Q.T*v == 0)
                problem.add_constraint(v << K.dual_cone)

            # Define l = λ.
            l = RealVariable("__{}".format(glyphs.lambda_()), lower=0)

            k, p = len(a), e.dim
            Ik = Constant("I", "I", (k, k))
            Ip = Constant("I", "I", (p, p))
            Op = Constant("0",  0,  (p, 1))

            M = block([[t*Ik, L,    g   ],   # noqa
                       [L.T,  l*Ip, Op  ],   # noqa
                       [g.T,  Op.T, t-l ]])  # noqa

            problem.add_constraint(M >> 0)

            return problem

    def __init__(self, norm, upper_bound):
        """Construct a :class:`BallUncertainNormConstraint`.

        :param norm:
            Uncertain norm that is bounded from above.
        :type norm:
            ~picos.expressions.UncertainNorm

        :param upper_bound:
            (Uncertain) upper bound on the norm.
        :type upper_bound:
            ~picos.expressions.AffineExpression or
            ~picos.expressions.UncertainAffineExpression
        """
        from ...expressions import AffineExpression
        from ...expressions.uncertain import (ConicPerturbationSet,
            UncertainAffineExpression, UncertainNorm, UnitBallPerturbationSet)

        assert isinstance(norm, UncertainNorm)
        assert isinstance(norm.x.universe, UnitBallPerturbationSet)
        assert isinstance(upper_bound,
            (AffineExpression, UncertainAffineExpression))
        assert upper_bound.scalar
        if upper_bound.uncertain:
            assert isinstance(upper_bound.universe, ConicPerturbationSet)
            assert norm.perturbation is not upper_bound.perturbation

        self.norm = norm
        self.ub = upper_bound

        super(BallUncertainNormConstraint, self).__init__("Ball-Uncertain Norm")

    @property
    def ne(self):
        """The uncertain affine expression under the norm."""
        return self.norm.x

    Subtype = namedtuple("Subtype", (
        "dim", "norm_universe_subtype", "bound_universe_subtype"))

    def _subtype(self):
        return self.Subtype(len(self.ne), self.ne.universe.subtype,
            self.ub.universe.subtype if self.ub.uncertain else None)

    @classmethod
    def _cost(cls, subtype):
        return float("inf")

    def _expression_names(self):
        yield "norm"
        yield "ub"

    def _str(self):
        if self.ub.uncertain:
            # Perturbations are required to differ.
            params = glyphs.comma(self.norm.perturbation, self.ub.perturbation)
        else:
            params = self.norm.perturbation

        return glyphs.forall(
            glyphs.le(self.norm.string, self.ub.string), params)

    def _get_size(self):
        return (1, 1)

    def _get_slack(self):
        if self.ub.certain:
            ub_value = self.ub.safe_value
        else:
            ub_value = self.ub.worst_case_value(direction="min")

        return ub_value - self.norm.worst_case_value(direction="max")


# --------------------------------------
__all__ = api_end(_API_START, globals())
