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

"""Implements :class:`ConicallyUncertainAffineConstraint`."""

import operator
from collections import namedtuple

from ... import glyphs
from ...apidoc import api_end, api_start
from ..constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class ConicallyUncertainAffineConstraint(Constraint):
    """A bound on an affine expression with conic uncertainty."""

    class RobustConversion(ConstraintConversion):
        """Robust counterpart conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ...expressions import AffineExpression, RealVariable
            from .. import AffineConstraint

            N = subtype.dim
            Z = subtype.universe_subtype

            K = Z.cone_type
            D = Z.dual_cone_type

            K_dim = K.subtype.dim
            z_dim = Z.param_dim

            y = AffineExpression.make_type(
                shape=(K_dim, 1), constant=False, nonneg=False)

            yield ("var", RealVariable.make_var_type(dim=K_dim, bnd=0), N)
            yield ("con", AffineConstraint.make_type(dim=1, eq=False), N)
            yield ("con", AffineConstraint.make_type(z_dim, eq=True),
                2*N if subtype.universe_subtype.has_B else N)
            yield ("con", y.predict(operator.__lshift__, D), N)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`.

            Conversion recipe and variable names based on the book
            *Robust Optimization* (Ben-Tal, El Ghaoui, Nemirovski, 2009).
            """
            from ...expressions import AffineExpression, RealVariable
            from ...modeling import Problem

            z = con.le0.perturbation
            Z = con.le0.universe
            P, Q, p, K = Z.A, Z.B, Z.c, Z.K

            problem = Problem()

            # Handle multidimensional constraints entry-wise.
            for i in range(len(con.le0)):
                scalar_le0 = con.le0[i]

                y = RealVariable("__y#{}".format(i), K.dim)

                # The certain linear part.
                a0Tx = AffineExpression("a0Tx", (1, 1), {
                    x: scalar_le0._linear_coefs[x]
                    for x in scalar_le0._linear_coefs if x is not z})

                # The certain constant part.
                b0 = AffineExpression(
                    "b0", (1, 1), {(): scalar_le0._constant_coef})

                aT = {}
                for (x, z), coef in scalar_le0._sorted_bilinear_coefs.items():
                    coef = coef[:, :]  # Make a copy of the row vector.
                    coef.size = (z.dim, x.dim)  # Devectorize it.
                    aT[x, z] = coef

                # The linear part for each scalar perturbation (v-stacked).
                a_Tx = AffineExpression("a_Tx", (z.dim, 1), {
                    x: aT[x, z] for (x, z) in aT})

                # The constant part for each scalar perturbation (v-stacked).
                b_ = AffineExpression("b_", (z.dim, 1),
                    {(): scalar_le0._linear_coefs[z].T}
                    if z in scalar_le0._linear_coefs else {})

                problem.add_constraint(p.T*y + a0Tx + b0 <= 0)
                problem.add_constraint(P.T*y + a_Tx + b_ == 0)
                problem.add_constraint(y << K.dual_cone)

                if Q is not None:
                    problem.add_constraint(Q.T*y == 0)

            return problem

    def __init__(self, le0):
        """Construct an :class:`ConicallyUncertainAffineConstraint`.

        :param ~picos.expressions.UncertainAffineExpression le0:
            Uncertain expression constrained to be at most zero.
        """
        from ...expressions import UncertainAffineExpression
        from ...expressions.uncertain.pert_conic import ConicPerturbationSet

        assert isinstance(le0, UncertainAffineExpression)
        assert isinstance(le0.universe, ConicPerturbationSet)

        self.le0 = le0

        super(ConicallyUncertainAffineConstraint, self).__init__(
            "Conically Uncertain Affine", printSize=True)

    Subtype = namedtuple("Subtype", ("dim", "universe_subtype"))

    def _subtype(self):
        return self.Subtype(len(self.le0), self.le0.universe.subtype)

    @classmethod
    def _cost(cls, subtype):
        return float("inf")

    def _expression_names(self):
        yield "le0"

    def _str(self):
        return glyphs.forall(
            glyphs.le(self.le0.string, 0), self.le0.perturbation)

    def _get_size(self):
        return self.le0.shape

    def _get_slack(self):
        return -self.le0.worst_case_value(direction="max")


# --------------------------------------
__all__ = api_end(_API_START, globals())
