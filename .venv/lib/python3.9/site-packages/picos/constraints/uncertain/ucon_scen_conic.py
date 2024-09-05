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

"""Implements :class:`ScenarioUncertainConicConstraint`."""

import operator
from collections import namedtuple

from ... import glyphs
from ...apidoc import api_end, api_start
from ..constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class ScenarioUncertainConicConstraint(Constraint):
    """Conic constraint with scenario uncertainty."""

    class RobustConversion(ConstraintConversion):
        """Robust counterpart conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ...expressions import AffineExpression

            n = subtype.dim
            k = subtype.scenario_count
            C = subtype.cone_type

            # HACK: It's possible that a member becomes constant for some
            #       realization of the uncertainty but we can't predict this.
            #       This is not a problem as long as the constraint outcome of
            #       claiming conic membership does not depend on whether the
            #       member is constant.
            a = AffineExpression.make_type(
                shape=(n, 1), constant=False, nonneg=False)

            yield ("con", a.predict(operator.__lshift__, C), k)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ...modeling import Problem

            A, b = con.element.vec.factor_out(con.element.perturbation)

            P = Problem()

            for s in con.element.universe.scenarios:
                P.add_constraint(A*s + b << con.cone)

            return P

    def __init__(self, element, cone):
        """Construct a :class:`ScenarioUncertainConicConstraint`.

        :param ~picos.expressions.UncertainAffineExpression element:
            Uncertain expression constrained to be in the cone.

        :param ~picos.expressions.Cone cone:
            The cone that the uncertain expression is constrained to.
        """
        from ...expressions import Cone, UncertainAffineExpression
        from ...expressions.uncertain.pert_scenario import (
            ScenarioPerturbationSet)

        assert isinstance(element, UncertainAffineExpression)
        assert isinstance(element.universe, ScenarioPerturbationSet)
        assert isinstance(cone, Cone)
        assert cone.dim is None or len(element) == cone.dim

        self.element = element
        self.cone = cone

        super(ScenarioUncertainConicConstraint, self).__init__(
            "Scenario Uncertain Conic", printSize=True)

    Subtype = namedtuple("Subtype", ("dim", "scenario_count", "cone_type"))

    def _subtype(self):
        return self.Subtype(
            dim=len(self.element),
            scenario_count=self.element.universe.scenarios.num,
            cone_type=self.cone.type)

    @classmethod
    def _cost(cls, subtype):
        return float("inf")

    def _expression_names(self):
        yield "element"
        yield "cone"

    def _str(self):
        return glyphs.forall(
            glyphs.element(self.element.string, self.cone.string),
            self.element.perturbation)

    def _get_size(self):
        return self.element.shape

    def _get_slack(self):
        from ...expressions import (NonnegativeOrthant, SecondOrderCone,
                                    PositiveSemidefiniteCone, RealVariable,
                                    RotatedSecondOrderCone)
        from ...expressions.data import cvxopt_hpsd

        if isinstance(self.cone, NonnegativeOrthant):
            return self.element.worst_case_value(direction="min")
        elif isinstance(self.cone, SecondOrderCone):
            ub = self.element[0]
            norm = abs(self.element[1:])

            if ub.certain:
                return ub.value - norm.worst_case_value(direction="max")
            else:
                # TODO: Use convex optimization to compute the slack.
                raise NotImplementedError("Computing the slack of a scenario-"
                    "uncertain second order conic constraint is not supported "
                    "if the first element of the cone member is uncertain.")
        elif isinstance(self.cone, RotatedSecondOrderCone):
            ub1 = self.element[0]
            ub2 = self.element[1]
            sqnorm = abs(self.element[2:])**2

            if ub1.certain and ub2.certain:
                ub1_value = ub1.value
                ub2_value = ub2.value

                ub_value = ub1_value*ub2_value
                slack = ub_value - sqnorm.worst_case_value(direction="max")

                if ub1_value < 0:
                    slack = min(ub1_value, slack)

                if ub2_value < 0:
                    slack = min(ub2_value, slack)

                return slack
            else:
                # TODO: Use convex optimization to compute the slack.
                raise NotImplementedError(
                    "Computing the slack of a scenario-uncertain rotated second"
                    " order conic constraint is not supported unless the first "
                    "two elements of the cone member are certain.")
        elif isinstance(self.cone, PositiveSemidefiniteCone):
            # Devectorize the cone element to a symmetric matrix A and replace
            # its perturbation parameter with a real variable x.
            x = RealVariable("x", self.element.perturbation.shape)
            A = self.element.desvec.replace_mutables(
                {self.element.perturbation: x})

            # Find the least-slack matrix S by scenario enumeration.
            S = None
            for s in self.element.universe.scenarios._cvxopt_vectors:
                x.value = s
                if S is None or cvxopt_hpsd(
                        S.safe_value_as_matrix - A.safe_value_as_matrix):
                    S = ~A

            # Vectorize the slack.
            return S.svec.safe_value
        else:
            # NOTE: This can be extended on a cone-by-cone basis if necessary.
            raise NotImplementedError("Computing the slack of a scenario-"
                "uncertain conic constraint is not supporeted for the cone {}."
                .format(self.cone.__class__.__name__))


# --------------------------------------
__all__ = api_end(_API_START, globals())
