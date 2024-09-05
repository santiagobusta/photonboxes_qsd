# ------------------------------------------------------------------------------
# Copyright (C) 2019 Maximilian Stahlberg
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

"""Implementation of :class:`EpigraphReformulation`."""

import operator

from ..apidoc import api_end, api_start
from ..expressions import AffineExpression, PredictedFailure, RealVariable
from .reformulation import Reformulation

_API_START = api_start(globals())
# -------------------------------


class EpigraphReformulation(Reformulation):
    """Epigraph reformulation."""

    NEW_OBJECTIVE = AffineExpression.make_type(
        shape=(1, 1), constant=False, nonneg=False)

    @classmethod
    def supports(cls, footprint):
        """Implement :meth:`~.reformulation.Reformulation.supports`."""
        if footprint.objective.clstype is AffineExpression:
            return False

        dir, obj = footprint.direction, footprint.objective
        relation = operator.__ge__ if dir == "max" else operator.__le__
        try:
            obj.predict(relation, cls.NEW_OBJECTIVE)
        except PredictedFailure:
            return False

        return True

    @classmethod
    def predict(cls, footprint):
        """Implement :meth:`~.reformulation.Reformulation.predict`."""
        dir, obj = footprint.direction, footprint.objective
        relation = operator.__ge__ if dir == "max" else operator.__le__
        constraint = obj.predict(relation, cls.NEW_OBJECTIVE)

        return footprint.updated((
            ("obj", footprint.NONE),
            ("obj", cls.NEW_OBJECTIVE, None),
            ("var", RealVariable.make_var_type(dim=1, bnd=0), 1),
            ("con", constraint, 1)))

    def forward(self):
        """Implement :meth:`~.reformulation.Reformulation.forward`."""
        self.output = self.input.clone(copyOptions=False)

        direction, objective = self.output.objective

        self.t = self.output.add_variable("__t")
        self.C = self.output.add_constraint(
            objective <= self.t if direction == "min" else objective >= self.t)
        self.output.set_objective(direction, self.t)

    def update(self):
        """Implement :meth:`~.reformulation.Reformulation.update`."""
        if self._objective_has_changed():
            # Remove the old auxilary constraint.
            self.output.remove_constraint(self.C.id)

            # Add a new one, using the existing variable.
            newDir, newObj = self.input.objective
            self.C = self.output.add_constraint(
                newObj <= self.t if newDir == "min" else newObj >= self.t)

        self._pass_updated_vars()
        self._pass_updated_cons()
        self._pass_updated_options()

    def backward(self, solution):
        """Implement :meth:`~.reformulation.Reformulation.backward`."""
        if self.t in solution.primals:
            solution.primals.pop(self.t)

        if self.C in solution.duals:
            solution.duals.pop(self.C)

        return solution


# --------------------------------------
__all__ = api_end(_API_START, globals())
