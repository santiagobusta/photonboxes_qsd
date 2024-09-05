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

"""Optimization problem solution representation."""

import warnings

from .. import glyphs
from ..apidoc import api_end, api_start

_API_START = api_start(globals())
# -------------------------------


# Solution status strings, as verified by PICOS.
VS_UNKNOWN           = "unverified"
"""PICOS failed to verify the solution."""

VS_DETACHED          = "detached"
"""The solution is not attached to a problem (it was given by the user)."""

VS_EMPTY             = "empty"
"""The solution is empty; there are neither primals nor duals."""

VS_DETACHED_EMPTY    = "detached empty"
"""The solution is both detached and empty."""

VS_OUTDATED          = "outdated"
"""The solution does not fit the problem formulation any more.

Variables or constraints were removed from the problem."""

VS_INCOMPLETE        = "incomplete"
"""The primal (dual) solution does not concern all variables (constraints)."""

VS_FEASIBLE          = "feasible"
"""The solution is primal feasible; there is no dual solution."""

VS_INFEASIBLE        = "infeasible"
"""The solution is primal infeasible; there is no dual solution."""

VS_PRIMAL_FEASIBLE   = "primal feasible"
"""The solution is primal feasible; a dual solution was not verified."""

VS_PRIMAL_INFEASIBLE = "primal infeasible"
"""The solution is primal infeasible; a dual solution was not verified."""


# Primal or dual solution (or search) status strings, as claimed by the solver.
SS_UNKNOWN           = "unknown"
"""The solver did not make a clear claim about the solution status."""

SS_EMPTY             = "empty"
"""The solver claims not to have produced a solution."""

SS_OPTIMAL           = "optimal"
"""The solution is optimal."""

SS_FEASIBLE          = "feasible"
"""The solution is feasible."""

SS_INFEASIBLE        = "infeasible"
"""No feasible solution exists.

In the case of a primal solution, the problem is infeasible. In the case of a
dual solution, the problem is unbounded.
"""

SS_PREMATURE         = "premature"
"""The search was prematurely terminated due to some limit."""

SS_FAILURE           = "failure"
"""The search was termined due to a solver failure."""


# Problem status strings, as claimed by the solver.
PS_UNKNOWN           = "unknown"
"""The solver did not make a clear claim about the problem status."""

PS_FEASIBLE          = "feasible"
"""The problem is primal (and dual) feasible and bounded."""

PS_INFEASIBLE        = "infeasible"
"""The problem is primal infeasible (and dual unbounded or infeasible)."""

PS_UNBOUNDED         = "unbounded"
"""The problem is primal unbounded (and dual infeasible)."""

PS_INF_OR_UNB        = "infeasible or unbounded"
"""The problem is primal infeasible or unbounded.

Being unbounded is usually infered from being dual infeasible."""

PS_UNSTABLE          = "unstable"
"""The problem was found numerically unstable or otherwise hard to handle."""

PS_ILLPOSED          = "illposed"
"""The problem was found to be in a state that is not amenable to solution."""


def _check_type(argument, *types):
    """Enforce the type of a method or function argument."""
    for type_ in types:
        if type_ is None:
            type_ = type(None)
        if isinstance(argument, type_):
            return

    raise TypeError("An argument is of type '{}' but must be instance of {}."
        .format(type(argument).__name__, " or ".join("'{}'".format(t.__name__)
            for t in types)))


# TODO: Make all public fields use snake_case, ensure backwards compatibility.
class Solution:
    """Assignment of primal and dual values to variables and constraints.

    Instances are usually returned by a solver (and thus bound to a
    :class:`problem <picos.Problem>` instance), but may be manually created by
    the user:

    >>> import picos
    >>> x = picos.RealVariable("x")
    >>> s = picos.Solution({x: 1}); s
    <detached primal solution from user>
    >>> s.apply()
    >>> x.value
    1.0

    If the solution was created by a solver (or attached to a problem via
    :func:`attach_to`), more information is available:

    >>> P = picos.Problem()
    >>> P.minimize = x
    >>> P += x >= 2
    >>> s = P.solve(solver = "cvxopt", duals = False); s
    <feasible primal solution (claimed optimal) from cvxopt>
    >>> "{:.2f} ms".format(1000.0 * s.searchTime) #doctest: +SKIP
    '0.83 ms'
    >>> P += x >= 3; s
    <infeasible primal solution (was feasible and claimed optimal) from cvxopt>
    """

    def __init__(self, primals, duals=None, problem=None, solver="user",
            primalStatus=SS_UNKNOWN, dualStatus=SS_UNKNOWN,
            problemStatus=PS_UNKNOWN, searchTime=0.0, info=None,
            vectorizedPrimals=False, reportedValue=None):
        """Create a solution to an optimization problem.

        :param dict(picos.expressions.BaseVariable, object) primals:
            A mapping of variables to their primal solution value.
        :param dict(picos.constraints.Constraint, object) duals:
            A mapping of constraints to their dual solution value.
        :param picos.Problem problem:
            The problem that was solved to create the solution. If ``None``,
            then the solution is "detached".
        :param str solver:
            The name of the solver that was used to create the solution.
        :param str primalStatus:
            The primal solution status as reported by the solver.
        :param str dualStatus:
            The dual solution status as reported by the solver.
        :param str problemStatus:
            The state of the problem as reported by the solver.
        :param float searchTime:
            Seconds that the solution process took.
        :param dict info:
            Additional solution (meta)data.
        :param bool vectorizedPrimals:
            Whether primal solution values are given with respect to the
            variable's special vectorization format as used by PICOS internally.
        :param float reportedValue:
            Objective value of the solution as reported by the solver.
        """
        from ..expressions import BaseVariable
        from ..constraints import Constraint
        from .problem      import Problem

        if primals is None:
            primals = {}

        if duals is None:
            duals = {}

        if info is None:
            info = {}

        # Be strict about the arguments as they are handed to the user.
        _check_type(primals, dict)
        _check_type(duals, dict)
        _check_type(problem, None, Problem)
        _check_type(solver, str)
        _check_type(primalStatus, str)
        _check_type(dualStatus, str)
        _check_type(problemStatus, str)
        _check_type(searchTime, float)
        _check_type(info, dict)
        _check_type(vectorizedPrimals, bool)
        _check_type(reportedValue, None, float)

        for variable, _ in primals.items():
            if not isinstance(variable, BaseVariable):
                raise TypeError("They keys in the primals argument of "
                    "Solution.__init__ must be variables.")

        for constraint, _ in duals.items():
            if not isinstance(constraint, Constraint):
                raise TypeError("They keys in the duals argument of "
                    "Solution.__init__ must be constraints.")

        # Derive a "claimed status" from the claimed primal and dual states.
        if primals and duals:
            if primalStatus == dualStatus:
                claimedStatus = primalStatus
            else:
                claimedStatus = "primal {} and dual {}".format(
                    primalStatus, dualStatus)
        elif primals:
            # Do not warn about correctingdualStatus, because the solver might
            # have produced primals but PICOS did not read them.
            dualStatus    = SS_EMPTY
            claimedStatus = primalStatus
        elif duals:
            # Do not warn about correcting primalStatus, because the solver
            # might have produced duals but PICOS did not read them.
            primalStatus  = SS_EMPTY
            claimedStatus = dualStatus
        else:
            primalStatus  = SS_EMPTY
            dualStatus    = SS_EMPTY
            claimedStatus = SS_EMPTY

        # Infeasible problem implies infeasible primal.
        if problemStatus == PS_INFEASIBLE \
        and primalStatus not in (SS_INFEASIBLE, SS_EMPTY):
            warnings.warn(
                "{} claims that a problem is infeasible but does not say the "
                "same about the nonempty primal solution. Correcting this.".
                format(solver), RuntimeWarning)
            primalStatus = SS_INFEASIBLE

        # Unbounded problem implies infeasible dual.
        if problemStatus == PS_UNBOUNDED \
        and dualStatus not in (SS_INFEASIBLE, SS_EMPTY):
            warnings.warn(
                "{} claims that a problem is unbounded but does not say that "
                "the nonempty dual solution is infeasible. Correcting this.".
                format(solver), RuntimeWarning)
            dualStatus = SS_INFEASIBLE

        # Optimal solution implies feasible problem.
        if claimedStatus == SS_OPTIMAL and problemStatus != PS_FEASIBLE:
            warnings.warn(
                "{} claims to have found an optimal solution but does not say "
                " that the problem is feasible. Correcting this."
                .format(solver), RuntimeWarning)
            problemStatus = PS_FEASIBLE

        self.problem       = problem
        """The problem that was solved to produce the solution."""

        self.solver        = solver
        """The solver that produced the solution."""

        self.searchTime    = searchTime
        """Time in seconds that the solution search took."""

        self.primals       = primals
        """The primal solution values returned by the solver."""

        self.duals         = duals
        """The dual solution values returned by the solver."""

        self.info          = info
        """Additional information provided by the solver."""

        self.lastStatus    = VS_UNKNOWN
        """The solution status as verified by PICOS when the solution was
        applied to the problem."""

        self.primalStatus  = primalStatus
        """The primal solution status as claimed by the solver."""

        self.dualStatus    = dualStatus
        """The dual solution status as claimed by the solver."""

        self.claimedStatus = claimedStatus
        """The primal and dual solution status as claimed by the solver."""

        self.problemStatus = problemStatus
        """The problem status as claimed by the solver."""

        self.vectorizedPrimals = vectorizedPrimals
        """Whether primal values refer to variables' special vectorizations."""

        self.reportedValue = reportedValue
        """The objective value of the solution as reported by the solver."""

    def _status_of_problem(self, problem):
        """Retrieve the problem's verified solution status.

        Requires that the solution has just been applied to the problem.
        """
        if not self.primals and not self.duals:
            return VS_EMPTY

        try:
            isFeasible = problem.check_current_value_feasibility()[0]
        except LookupError:
            return VS_INCOMPLETE
        except Exception:
            return VS_UNKNOWN

        if isFeasible:
            return VS_PRIMAL_FEASIBLE if self.duals else VS_FEASIBLE
        else:
            return VS_PRIMAL_INFEASIBLE if self.duals else VS_INFEASIBLE

    @property
    def status(self):
        """The current solution status as verified by PICOS.

        .. warning::

            Accessing this attribute is expensive for large problems as a copy
            of the problem needs to be created and valued. If you have just
            applied the solution to a :class:`problem <picos.Problem>`, query
            the solution's lastStatus attribute instead.
        """
        if not self.primals and not self.duals:
            if not self.problem:
                return VS_DETACHED_EMPTY
            else:
                return VS_EMPTY
        elif not self.problem:
            return VS_DETACHED
        elif not self.primals:
            return VS_UNKNOWN

        problemCopy = self.problem.copy()

        try:
            self.apply(toProblem=problemCopy)
        except RuntimeError:
            return VS_OUTDATED

        return self._status_of_problem(problemCopy)

    @property
    def value(self):
        """The objective value of the solution as computed by PICOS.

        .. warning::

            Accessing this attribute is expensive for large problems as a copy
            of the problem needs to be created and valued. If you have just
            applied the solution to a :class:`problem <picos.Problem>`, query
            that problem instead.
        """
        if not self.problem:
            raise RuntimeError(
                "Cannot compute the objective value of a detached solution. "
                "Use attach_to to assign the solution to a problem.")

        problemCopy = self.problem.copy()

        self.apply(toProblem=problemCopy)

        return problemCopy.value

    @property
    def reported_value(self):
        """The objective value as reported by the solver, or :obj:`None`."""
        return self.reportedValue

    def __str__(self):
        verifiedStatus = self.status
        lastStatus     = self.lastStatus
        claimedStatus  = self.claimedStatus
        problemStatus  = self.problemStatus

        if self.primals and self.duals:
            solutionType = "solution pair"
        elif self.primals:
            solutionType = "primal solution"
        elif self.duals:
            solutionType = "dual solution"
        else:
            solutionType = "solution"  # "(detached) empty solution"

        # Print the last status if it is known and differs from the current one.
        printLastStatus = lastStatus != VS_UNKNOWN and \
            verifiedStatus != lastStatus

        # Print the claimed status only if it differs from the initial verified
        # one is not implied by a problem status that will be printed.
        printClaimedStatus = \
            claimedStatus not in (verifiedStatus, SS_UNKNOWN) and \
            problemStatus not in (PS_INFEASIBLE, PS_UNBOUNDED)

        # Print the problem status only if it is interesting.
        printProblemStatus = \
            problemStatus not in (PS_UNKNOWN, PS_FEASIBLE)

        if printLastStatus and printClaimedStatus:
            unverifiedStatus = " (was {} and claimed {})".format(
                lastStatus, claimedStatus)
        elif printLastStatus:
            unverifiedStatus = " (was {})".format(lastStatus)
        elif printClaimedStatus:
            unverifiedStatus = " (claimed {})".format(claimedStatus)
        else:
            unverifiedStatus = ""

        if printProblemStatus:
            unverifiedStatus += \
                " for a problem claimed {}".format(problemStatus)

        return "{} {}{} from {}".format(verifiedStatus, solutionType,
            unverifiedStatus, self.solver)

    def __repr__(self):
        return glyphs.repr1(self.__str__())

    def apply(self, primals=True, duals=True, clearOnNone=True, toProblem=None,
              snapshotStatus=False):
        """Apply the solution to the involved variables and constraints.

        :param bool primals: Whether to apply the primal solution.
        :param bool duals: Whether to apply the dual solution.
        :param bool clearOnNone: Whether to clear the value of a variable or
            constraint if the solution has it set to None. This could happen in
            case of an error or shortcoming of the solver or PICOS.
        :param picos.Problem toProblem: If set to a copy of the problem that was
            used to produce the solution, will apply the solution to that copy's
            variables and constraints instead.
        :param bool snapshotStatus: Whether to update the lastStatus attribute
            with the new (verified) solution status. PICOS enables this whenever
            it applies a solution returned by a solver.
        """
        if toProblem:
            if primals:
                thePrimals = {}
                try:
                    for variable, primal in self.primals.items():
                        thePrimals[toProblem.variables[variable.name]] = primal
                except KeyError as error:
                    raise RuntimeError(
                        "Cannot apply solution to specified problem as not all "
                        "variables for which primal values exist were found.") \
                        from error

            if duals:
                theDuals = {}
                try:
                    for constraint, dual in self.duals.items():
                        theDuals[toProblem.constraints[constraint.id]] = dual
                except KeyError as error:
                    raise RuntimeError(
                        "Cannot apply solution to specified problem as not all "
                        "constraints for which dual values exist were found.") \
                        from error
        else:
            thePrimals = self.primals
            theDuals   = self.duals

        if primals:
            for variable, primal in thePrimals.items():
                if primal is None and not clearOnNone:
                    continue

                if self.vectorizedPrimals:
                    variable.internal_value = primal
                else:
                    variable.value = primal

        if duals:
            for constraint, dual in theDuals.items():
                if dual is None and not clearOnNone:
                    continue
                constraint.dual = dual

        if snapshotStatus:
            if toProblem:
                self.lastStatus = self._status_of_problem(toProblem)
            elif self.problem:
                self.lastStatus = self._status_of_problem(self.problem)
            else:  # detached solution
                self.lastStatus = self.status

        if toProblem:
            toProblem._last_solution = self
        elif self.problem:
            self.problem._last_solution = self

    def attach_to(self, problem, snapshotStatus=False):
        """Attach (or move) the solution to a problem.

        Only variables and constraints that exist on the problem (same name or
        ID, respectively) are kept.

        :param bool snapshotStatus: Whether to set the lastStatus attribute
            of the copy to match the new problem.
        """
        self.problem = problem

        # Find variables of same name in the problem and assign primals.
        oldPrimals, self.primals = self.primals, {}
        for variable, primal in oldPrimals.items():
            if variable.name in problem.variables:
                self.primals[problem.variables[variable.name]] = primal

        # Find constraints of same ID in the problem and assign duals.
        oldDuals, self.duals = self.duals, {}
        for constraint, dual in oldDuals.items():
            if constraint.id in problem.constraints:
                self.duals[problem.constraints[constraint.id]] = dual

        # Update the last (verified) status.
        if snapshotStatus:
            self.lastStatus = problem.status
        else:
            self.lastStatus = VS_UNKNOWN


# --------------------------------------
__all__ = api_end(_API_START, globals())
