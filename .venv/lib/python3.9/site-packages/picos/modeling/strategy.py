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

"""Optimization problem solution strategy search."""

from collections import OrderedDict

from .. import glyphs
from ..apidoc import api_end, api_start
from ..containers import OrderedSet
from ..reforms import SORTED_REFORMS, ExtraOptions, Reformulation
from ..solvers import Solver, available_solvers, get_solver, get_solver_name
from .problem import Problem

_API_START = api_start(globals())
# -------------------------------


class NoStrategyFound(RuntimeError):
    """No solution strategy found.

    Raised when no viable combination of reformulations and solver to tackle the
    problem could be found.
    """

    pass


class Strategy:
    """Optimization problem solution strategy."""

    def __init__(self, problem, solver, *reforms):
        """Construct a :class:`Strategy`.

        :param ~picos.Problem problem:
            The first step in the solution pipeline; the problem to be solved.

        :param type solver:
            The last step in the solution pipeline; the solver class to be used.

        :param list(~picos.reforms.reformulation.Reformulation) reforms:
            Intermediate steps in the pipeline; reformulations to be applied.
            May not include :class:`~picos.reforms.ExtraOptions` which is
            automatically made the first reformulation.
        """
        if not isinstance(problem, Problem):
            raise TypeError("First argument must be a problem instance.")

        if not issubclass(solver, Solver):
            raise TypeError("Second argument must be a solver class.")

        if not all(issubclass(reform, Reformulation) for reform in reforms):
            raise TypeError("Extra arguments must be reformulation classes.")

        if ExtraOptions in reforms:
            raise TypeError("The ExtraOptions reformulation is implicitly part "
                "of any strategy and may not be added explicitly.")

        self.nodes = [problem, ExtraOptions(problem)]

        for reform in reforms:
            self.nodes.append(reform(self.nodes[-1]))

        self.nodes.append(solver(self.nodes[-1]))

    @property
    def problem(self):
        """The problem to be solved."""
        return self.nodes[0]

    @property
    def reforms(self):
        """All reformulations in use.

        This includes the implicit :class:`~picos.reforms.ExtraOptions`.
        """
        return self.nodes[1:-1]

    @property
    def solver(self):
        """The solver instance in use."""
        return self.nodes[-1]

    def __str__(self):
        return "\n".join(
            "{}. {}".format(num + 1, node.__class__.__name__)
            for num, node in enumerate(self.nodes[1:]))

    def __repr__(self):
        return glyphs.repr1("Solution strategy for {}".format(self.solver.name))

    def valid(self, **extra_options):
        """Whether the solution strategy can be executed.

        :param extra_options:
            A keyword parameter sequence of additional options (in addition to
            those of the problem) to assume used.
        """
        problem = self.nodes[0]
        solver = self.nodes[-1]

        # Determine the footprint with extra options set.
        footprint = problem.footprint.with_extra_options(**extra_options)
        options = footprint.options

        # Handle a conflicting solver selection.
        if options.ad_hoc_solver and options.ad_hoc_solver != solver:
            return False
        elif options.solver and options.solver != get_solver_name(solver):
            return False

        # Skip ExtraOptions but include the solver with the following.
        for node in self.nodes[2:]:
            if not node.supports(footprint):
                return False

            footprint = node.predict(footprint)

        return True

    def execute(self, **extra_options):
        """Execute the solution strategy.

        :param extra_options:
            A keyword parameter sequence of additional options (in addition to
            those of the problem) to use for this search.

        :returns:
             :class:`~picos.modeling.Solution` to the problem.
        """
        # Defer solving to the first reformulation, which is responsible for
        # applying the extra options (i.e. ExtraOptions).
        solution = self.nodes[1].execute(**extra_options)

        # Attach the solution to the root problem. Note that reformulations are
        # allowed to already do this within their 'backward' method, but for
        # performance reasons it is best to do this just once, here.
        if isinstance(solution, list):
            for s in solution:
                s.attach_to(self.nodes[0])
        else:
            solution.attach_to(self.nodes[0])

        return solution

    @classmethod
    def from_problem(cls, problem, **extra_options):
        """Create a solution strategy for the given problem.

        :param ~picos.Problem problem:
            The optimization problem to search a strategy for.

        :param extra_options:
            A keyword parameter sequence of additional options (in addition to
            those of the problem) to assume used.
        """
        # Determine the footprint with extra options set.
        footprint = problem.footprint.with_extra_options(**extra_options)
        options = footprint.options

        # Decide on solvers to consider.
        solvers = []
        if options.ad_hoc_solver:
            solvers.append(options.ad_hoc_solver)
        elif options.solver:
            solver = get_solver(options.solver)

            if solver.available():
                solvers.append(solver)
            else:
                raise RuntimeError(
                    "Selected solver {} is not available on the system."
                    .format(solver.get_via_name()))
        else:
            for solver_name in available_solvers():
                solver = get_solver(solver_name)
                solvers.append(solver)

        if not solvers:
            raise RuntimeError("Not even CVXOPT seems to be available. "
                "Did you blacklist all available solvers?")

        if len(solvers) == 1 and solvers[0].supports(footprint):
            if options.verbosity >= 2:
                print("{} supports the problem directly.".format(
                    solvers[0].get_via_name()))

            return cls(problem, solvers[0])

        if options.verbosity >= 2:
            print("Selected solvers:\n  {}".format(", ".join(
                solver.get_via_name() for solver in solvers)))

        paths = OrderedDict({footprint: tuple()})
        new_footprints = [footprint]

        while new_footprints:
            if options.max_footprints is not None \
            and len(paths) >= options.max_footprints:
                if options.verbosity >= 1:
                    print("Footprint limit reached ({}/{}).".format(
                        len(paths), options.max_footprints))
                break

            active_footprints = new_footprints
            new_footprints = []

            if options.verbosity >= 3:
                print("Active footprints:\n{}".format("\n".join("  ({}) {}"
                    .format(len(paths) - len(active_footprints) + i, f)
                    for i, f in enumerate(active_footprints))))

            for num, footprint in enumerate(active_footprints):
                if options.verbosity >= 3:
                    print("Prediction for ({}):".format(
                        len(paths) - len(active_footprints) + num))

                for step in SORTED_REFORMS:
                    if options.verbosity >= 3:
                        print("  {}:".format(step.__name__), end=" ")

                    # Don't apply the same reformulation multiple times.
                    if step in paths[footprint]:
                        if options.verbosity >= 3:
                            print("Already part of current path.")
                        continue

                    # Check if the reformulation applies.
                    if not step.supports(footprint):
                        if options.verbosity >= 3:
                            print("Not supported.")
                        continue

                    # Predict the reformulation outcome.
                    new_footprint = step.predict(footprint)

                    # Remember only the first (shortest) path to any footprint.
                    if new_footprint in paths:
                        if options.verbosity >= 3:
                            print("Resulting footprint already reached.")
                        continue

                    if options.verbosity >= 3:
                        print("Reached new footprint ({}).".format(len(paths)))

                    paths[new_footprint] = paths[footprint] + (step,)
                    new_footprints.append(new_footprint)

        # Sort footprints by cost. 'paths' being an ordered dict ensures a
        # deterministic order with respect to same-cost footprints (with shorter
        # reformulation pipelines taking precedence).
        footprints = sorted(paths, key=(lambda f: f.cost))

        if options.verbosity >= 2:
            print("Reachable footprints:\n{}".format("\n".join(
                "  ({}) [{:.4f}] {}".format(i, f.cost, f)
                for i, f in enumerate(footprints))))

        strategies = []
        costs = []

        if options.verbosity >= 2:
            print("Solvable footprints:")

        for num, footprint in enumerate(footprints):
            if not solvers:
                break

            for solver in tuple(solvers):
                if solver.supports(footprint):
                    cost = footprint.cost
                    penalty = solver.penalty(options)
                    total_cost = cost + penalty

                    if options.verbosity >= 2:
                        print("  {} supports ({}) at cost {:.2f} + {:.2f} = "
                            "{:.2f}.".format(solver.get_via_name(), num, cost,
                            penalty, total_cost))

                    strategies.append(cls(problem, solver, *paths[footprint]))
                    costs.append(total_cost)

                    solvers.remove(solver)

        if not strategies:
            if options.verbosity >= 2:
                print("  None found.")

            synopsis = "No problem reformulation strategy found.\nSelected " \
                "reasons for discarding reachable problem formulations:"

            solver_reasons = {}

            for solver in solvers:
                reasons = OrderedSet()
                for footprint in paths:  # Footprints in original order.
                    supported, reason = solver.supports(footprint, explain=True)
                    assert not supported
                    reasons.add(reason)
                reasons = tuple(reasons)

                solver_reasons.setdefault(reasons, [])
                solver_reasons[reasons].append(solver)

            for reasons, unsupported_solvers in solver_reasons.items():
                if len(unsupported_solvers) == 1:
                    synopsis += "\n  {} does not support:".format(
                        unsupported_solvers.pop().get_via_name())
                else:
                    names = tuple(s.get_via_name() for s in unsupported_solvers)
                    synopsis += "\n  {} and {} do not support:".format(
                        ", ".join(names[:-1]), names[-1])

                synopsis += "\n    - ".join(("",) + reasons)

            raise NoStrategyFound(synopsis)

        return strategies[costs.index(min(costs))]


# --------------------------------------
__all__ = api_end(_API_START, globals())
