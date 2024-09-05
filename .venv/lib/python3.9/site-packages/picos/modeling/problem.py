# ------------------------------------------------------------------------------
# Copyright (C) 2012-2017 Guillaume Sagnol
# Copyright (C) 2017-2020 Maximilian Stahlberg
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

"""Implementation of :class:`Problem`."""

import copy as pycopy
import re
import time
from collections import OrderedDict
from functools import lru_cache
from textwrap import TextWrapper
from types import MappingProxyType

import cvxopt as cvx
import numpy as np

from .. import constraints, expressions, glyphs, settings
from ..apidoc import api_end, api_start
from ..expressions.data import cvx2np
from ..expressions.uncertain import IntractableWorstCase
from ..expressions.variables import BaseVariable
from ..formatting import natsorted, parameterized_string, picos_box
from ..legacy import deprecated, map_legacy_options, throw_deprecation_warning
from ..solvers import Solver, get_solver
from ..valuable import Valuable
from .file_out import write
from .footprint import Footprint, Specification
from .objective import Objective
from .options import Options
from .solution import SS_OPTIMAL, Solution

_API_START = api_start(globals())
# -------------------------------


class SolutionFailure(RuntimeError):
    """Solving the problem failed."""

    def __init__(self, code, message):
        """Construct a :exc:`SolutionFailure`.

        :param int code:
            Status code, as defined in :meth:`Problem.solve`.

        :param str message:
            Text description of the failure.
        """
        #: Status code, as defined in :meth:`Problem.solve`.
        self.code = code

        #: Text description of the failure.
        self.message = message

    def __str__(self):
        return "Code {}: {}".format(self.code, self.message)


class Problem(Valuable):
    """PICOS' representation of an optimization problem.

    :Example:

    >>> from picos import Problem, RealVariable
    >>> X = RealVariable("X", (2,2), lower = 0)
    >>> P = Problem("Example")
    >>> P.maximize = X.tr
    >>> C = X.sum <= 10
    >>> P += C, X[0,0] == 1
    >>> print(P)
    Example (Linear Program)
      maximize tr(X)
      over
        2×2 real variable X (bounded below)
      subject to
        ∑(X) ≤ 10
        X[0,0] = 1
    >>> # PICOS will select a suitable solver if you don't specify one.
    >>> solution = P.solve(solver = "cvxopt")
    >>> solution.claimedStatus
    'optimal'
    >>> solution.searchTime #doctest: +SKIP
    0.002137422561645508
    >>> round(P, 1)
    10.0
    >>> print(X) #doctest: +SKIP
    [ 1.00e+00  4.89e-10]
    [ 4.89e-10  9.00e+00]
    >>> round(C.dual, 1)
    1.0
    """

    #: The specification for problems returned by :meth:`conic_form`.
    CONIC_FORM = Specification(
        objectives=[expressions.AffineExpression],
        constraints=[C for C in
            (getattr(constraints, Cname) for Cname in constraints.__all__)
            if issubclass(C, constraints.ConicConstraint)
            and C is not constraints.ConicConstraint])

    # --------------------------------------------------------------------------
    # Initialization and reset methods.
    # --------------------------------------------------------------------------

    def __init__(
            self, name=None, *, copyOptions=None, useOptions=None,
            **extra_options
    ):
        """Create an empty problem and optionally set initial solver options.

        :param str name:
            A name or title to give to the optimization problem.

        :param copyOptions:
            An :class:`Options <picos.Options>` object to copy instead of using
            the default options.

        :param useOptions: An :class:`Options <picos.Options>` object to use
            (without making a copy) instead of using the default options.

        :param extra_options:
            A sequence of additional solver options to apply on top of the
            default options or those given by ``copyOptions`` or ``useOptions``.
        """
        if name and not isinstance(name, str):
            raise TypeError(
                "The first positional argument denotes the name of the problem "
                "and must be a string.")

        if copyOptions and useOptions:
            raise ValueError(
                "Can only copy or use existing solver options, not both.")

        extra_options = map_legacy_options(**extra_options)

        if copyOptions:
            self._options = copyOptions.copy()
            self._options.update(**extra_options)
        elif useOptions:
            self._options = useOptions
            self._options.update(**extra_options)
        else:
            self._options = Options(**extra_options)

        #: Explicit name for the problem.
        self._name = name

        #: The optimization objective.
        self._objective = Objective()

        #: Maps constraint IDs to constraints.
        self._constraints = OrderedDict()

        #: Contains lists of constraints added together, all in order.
        self._con_groups = []

        #: Maps mutables to number of occurences in objective or constraints.
        self._mtb_count = {}

        #: Maps mutable names to mutables.
        self._mutables = OrderedDict()

        #: Maps variable names to variables.
        self._variables = OrderedDict()

        #: Maps parameter names to parameters.
        self._parameters = OrderedDict()

        #: Current solution strategy.
        self._strategy = None

        #: The last :class:`Solution` applied to the problem.
        self._last_solution = None  # Set by Solution.apply.

    def _reset_mutable_registry(self):
        self._mtb_count.clear()
        self._mutables.clear()
        self._variables.clear()
        self._parameters.clear()

    def reset(self, resetOptions=False):
        """Reset the problem instance to its initial empty state.

        :param bool resetOptions:
            Whether also solver options should be reset to their default values.
        """
        # Reset options if requested.
        if resetOptions:
            self._options.reset()

        # Reset objective to "find an assignment".
        del self.objective

        # Reset constraint registry.
        self._constraints.clear()
        self._con_groups.clear()

        # Reset mutable registry.
        self._reset_mutable_registry()

        # Reset strategy and solution data.
        self._strategy = None
        self._last_solution = None

    # --------------------------------------------------------------------------
    # Properties.
    # --------------------------------------------------------------------------

    @property
    def name(self):
        """Name or title of the problem."""
        return self._name

    @name.setter
    def name(self, value):
        if value and not isinstance(value, str):
            raise TypeError("The problem name must be a string.")

        if not value:
            self._name = None
        else:
            self._name = value

    @name.deleter
    def name(self):
        self._name = None

    @property
    def mutables(self):
        """Maps names to variables and parameters in use by the problem.

        :returns:
            A read-only view to an :class:`~collections.OrderedDict`. The order
            is deterministic and depends on the order of operations performed on
            the :class:`Problem` instance as well as on the mutables' names.
        """
        return MappingProxyType(self._mutables)

    @property
    def variables(self):
        """Maps names to variables in use by the problem.

        :returns:
            See :attr:`mutables`.
        """
        return MappingProxyType(self._variables)

    @property
    def parameters(self):
        """Maps names to parameters in use by the problem.

        :returns:
            See :attr:`mutables`.
        """
        return MappingProxyType(self._parameters)

    @property
    def constraints(self):
        """Maps constraint IDs to constraints that are part of the problem.

        :returns:
            A read-only view to an :class:`~collections.OrderedDict`. The order
            is that in which constraints were added.
        """
        return MappingProxyType(self._constraints)

    @constraints.deleter
    def constraints(self):
        # Clear constraint registry.
        self._constraints.clear()
        self._con_groups.clear()

        # Update mutable registry.
        self._reset_mutable_registry()
        self._register_mutables(self.no.function.mutables)

    @property
    def objective(self):
        """Optimization objective as an :class:`~picos.Objective` instance."""
        return self._objective

    @objective.setter
    def objective(self, value):
        self._unregister_mutables(self.no.function.mutables)

        try:
            if isinstance(value, Objective):
                self._objective = value
            else:
                direction, function = value
                self._objective = Objective(direction, function)
        finally:
            self._register_mutables(self.no.function.mutables)

    @objective.deleter
    def objective(self):
        self._unregister_mutables(self.no.function.mutables)

        self._objective = Objective()

    @property
    def no(self):
        """Normalized objective as an :class:`~picos.Objective` instance.

        Either a minimization or a maximization objective, with feasibility
        posed as "minimize 0".

        The same as the :attr:`~.objective.Objective.normalized` attribute of
        the :attr:`objective`.
        """
        return self._objective.normalized

    @property
    def minimize(self):
        """Minimization objective as an :class:`~.expression.Expression`.

        This can be used to set a minimization objective. For querying the
        objective, it is recommended to use :attr:`objective` instead.
        """
        if self._objective.direction == Objective.MIN:
            return self._objective.function
        else:
            raise ValueError("Objective direction is not minimize.")

    @minimize.setter
    def minimize(self, value):
        self.objective = "min", value

    @minimize.deleter
    def minimize(self):
        if self._objective.direction == Objective.MIN:
            del self.objective
        else:
            raise ValueError("Objective direction is not minimize.")

    @property
    def maximize(self):
        """Maximization objective as an :class:`~.expression.Expression`.

        This can be used to set a maximization objective. For querying the
        objective, it is recommended to use :attr:`objective` instead.
        """
        if self._objective.direction == Objective.MAX:
            return self._objective.function
        else:
            raise ValueError("Objective direction is not maximize.")

    @maximize.setter
    def maximize(self, value):
        self.objective = "max", value

    @maximize.deleter
    def maximize(self):
        if self._objective.direction == Objective.MAX:
            del self.objective
        else:
            raise ValueError("Objective direction is not maximize.")

    @property
    def options(self):
        """Solution search parameters as an :class:`~picos.Options` object."""
        return self._options

    @options.setter
    def options(self, value):
        if not isinstance(value, Options):
            raise TypeError("Cannot assign an object of type {} as a problem's "
                " options.".format(type(value).__name__))

        self._options = value

    @options.deleter
    def options(self, value):
        self._options.reset()

    @property
    def strategy(self):
        """Solution strategy as a :class:`~picos.modeling.Strategy` object.

        A strategy is available once you order the problem to be solved and it
        will be reused for successive solution attempts (of a modified problem)
        while it remains valid with respect to the problem's :attr:`footprint`.

        When a strategy is reused, modifications to the objective and
        constraints of a problem are passed step by step through the strategy's
        reformulation pipeline while existing reformulation work is not
        repeated. If the solver also supports these kinds of updates, then
        modifying and re-solving a problem can be much faster than solving the
        problem from scratch.

        :Example:

        >>> from picos import Problem, RealVariable
        >>> x = RealVariable("x", 2)
        >>> P = Problem()
        >>> P.set_objective("min", abs(x)**2)
        >>> print(P.strategy)
        None
        >>> sol = P.solve(solver = "cvxopt")  # Creates a solution strategy.
        >>> print(P.strategy)
        1. ExtraOptions
        2. EpigraphReformulation
        3. SquaredNormToConicReformulation
        4. CVXOPTSolver
        >>> # Add another constraint handled by SquaredNormToConicReformulation:
        >>> P.add_constraint(abs(x - 2)**2 <= 1)
        <Squared Norm Constraint: ‖x - [2]‖² ≤ 1>
        >>> P.strategy.valid(solver = "cvxopt")
        True
        >>> P.strategy.valid(solver = "glpk")
        False
        >>> sol = P.solve(solver = "cvxopt")  # Reuses the strategy.

        It's also possible to create a startegy from scratch:

        >>> from picos.modeling import Strategy
        >>> from picos.reforms import (EpigraphReformulation,
        ...     ConvexQuadraticToConicReformulation)
        >>> from picos.solvers import CVXOPTSolver
        >>> # Mimic what solve() does when no strategy exists:
        >>> P.strategy = Strategy(P, CVXOPTSolver, EpigraphReformulation,
        ...     ConvexQuadraticToConicReformulation)
        """
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        from .strategy import Strategy

        if not isinstance(value, Strategy):
            raise TypeError(
                "Cannot assign an object of type {} as a solution strategy."
                .format(type(value).__name__))

        if value.problem is not self:
            raise ValueError("The solution strategy was constructed for a "
                "different problem.")

        self._strategy = value

    @strategy.deleter
    def strategy(self):
        self._strategy = None

    @property
    def last_solution(self):
        """The last :class:`~picos.Solution` applied to the problem."""
        return self._last_solution

    @property
    def status(self):
        """The solution status string as claimed by :attr:`last_solution`."""
        if not self._last_solution:
            return "unsolved"
        else:
            return self._last_solution.claimedStatus

    @property
    def footprint(self):
        """Problem footprint as a :class:`~picos.modeling.Footprint` object."""
        return Footprint.from_problem(self)

    @property
    def continuous(self):
        """Whether all variables are of continuous types."""
        return all(
            isinstance(variable, expressions.CONTINUOUS_VARTYPES)
            for variable in self._variables.values())

    @property
    def pure_integer(self):
        """Whether all variables are of integral types."""
        return not any(
            isinstance(variable, expressions.CONTINUOUS_VARTYPES)
            for variable in self._variables.values())

    @property
    def type(self):
        """The problem type as a string, such as "Linear Program"."""
        C = set(type(c) for c in self._constraints.values())
        objective = self._objective.function
        base = "Optimization Problem"

        linear = [
            constraints.AffineConstraint,
            constraints.ComplexAffineConstraint,
            constraints.AbsoluteValueConstraint,
            constraints.SimplexConstraint,
            constraints.FlowConstraint]
        sdp = [
            constraints.LMIConstraint,
            constraints.ComplexLMIConstraint]
        quadratic = [
            constraints.ConvexQuadraticConstraint,
            constraints.ConicQuadraticConstraint,
            constraints.NonconvexQuadraticConstraint]
        quadconic = [
            constraints.SOCConstraint,
            constraints.RSOCConstraint]
        exponential = [
            constraints.ExpConeConstraint,
            constraints.SumExponentialsConstraint,
            constraints.LogSumExpConstraint,
            constraints.LogConstraint,
            constraints.KullbackLeiblerConstraint]
        complex = [
            constraints.ComplexAffineConstraint,
            constraints.ComplexLMIConstraint]

        if objective is None:
            if not C:
                base = "Empty Problem"
            elif C.issubset(set(linear)):
                base = "Linear Feasibility Problem"
            else:
                base = "Feasibility Problem"
        elif isinstance(objective, expressions.AffineExpression):
            if not C:
                if objective.constant:
                    base = "Constant Problem"
                else:
                    base = "Linear Program"  # Could have variable bounds.
            elif C.issubset(set(linear)):
                base = "Linear Program"
            elif C.issubset(set(linear + quadconic)):
                base = "Second Order Cone Program"
            elif C.issubset(set(linear + sdp)):
                base = "Semidefinite Program"
            elif C.issubset(set(linear + [constraints.LogSumExpConstraint])):
                base = "Geometric Program"
            elif C.issubset(set(linear + exponential)):
                base = "Exponential Program"
            elif C.issubset(set(linear + quadratic)):
                base = "Quadratically Constrained Program"
        elif isinstance(objective, expressions.QuadraticExpression):
            if C.issubset(set(linear)):
                base = "Quadratic Program"
            elif C.issubset(set(linear + quadratic)):
                base = "Quadratically Constrained Quadratic Program"
        elif isinstance(objective, expressions.LogSumExp):
            if C.issubset(set(linear + [constraints.LogSumExpConstraint])):
                base = "Geometric Program"

        if self.continuous:
            integrality = ""
        elif self.pure_integer:
            integrality = "Integer "
        else:
            integrality = "Mixed-Integer "

        if any(c in complex for c in C):
            complexity = "Complex "
        else:
            complexity = ""

        return "{}{}{}".format(complexity, integrality, base)

    @property
    def dual(self):
        """The Lagrangian dual problem of the standardized problem.

        More precisely, this property invokes the following:

        1. The primal problem is posed as an equivalent conic standard form
           minimization problem, with variable bounds expressed as additional
           constraints.
        2. The Lagrangian dual problem of the reposed primal is computed.
        3. The optimization direction and objective function sign of the dual
           are adjusted such that, given strong duality and primal feasibility,
           the optimal values of both problems are equal. In particular, if the
           primal problem is a minimization or a maximization problem, the dual
           problem returned will be the respective other.

        :raises ~picos.modeling.strategy.NoStrategyFound:
            If no reformulation strategy was found.

        .. note::

            This property is intended for educational purposes.
            If you want to solve the primal problem via its dual, use the
            :ref:`dualize <option_dualize>` option instead.
        """
        from ..reforms import Dualization
        return self.reformulated(Dualization.SUPPORTED, dualize=True)

    @property
    def conic_form(self):
        """The problem in conic form.

        Reformulates the problem such that the objective is affine and all
        constraints are :class:`~.constraints.ConicConstraint` instances.

        :raises ~picos.modeling.strategy.NoStrategyFound:
            If no reformulation strategy was found.

        :Example:

        >>> from picos import Problem, RealVariable
        >>> x = RealVariable("x", 2)
        >>> P = Problem()
        >>> P.set_objective("min", abs(x)**2)
        >>> print(P)
        Quadratic Program
          minimize ‖x‖²
          over
            2×1 real variable x
        >>> print(P.conic_form)# doctest: +ELLIPSIS
        Second Order Cone Program
          minimize __..._t
          over
            1×1 real variable __..._t
            2×1 real variable x
          subject to
            ‖fullroot(‖x‖²)‖² ≤ __..._t ∧ __..._t ≥ 0

        .. note::

            This property is intended for educational purposes.
            You do not need to use it when solving a problem as PICOS will
            perform the necessary reformulations automatically.
        """
        return self.reformulated(self.CONIC_FORM)

    # --------------------------------------------------------------------------
    # Python special methods, except __init__.
    # --------------------------------------------------------------------------

    @property
    def _var_groups(self):
        """Support :meth:`__str__`."""
        vars_by_type = {}
        for var in self._variables.values():
            vtype = type(var).__name__
            shape = var.shape
            bound = tuple(bool(bound) for bound in var.bound_dicts)
            index = (vtype, shape, bound)

            vars_by_type.setdefault(index, set())
            vars_by_type[index].add(var)

        groups = []
        for index in sorted(vars_by_type.keys()):
            groups.append(natsorted(vars_by_type[index], key=lambda v: v.name))

        return groups

    @property
    def _prm_groups(self):
        """Support :meth:`__str__`."""
        prms_by_type = {}
        for prm in self._parameters.values():
            vtype = type(prm).__name__
            shape = prm.shape
            index = (vtype, shape)

            prms_by_type.setdefault(index, set())
            prms_by_type[index].add(prm)

        groups = []
        for index in sorted(prms_by_type.keys()):
            groups.append(natsorted(prms_by_type[index], key=lambda v: v.name))

        return groups

    @lru_cache()
    def _mtb_group_string(self, group):
        """Support :meth:`__str__`."""
        if len(group) == 0:
            return "[no mutables]"

        if len(group) == 1:
            return group[0].long_string

        try:
            template, data = parameterized_string(
                [mtb.long_string for mtb in group])
        except ValueError:
            # HACK: Use the plural of the type string (e.g. "real variables").
            type_string = group[0]._get_type_string_base().lower()
            base_string = group[0].long_string.replace(
                type_string, type_string + "s")

            # HACK: Move any bound string to the end.
            match = re.match(r"([^(]*)( \([^)]*\))", base_string)
            if match:
                base_string = match[1]
                bound_string = match[2]
            else:
                bound_string = ""

            return base_string \
                + ", " + ", ".join([v.name for v in group[1:]]) + bound_string
        else:
            return glyphs.forall(template, data)

    @lru_cache()
    def _con_group_string(self, group):
        """Support :meth:`__str__`."""
        if len(group) == 0:
            return "[no constraints]"

        if len(group) == 1:
            return str(group[0])

        try:
            template, data = parameterized_string([str(con) for con in group])
        except ValueError:
            return "[{} constraints (1st: {})]".format(len(group), group[0])
        else:
            return glyphs.forall(template, data)

    def __repr__(self):
        if self._name:
            return glyphs.repr2(self.type, self._name)
        else:
            return glyphs.repr1(self.type)

    def __str__(self):
        # Print problem name (if available) and type.
        if self._name:
            string = "{} ({})\n".format(self._name, self.type)
        else:
            string = "{}\n".format(self.type)

        # Print objective.
        string += "  {}\n".format(self._objective)

        wrapper = TextWrapper(
            initial_indent=" "*4,
            subsequent_indent=" "*6,
            break_long_words=False,
            break_on_hyphens=False)

        # Print variables.
        if self._variables:
            string += "  {}\n".format(
                "for" if self._objective.direction == "find" else "over")
            for group in self._var_groups:
                string += wrapper.fill(self._mtb_group_string(tuple(group)))
                string += "\n"

        # Print constraints.
        if self._constraints:
            string += "  subject to\n"
            for index, group in enumerate(self._con_groups):
                string += wrapper.fill(self._con_group_string(tuple(group)))
                string += "\n"

        # Print parameters.
        if self._parameters:
            string += "  given\n"
            for group in self._prm_groups:
                string += wrapper.fill(self._mtb_group_string(tuple(group)))
                string += "\n"

        return string.rstrip("\n")

    def __iadd__(self, constraints):
        """See :meth:`require`."""
        if isinstance(constraints, tuple):
            self.require(*constraints)
        else:
            self.require(constraints)

        return self

    # --------------------------------------------------------------------------
    # Bookkeeping methods.
    # --------------------------------------------------------------------------

    def _register_mutables(self, mtbs):
        """Register the mutables of an objective function or constraint."""
        # Register every mutable at most once per call.
        if not isinstance(mtbs, (set, frozenset)):
            raise TypeError("Mutable registry can (un)register a mutable "
                "only once per call, so the argument must be a set type.")

        # Retrieve old and new mutables as mapping from name to object.
        old_mtbs = self._mutables
        new_mtbs = OrderedDict(
            (mtb.name, mtb) for mtb in sorted(mtbs, key=(lambda m: m.name)))
        new_vars = OrderedDict((name, mtb) for name, mtb in new_mtbs.items()
            if isinstance(mtb, BaseVariable))
        new_prms = OrderedDict((name, mtb) for name, mtb in new_mtbs.items()
            if not isinstance(mtb, BaseVariable))

        # Check for mutable name clashes within the new set.
        if len(new_mtbs) != len(mtbs):
            raise ValueError(
                "The object you are trying to add to a problem contains "
                "multiple mutables of the same name. This is not allowed.")

        # Check for mutable name clashes with existing mutables.
        for name in set(old_mtbs).intersection(set(new_mtbs)):
            if old_mtbs[name] is not new_mtbs[name]:
                raise ValueError("Cannot register the mutable {} with the "
                    "problem because it already tracks another mutable with "
                    "the same name.".format(name))

        # Keep track of new mutables.
        self._mutables.update(new_mtbs)
        self._variables.update(new_vars)
        self._parameters.update(new_prms)

        # Count up the mutable references.
        for mtb in mtbs:
            self._mtb_count.setdefault(mtb, 0)
            self._mtb_count[mtb] += 1

    def _unregister_mutables(self, mtbs):
        """Unregister the mutables of an objective function or constraint."""
        # Unregister every mutable at most once per call.
        if not isinstance(mtbs, (set, frozenset)):
            raise TypeError("Mutable registry can (un)register a mutable "
                "only once per call, so the argument must be a set type.")

        for mtb in mtbs:
            name = mtb.name

            # Make sure the mutable is properly registered.
            assert name in self._mutables and mtb in self._mtb_count, \
                "Tried to unregister a mutable that is not registered."
            assert self._mtb_count[mtb] >= 1, \
                "Found a nonpostive mutable count."

            # Count down the mutable references.
            self._mtb_count[mtb] -= 1

            # Remove a mutable with a reference count of zero.
            if not self._mtb_count[mtb]:
                self._mtb_count.pop(mtb)
                self._mutables.pop(name)

                if isinstance(mtb, BaseVariable):
                    self._variables.pop(name)
                else:
                    self._parameters.pop(name)

    # --------------------------------------------------------------------------
    # Methods to manipulate the objective function and its direction.
    # --------------------------------------------------------------------------

    def set_objective(self, direction=None, expression=None):
        """Set the optimization direction and objective function of the problem.

        :param str direction:
            Case insensitive search direction string. One of

            - ``"min"`` or ``"minimize"``,
            - ``"max"`` or ``"maximize"``,
            - ``"find"`` or :obj:`None` (for a feasibility problem).

        :param ~picos.expressions.Expression expression:
            The objective function. Must be :obj:`None` for a feasibility
            problem.
        """
        self.objective = direction, expression

    # --------------------------------------------------------------------------
    # Methods to add, retrieve and remove constraints.
    # --------------------------------------------------------------------------

    def _lookup_constraint(self, idOrIndOrCon):
        """Look for a constraint with the given identifier.

        Given a constraint object or ID or offset or a constraint group index or
        index pair, returns a matching (list of) constraint ID(s) that is (are)
        part of the problem.
        """
        if isinstance(idOrIndOrCon, int):
            if idOrIndOrCon in self._constraints:
                # A valid ID.
                return idOrIndOrCon
            elif idOrIndOrCon < len(self._constraints):
                # An offset.
                return list(self._constraints.keys())[idOrIndOrCon]
            else:
                raise LookupError(
                    "The problem has no constraint with ID or offset {}."
                    .format(idOrIndOrCon))
        elif isinstance(idOrIndOrCon, constraints.Constraint):
            # A constraint object.
            id = idOrIndOrCon.id
            if id in self._constraints:
                return id
            else:
                raise KeyError("The constraint '{}' is not part of the problem."
                    .format(idOrIndOrCon))
        elif isinstance(idOrIndOrCon, tuple) or isinstance(idOrIndOrCon, list):
            if len(idOrIndOrCon) == 1:
                groupIndex = idOrIndOrCon[0]
                if groupIndex < len(self._con_groups):
                    return [c.id for c in self._con_groups[groupIndex]]
                else:
                    raise IndexError("Constraint group index out of range.")
            elif len(idOrIndOrCon) == 2:
                groupIndex, groupOffset = idOrIndOrCon
                if groupIndex < len(self._con_groups):
                    group = self._con_groups[groupIndex]
                    if groupOffset < len(group):
                        return group[groupOffset].id
                    else:
                        raise IndexError(
                            "Constraint group offset out of range.")
                else:
                    raise IndexError("Constraint group index out of range.")
            else:
                raise TypeError("If looking up constraints by group, the index "
                    "must be a tuple or list of length at most two.")
        else:
            raise TypeError("Argument of type '{}' not supported when looking "
                "up constraints".format(type(idOrIndOrCon)))

    def get_constraint(self, idOrIndOrCon):
        """Return a (list of) constraint(s) of the problem.

        :param idOrIndOrCon: One of the following:

            * A constraint object. It will be returned when the constraint is
              part of the problem, otherwise a KeyError is raised.
            * The integer ID of the constraint.
            * The integer offset of the constraint in the list of all
              constraints that are part of the problem, in the order that they
              were added.
            * A list or tuple of length 1. Its only element is the index of a
              constraint group (of constraints that were added together), where
              groups are indexed in the order that they were added to the
              problem. The whole group is returned as a list of constraints.
              That list has the constraints in the order that they were added.
            * A list or tuple of length 2. The first element is a constraint
              group offset as above, the second an offset within that list.

        :type idOrIndOrCon: picos.constraints.Constraint or int or tuple or list

        :returns: A :class:`constraint <picos.constraints.Constraint>` or a list
            thereof.

        :Example:

        >>> import picos as pic
        >>> import cvxopt as cvx
        >>> from pprint import pprint
        >>> prob=pic.Problem()
        >>> x=[prob.add_variable('x[{0}]'.format(i),2) for i in range(5)]
        >>> y=prob.add_variable('y',5)
        >>> Cx=prob.add_list_of_constraints([(1|x[i]) < y[i] for i in range(5)])
        >>> Cy=prob.add_constraint(y>0)
        >>> print(prob)
        Linear Feasibility Problem
          find an assignment
          for
            2×1 real variable x[i] ∀ i ∈ [0…4]
            5×1 real variable y
          subject to
            ∑(x[i]) ≤ y[i] ∀ i ∈ [0…4]
            y ≥ 0
        >>> # Retrieve the second constraint, indexed from zero:
        >>> prob.get_constraint(1)
        <1×1 Affine Constraint: ∑(x[1]) ≤ y[1]>
        >>> # Retrieve the fourth consraint from the first group:
        >>> prob.get_constraint((0,3))
        <1×1 Affine Constraint: ∑(x[3]) ≤ y[3]>
        >>> # Retrieve the whole first group of constraints:
        >>> pprint(prob.get_constraint((0,)))
        [<1×1 Affine Constraint: ∑(x[0]) ≤ y[0]>,
         <1×1 Affine Constraint: ∑(x[1]) ≤ y[1]>,
         <1×1 Affine Constraint: ∑(x[2]) ≤ y[2]>,
         <1×1 Affine Constraint: ∑(x[3]) ≤ y[3]>,
         <1×1 Affine Constraint: ∑(x[4]) ≤ y[4]>]
        >>> # Retrieve the second "group", containing just one constraint:
        >>> prob.get_constraint((1,))
        [<5×1 Affine Constraint: y ≥ 0>]
        """
        idOrIds = self._lookup_constraint(idOrIndOrCon)

        if isinstance(idOrIds, list):
            return [self._constraints[id] for id in idOrIds]
        else:
            return self._constraints[idOrIds]

    def add_constraint(self, constraint, key=None):
        """Add a single constraint to the problem and return it.

        :param constraint:
            The constraint to be added.
        :type constraint:
            :class:`Constraint <picos.constraints.Constraint>`

        :param key: DEPRECATED

        :returns:
            The constraint that was added to the problem.

        .. note::

            This method is superseded by the more compact and more flexible
            :meth:`require` method or, at your preference, the ``+=`` operator.
        """
        # Handle deprecated 'key' parameter.
        if key is not None:
            throw_deprecation_warning(
                "Naming constraints is currently not supported.")

        # Register the constraint.
        self._constraints[constraint.id] = constraint
        self._con_groups.append([constraint])

        # Register the constraint's mutables.
        self._register_mutables(constraint.mutables)

        return constraint

    def add_list_of_constraints(self, lst, it=None, indices=None, key=None):
        """Add constraints from an iterable to the problem.

        :param lst:
            Iterable of constraints to add.

        :param it: DEPRECATED
        :param indices: DEPRECATED
        :param key: DEPRECATED

        :returns:
            A list of all constraints that were added.

        :Example:

        >>> import picos as pic
        >>> import cvxopt as cvx
        >>> from pprint import pprint
        >>> prob=pic.Problem()
        >>> x=[prob.add_variable('x[{0}]'.format(i),2) for i in range(5)]
        >>> pprint(x)
        [<2×1 Real Variable: x[0]>,
         <2×1 Real Variable: x[1]>,
         <2×1 Real Variable: x[2]>,
         <2×1 Real Variable: x[3]>,
         <2×1 Real Variable: x[4]>]
        >>> y=prob.add_variable('y',5)
        >>> IJ=[(1,2),(2,0),(4,2)]
        >>> w={}
        >>> for ij in IJ:
        ...         w[ij]=prob.add_variable('w[{},{}]'.format(*ij),3)
        ...
        >>> u=pic.new_param('u',cvx.matrix([2,5]))
        >>> C1=prob.add_list_of_constraints([u.T*x[i] < y[i] for i in range(5)])
        >>> C2=prob.add_list_of_constraints([abs(w[i,j])<y[j] for (i,j) in IJ])
        >>> C3=prob.add_list_of_constraints([y[t] > y[t+1] for t in range(4)])
        >>> print(prob)
        Feasibility Problem
          find an assignment
          for
            2×1 real variable x[i] ∀ i ∈ [0…4]
            3×1 real variable w[i,j] ∀ (i,j) ∈ zip([1,2,4],[2,0,2])
            5×1 real variable y
          subject to
            uᵀ·x[i] ≤ y[i] ∀ i ∈ [0…4]
            ‖w[i,j]‖ ≤ y[j] ∀ (i,j) ∈ zip([1,2,4],[2,0,2])
            y[i] ≥ y[i+1] ∀ i ∈ [0…3]

        .. note::

            This method is superseded by the more compact and more flexible
            :meth:`require` method or, at your preference, the ``+=`` operator.
        """
        if it is not None or indices is not None or key is not None:
            # Deprecated as of 2.0.
            throw_deprecation_warning("Arguments 'it', 'indices' and 'key' to "
                "add_list_of_constraints are deprecated and ignored.")

        added = []
        for constraint in lst:
            added.append(self.add_constraint(constraint))
            self._con_groups.pop()

        if added:
            self._con_groups.append(added)

        return added

    def require(self, *constraints, ret=False):
        """Add constraints to the problem.

        :param constraints:
            A sequence of constraints or constraint groups (iterables yielding
            constraints) or a mix thereof.

        :param bool ret:
            Whether to return the added constraints.

        :returns:
            When ``ret=True``, returns either the single constraint that was
            added, the single group of constraint that was added in the form of
            a :class:`list` or, when multiple arguments are given, a list of
            constraints or constraint groups represented as above. When
            ``ret=False``, returns nothing.

        :Example:

        >>> from picos import Problem, RealVariable
        >>> x = RealVariable("x", 5)
        >>> P = Problem()
        >>> P.require(x >= -1, x <= 1)  # Add individual constraints.
        >>> P.require([x[i] <= x[i+1] for i in range(4)])  # Add groups.
        >>> print(P)
        Linear Feasibility Problem
          find an assignment
          for
            5×1 real variable x
          subject to
            x ≥ [-1]
            x ≤ [1]
            x[i] ≤ x[i+1] ∀ i ∈ [0…3]

        .. note::

            For a single constraint ``C``, ``P.require(C)`` may also be written
            as ``P += C``. For multiple constraints, ``P.require([C1, C2])`` can
            be abbreviated ``P += [C1, C2]`` while ``P.require(C1, C2)`` can be
            written as either ``P += (C1, C2)`` or just ``P += C1, C2``.
        """
        from ..constraints import Constraint

        added = []
        for constraint in constraints:
            if isinstance(constraint, Constraint):
                added.append(self.add_constraint(constraint))
            else:
                try:
                    if not all(isinstance(c, Constraint) for c in constraint):
                        raise TypeError
                except TypeError:
                    raise TypeError(
                        "An argument is neither a constraint nor an iterable "
                        "yielding constraints.") from None
                else:
                    added.append(self.add_list_of_constraints(constraint))

        if ret:
            return added[0] if len(added) == 1 else added

    def _con_group_index(self, conOrConID):
        """Support :meth:`remove_constraint`."""
        if isinstance(conOrConID, int):
            constraint = self._constraints[conOrConID]
        else:
            constraint = conOrConID

        for i, group in enumerate(self._con_groups):
            for j, candidate in enumerate(group):
                if candidate is constraint:
                    return i, j

        if constraint in self._constraints.values():
            raise RuntimeError("The problem's constraint and constraint group "
                "registries are out of sync.")
        else:
            raise KeyError("The constraint is not part of the problem.")

    def remove_constraint(self, idOrIndOrCon):
        """Delete a constraint from the problem.

        :param idOrIndOrCon: See :meth:`get_constraint`.

        :Example:

        >>> import picos
        >>> from pprint import pprint
        >>> P = picos.Problem()
        >>> x = [P.add_variable('x[{0}]'.format(i), 2) for i in range(4)]
        >>> y = P.add_variable('y', 4)
        >>> Cxy = P.add_list_of_constraints(
        ...     [(1 | x[i]) <= y[i] for i in range(4)])
        >>> Cy = P.add_constraint(y >= 0)
        >>> Cx0to2 = P.add_list_of_constraints([x[i] <= 2 for i in range(3)])
        >>> Cx3 = P.add_constraint(x[3] <= 1)
        >>> pprint(list(P.constraints.values()))#doctest: +NORMALIZE_WHITESPACE
        [<1×1 Affine Constraint: ∑(x[0]) ≤ y[0]>,
         <1×1 Affine Constraint: ∑(x[1]) ≤ y[1]>,
         <1×1 Affine Constraint: ∑(x[2]) ≤ y[2]>,
         <1×1 Affine Constraint: ∑(x[3]) ≤ y[3]>,
         <4×1 Affine Constraint: y ≥ 0>,
         <2×1 Affine Constraint: x[0] ≤ [2]>,
         <2×1 Affine Constraint: x[1] ≤ [2]>,
         <2×1 Affine Constraint: x[2] ≤ [2]>,
         <2×1 Affine Constraint: x[3] ≤ [1]>]
        >>> # Delete the 2nd constraint (counted from 0):
        >>> P.remove_constraint(1)
        >>> pprint(list(P.constraints.values()))#doctest: +NORMALIZE_WHITESPACE
        [<1×1 Affine Constraint: ∑(x[0]) ≤ y[0]>,
         <1×1 Affine Constraint: ∑(x[2]) ≤ y[2]>,
         <1×1 Affine Constraint: ∑(x[3]) ≤ y[3]>,
         <4×1 Affine Constraint: y ≥ 0>,
         <2×1 Affine Constraint: x[0] ≤ [2]>,
         <2×1 Affine Constraint: x[1] ≤ [2]>,
         <2×1 Affine Constraint: x[2] ≤ [2]>,
         <2×1 Affine Constraint: x[3] ≤ [1]>]
        >>> # Delete the 2nd group of constraints, i.e. the constraint y > 0:
        >>> P.remove_constraint((1,))
        >>> pprint(list(P.constraints.values()))#doctest: +NORMALIZE_WHITESPACE
        [<1×1 Affine Constraint: ∑(x[0]) ≤ y[0]>,
         <1×1 Affine Constraint: ∑(x[2]) ≤ y[2]>,
         <1×1 Affine Constraint: ∑(x[3]) ≤ y[3]>,
         <2×1 Affine Constraint: x[0] ≤ [2]>,
         <2×1 Affine Constraint: x[1] ≤ [2]>,
         <2×1 Affine Constraint: x[2] ≤ [2]>,
         <2×1 Affine Constraint: x[3] ≤ [1]>]
        >>> # Delete the 3rd remaining group of constraints, i.e. x[3] < [1]:
        >>> P.remove_constraint((2,))
        >>> pprint(list(P.constraints.values()))#doctest: +NORMALIZE_WHITESPACE
        [<1×1 Affine Constraint: ∑(x[0]) ≤ y[0]>,
         <1×1 Affine Constraint: ∑(x[2]) ≤ y[2]>,
         <1×1 Affine Constraint: ∑(x[3]) ≤ y[3]>,
         <2×1 Affine Constraint: x[0] ≤ [2]>,
         <2×1 Affine Constraint: x[1] ≤ [2]>,
         <2×1 Affine Constraint: x[2] ≤ [2]>]
        >>> # Delete 2nd constraint of the 2nd remaining group, i.e. x[1] < |2|:
        >>> P.remove_constraint((1,1))
        >>> pprint(list(P.constraints.values()))#doctest: +NORMALIZE_WHITESPACE
        [<1×1 Affine Constraint: ∑(x[0]) ≤ y[0]>,
         <1×1 Affine Constraint: ∑(x[2]) ≤ y[2]>,
         <1×1 Affine Constraint: ∑(x[3]) ≤ y[3]>,
         <2×1 Affine Constraint: x[0] ≤ [2]>,
         <2×1 Affine Constraint: x[2] ≤ [2]>]
        """
        idOrIds = self._lookup_constraint(idOrIndOrCon)

        removedCons = []

        if isinstance(idOrIds, list):
            assert idOrIds, "There is an empty constraint group."
            groupIndex, _ = self._con_group_index(idOrIds[0])
            self._con_groups.pop(groupIndex)
            for id in idOrIds:
                removedCons.append(self._constraints.pop(id))
        else:
            constraint = self._constraints.pop(idOrIds)
            removedCons.append(constraint)
            groupIndex, groupOffset = self._con_group_index(constraint)
            group = self._con_groups[groupIndex]
            group.pop(groupOffset)
            if not group:
                self._con_groups.pop(groupIndex)

        # Unregister the mutables added by the removed constraints.
        for con in removedCons:
            self._unregister_mutables(con.mutables)

    def remove_all_constraints(self):
        """Remove all constraints from the problem.

        .. note::

            This method does not remove bounds set directly on variables.
        """
        del self.constraints

    # --------------------------------------------------------------------------
    # Borderline legacy methods to deal with variables.
    # --------------------------------------------------------------------------

    _PARAMETERIZED_VARIABLE_REGEX = re.compile(r"^([^[]+)\[([^\]]+)\]$")

    def get_variable(self, name):
        """Retrieve variables referenced by the problem.

        Retrieves either a single variable with the given name or a group of
        variables all named ``name[param]`` with different values for ``param``.
        If the values for ``param`` are the integers from zero to the size of
        the group minus one, then the group is returned as a :obj:`list` ordered
        by ``param``, otherwise it is returned as a :obj:`dict` with the values
        of ``param`` as keys.

        .. note::

            Since PICOS 2.0, variables are independent of problems and only
            appear in a problem for as long as they are referenced by the
            problem's objective function or constraints.

        :param str name:
            The name of a variable, or the base name of a group of variables.

        :returns:
            A :class:`variable <picos.expressions.BaseVariable>` or a
            :class:`list` or :class:`dict` thereof.

        :Example:

        >>> from picos import Problem, RealVariable
        >>> from pprint import pprint
        >>> # Create a number of variables with structured names.
        >>> vars = [RealVariable("x")]
        >>> for i in range(4):
        ...     vars.append(RealVariable("y[{}]".format(i)))
        >>> for key in ["alice", "bob", "carol"]:
        ...     vars.append(RealVariable("z[{}]".format(key)))
        >>> # Make the variables appear in a problem.
        >>> P = Problem()
        >>> P.set_objective("min", sum([var for var in vars]))
        >>> print(P)
        Linear Program
          minimize x + y[0] + y[1] + y[2] + y[3] + z[alice] + z[bob] + z[carol]
          over
            1×1 real variables x, y[0], y[1], y[2], y[3], z[alice], z[bob],
              z[carol]
        >>> # Retrieve the variables from the problem.
        >>> P.get_variable("x")
        <1×1 Real Variable: x>
        >>> pprint(P.get_variable("y"))
        [<1×1 Real Variable: y[0]>,
         <1×1 Real Variable: y[1]>,
         <1×1 Real Variable: y[2]>,
         <1×1 Real Variable: y[3]>]
        >>> pprint(P.get_variable("z"))
        {'alice': <1×1 Real Variable: z[alice]>,
         'bob': <1×1 Real Variable: z[bob]>,
         'carol': <1×1 Real Variable: z[carol]>}
        >>> P.get_variable("z")["alice"] is P.get_variable("z[alice]")
        True
        """
        if name in self._variables:
            return self._variables[name]
        else:
            # Check if the name is really just a basename.
            params = []
            for otherName in sorted(self._variables.keys()):
                match = self._PARAMETERIZED_VARIABLE_REGEX.match(otherName)
                if not match:
                    continue
                base, param = match.groups()
                if name == base:
                    params.append(param)

            if params:
                # Return a list if the parameters are a range.
                try:
                    intParams = sorted([int(p) for p in params])
                except ValueError:
                    pass
                else:
                    if intParams == list(range(len(intParams))):
                        return [self._variables["{}[{}]".format(name, param)]
                            for param in intParams]

                # Otherwise return a dict.
                return {param: self._variables["{}[{}]".format(name, param)]
                    for param in params}
            else:
                raise KeyError("The problem references no variable or group of "
                    "variables named '{}'.".format(name))

    def get_valued_variable(self, name):
        """Retrieve values of variables referenced by the problem.

        This method works the same :meth:`get_variable` but it returns the
        variable's :attr:`values <.valuable.Valuable.value>` instead of the
        variable objects.

        :raises ~picos.expressions.NotValued:
            If any of the selected variables is not valued.
        """
        exp = self.get_variable(name)
        if isinstance(exp, list):
            for i in range(len(exp)):
                exp[i] = exp[i].value
        elif isinstance(exp, dict):
            for i in exp:
                exp[i] = exp[i].value
        else:
            exp = exp.value
        return exp

    # --------------------------------------------------------------------------
    # Methods to create copies of the problem.
    # --------------------------------------------------------------------------

    def copy(self):
        """Create a deep copy of the problem, using new mutables."""
        the_copy = Problem(copyOptions=self._options)

        # Duplicate the mutables.
        new_mtbs = {mtb: mtb.copy() for name, mtb in self._mutables.items()}

        # Make copies of constraints on top of the new mutables.
        for group in self._con_groups:
            the_copy.add_list_of_constraints(
                constraint.replace_mutables(new_mtbs) for constraint in group)

        # Make a copy of the objective on top of the new mutables.
        direction, function = self._objective
        if function is not None:
            the_copy.objective = direction, function.replace_mutables(new_mtbs)

        return the_copy

    def continuous_relaxation(self, copy_other_mutables=True):
        """Return a continuous relaxation of the problem.

        This is done by replacing integer variables with continuous ones.

        :param bool copy_other_mutables:
            Whether variables that are already continuous as well as parameters
            should be copied. If this is :obj:`False`, then the relxation shares
            these mutables with the original problem.
        """
        the_copy = Problem(copyOptions=self._options)

        # Relax integral variables and copy other mutables if requested.
        new_mtbs = {}
        for name, var in self._mutables.items():
            if isinstance(var, expressions.IntegerVariable):
                new_mtbs[name] = expressions.RealVariable(
                    name, var.shape, var._lower, var._upper)
            elif isinstance(var, expressions.BinaryVariable):
                new_mtbs[name] = expressions.RealVariable(name, var.shape, 0, 1)
            else:
                if copy_other_mutables:
                    new_mtbs[name] = var.copy()
                else:
                    new_mtbs[name] = var

        # Make copies of constraints on top of the new mutables.
        for group in self._con_groups:
            the_copy.add_list_of_constraints(
                constraint.replace_mutables(new_mtbs) for constraint in group)

        # Make a copy of the objective on top of the new mutables.
        direction, function = self._objective
        if function is not None:
            the_copy.objective = direction, function.replace_mutables(new_mtbs)

        return the_copy

    def clone(self, copyOptions=True):
        """Create a semi-deep copy of the problem.

        The copy is constrained by the same constraint objects and has the same
        objective function and thereby references the existing variables and
        parameters that appear in these objects.

        The clone can be modified to describe a new problem but when its
        variables and parameters are valued, in particular when a solution is
        applied to the new problem, then the same values are found in the
        corresponding variables and parameters of the old problem. If this is
        not a problem to you, then cloning can be much faster than copying.

        :param bool copyOptions:
            Whether to make an independent copy of the problem's options.
            Disabling this will apply any option changes to the original problem
            as well but yields a (very small) reduction in cloning time.
        """
        # Start with a shallow copy of self.
        # TODO: Consider adding Problem.__new__ to speed this up further.
        theClone = pycopy.copy(self)

        # Make the constraint registry independent.
        theClone._constraints = self._constraints.copy()
        theClone._con_groups = []
        for group in self._con_groups:
            theClone._con_groups.append(pycopy.copy(group))

        # Make the mutable registry independent.
        theClone._mtb_count = self._mtb_count.copy()
        theClone._mutables = self._mutables.copy()
        theClone._variables = self._variables.copy()
        theClone._parameters = self._parameters.copy()

        # Reset the clone's solution strategy and last solution.
        theClone._strategy = None

        # Make the solver options independent, if requested.
        if copyOptions:
            theClone._options = self._options.copy()

        # NOTE: No need to change the following attributes:
        #       - objective: Is immutable as a tuple.
        #       - _last_solution: Remains as valid as it is.

        return theClone

    # --------------------------------------------------------------------------
    # Methods to solve or export the problem.
    # --------------------------------------------------------------------------

    def prepared(self, steps=None, **extra_options):
        """Perform a dry-run returning the reformulated (prepared) problem.

        This behaves like :meth:`solve` in that it takes a number of additional
        temporary options, finds a solution strategy matching the problem and
        options, and performs the strategy's reformulations in turn to obtain
        modified problems. However, it stops after the given number of steps and
        never hands the reformulated problem to a solver. Instead of a solution,
        :meth:`prepared` then returns the last reformulated problem.

        Unless this method returns the problem itself, the special attributes
        ``prepared_strategy`` and ``prepared_steps`` are added to the returned
        problem. They then contain the (partially) executed solution strategy
        and the number of performed reformulations, respectively.

        :param int steps:
            Number of reformulations to perform. :obj:`None` means as many as
            there are. If this parameter is :math:`0`, then the problem itself
            is returned. If it is :math:`1`, then only the implicit first
            reformulation :class:`~.reform_options.ExtraOptions` is executed,
            which may also output the problem itself, depending on
            ``extra_options``.

        :param extra_options:
            Additional solver options to use with this dry-run only.

        :returns:
            The reformulated problem, with ``extra_options`` set unless they
            were "consumed" by a reformulation (e.g.
            :ref:`option_dualize <option_dualize>`).

        :raises ~picos.modeling.strategy.NoStrategyFound:
            If no solution strategy was found.

        :raises ValueError:
            If there are not as many reformulation steps as requested.

        :Example:

        >>> from picos import Problem, RealVariable
        >>> x = RealVariable("x", 2)
        >>> P = Problem()
        >>> P.set_objective("min", abs(x)**2)
        >>> Q = P.prepared(solver = "cvxopt")
        >>> print(Q.prepared_strategy)  # Show prepared reformulation steps.
        1. ExtraOptions
        2. EpigraphReformulation
        3. SquaredNormToConicReformulation
        4. CVXOPTSolver
        >>> Q.prepared_steps  # Check how many steps have been performed.
        3
        >>> print(P)
        Quadratic Program
          minimize ‖x‖²
          over
            2×1 real variable x
        >>> print(Q)# doctest: +ELLIPSIS
        Second Order Cone Program
          minimize __..._t
          over
            1×1 real variable __..._t
            2×1 real variable x
          subject to
            ‖fullroot(‖x‖²)‖² ≤ __..._t ∧ __..._t ≥ 0
        """
        from .strategy import Strategy

        # Produce a strategy for the clone.
        strategy = Strategy.from_problem(self, **extra_options)
        numReforms = len(strategy.reforms)

        if steps is None:
            steps = numReforms

        if steps == 0:
            return self
        elif steps > numReforms:
            raise ValueError("The pipeline {} has only {} reformulation steps "
                "to choose from.".format(strategy, numReforms))

        # Replace the successor of the last reformulation with a dummy solver.
        lastReform = strategy.reforms[steps - 1]
        oldSuccessor = lastReform.successor
        lastReform.successor = type("DummySolver", (), {
            "execute": lambda self: Solution(
                {}, solver="dummy", vectorizedPrimals=True)})()

        # Execute the cut-short strategy.
        strategy.execute(**extra_options)

        # Repair the last reformulation.
        lastReform.successor = oldSuccessor

        # Retrieve and augment the output problem (unless it's self).
        output = lastReform.output
        if output is not self:
            output.prepared_strategy = strategy
            output.prepared_steps = steps

        return output

    def reformulated(self, specification, **extra_options):
        r"""Return the problem reformulated to match a specification.

        Internally this creates a dummy solver accepting problems of the desired
        form and then calls :meth:`prepared` with the dummy solver passed via
        :ref:`option_ad_hoc_solver <option_ad_hoc_solver>`. See meth:`prepared`
        for more details.

        :param specification:
            A problem class that the resulting problem must be a member of.
        :type specification:
            ~picos.modeling.Specification

        :param extra_options:
            Additional solver options to use with this reformulation only.

        :returns:
            The reformulated problem, with ``extra_options`` set unless they
            were "consumed" by a reformulation (e.g.
            :ref:`dualize <option_dualize>`).

        :raises ~picos.modeling.strategy.NoStrategyFound:
            If no reformulation strategy was found.

        :Example:

        >>> from picos import Problem, RealVariable
        >>> from picos.modeling import Specification
        >>> from picos.expressions import AffineExpression
        >>> from picos.constraints import (
        ...     AffineConstraint, SOCConstraint, RSOCConstraint)
        >>> # Define the class/specification of second order conic problems:
        >>> S = Specification(objectives=[AffineExpression],
        ...     constraints=[AffineConstraint, SOCConstraint, RSOCConstraint])
        >>> # Define a quadratic program and reformulate it:
        >>> x = RealVariable("x", 2)
        >>> P = Problem()
        >>> P.set_objective("min", abs(x)**2)
        >>> Q = P.reformulated(S)
        >>> print(P)
        Quadratic Program
          minimize ‖x‖²
          over
            2×1 real variable x
        >>> print(Q)# doctest: +ELLIPSIS
        Second Order Cone Program
          minimize __..._t
          over
            1×1 real variable __..._t
            2×1 real variable x
          subject to
            ‖fullroot(‖x‖²)‖² ≤ __..._t ∧ __..._t ≥ 0

        .. note::

            This method is intended for educational purposes.
            You do not need to use it when solving a problem as PICOS will
            perform the necessary reformulations automatically.
        """
        if not isinstance(specification, Specification):
            raise TypeError("The desired problem type must be given as a "
                "Specification object.")

        # Create a placeholder function for abstract methods of a dummy solver.
        def placeholder(the_self):
            raise RuntimeError("The dummy solver created by "
                "Problem.reformulated must not be executed.")

        # Declare a dummy solver that accepts specified problems.
        DummySolver = type("DummySolver", (Solver,), {
            # Abstract class methods.
            "supports": classmethod(lambda cls, footprint:
                Solver.supports(footprint) and footprint in specification),
            "default_penalty": classmethod(lambda cls: 0),
            "test_availability": classmethod(lambda cls: None),
            "names": classmethod(lambda cls: ("Dummy Solver", "DummySolver",
                "Dummy Solver accepting {}".format(specification), None)),
            "is_free": classmethod(lambda cls: True),

            # Additional class methods needed for an ad-hoc solver.
            "penalty": classmethod(lambda cls, options: 0),

            # Abstract instance methods.
            "reset_problem": lambda self: placeholder(self),
            "_import_problem": lambda self: placeholder(self),
            "_update_problem": lambda self: placeholder(self),
            "_solve": lambda self: placeholder(self)
        })

        # Ad-hoc the dummy solver and prepare the problem for it.
        oldAdHocSolver = self.options.ad_hoc_solver
        extra_options["ad_hoc_solver"] = DummySolver
        problem = self.prepared(**extra_options)

        # Restore the ad_hoc_solver option of the original problem.
        problem.options.ad_hoc_solver = oldAdHocSolver

        return problem

    def solve(self, **extra_options):
        """Hand the problem to a solver.

        You can select the solver manually with the ``solver`` option. Otherwise
        a suitable solver will be selected among those that are available on the
        platform.

        The default behavior (options ``primals=True``, ``duals=None``) is to
        raise a :exc:`~picos.SolutionFailure` when the primal solution is not
        found optimal by the solver, while the dual solution is allowed to be
        missing or incomplete.

        When this method succeeds and unless ``apply_solution=False``, you can
        access the solution as follows:

            - The problem's :attr:`value` denotes the objective function value.
            - The variables' :attr:`~.valuable.Valuable.value` is set according
              to the primal solution. You can in fact query the value of any
              expression involving valued variables like this.
            - The constraints' :attr:`~.constraint.Constraint.dual` is set
              according to the dual solution.
            - The value of any parameter involved in the problem may have
              changed, depending on the parameter.

        :param extra_options:
            A sequence of additional solver options to use with this solution
            search only. In particular, this lets you

            - select a solver via the ``solver`` option,
            - obtain non-optimal primal solutions by setting ``primals=None``,
            - require a complete and optimal dual solution with ``duals=True``,
              and
            - skip valuing variables or constraints with
              ``apply_solution=False``.

        :returns ~picos.Solution or list(~picos.Solution):
            A solution object or list thereof.

        :raises ~picos.SolutionFailure:
            In the following cases:

            1. No solution strategy was found.
            2. Multiple solutions were requested but none were returned.
            3. A primal solution was explicitly requested (``primals=True``) but
               the primal solution is missing/incomplete or not claimed optimal.
            4. A dual solution was explicitly requested (``duals=True``) but
               the dual solution is missing/incomplete or not claimed optimal.

            The case number is stored in the ``code`` attribute of the
            exception.
        """
        from .strategy import NoStrategyFound, Strategy

        startTime = time.time()

        extra_options = map_legacy_options(**extra_options)
        options = self.options.self_or_updated(**extra_options)
        verbose = options.verbosity > 0

        with picos_box(show=verbose):
            if verbose:
                print("Problem type: {}.".format(self.type))

            # Reset an outdated strategy.
            if self._strategy and not self._strategy.valid(**extra_options):
                if verbose:
                    print("Strategy outdated:\n{}.".format(self._strategy))

                self._strategy = None

            # Find a new solution strategy, if necessary.
            if not self._strategy:
                if verbose:
                    if options.ad_hoc_solver:
                        solverName = options.ad_hoc_solver.get_via_name()
                    elif options.solver:
                        solverName = get_solver(options.solver).get_via_name()
                    else:
                        solverName = None

                    print("Searching a solution strategy{}.".format(
                    " for {}".format(solverName) if solverName else ""))

                try:
                    self._strategy = Strategy.from_problem(
                        self, **extra_options)
                except NoStrategyFound as error:
                    s = str(error)

                    if verbose:
                        print(s, flush=True)

                    raise SolutionFailure(1, "No solution strategy found.") \
                        from error

                if verbose:
                    print("Solution strategy:\n  {}".format(
                        "\n  ".join(str(self._strategy).splitlines())))
            else:
                if verbose:
                    print("Reusing strategy:\n  {}".format(
                        "\n  ".join(str(self._strategy).splitlines())))

            # Execute the strategy to obtain one or more solutions.
            solutions = self._strategy.execute(**extra_options)

            # Report how many solutions were obtained, select the first.
            if isinstance(solutions, list):
                assert all(isinstance(s, Solution) for s in solutions)

                if not solutions:
                    raise SolutionFailure(
                        2, "The solver returned an empty list of solutions.")

                solution = solutions[0]

                if verbose:
                    print("Selecting the first of {} solutions obtained for "
                        "processing.".format(len(solutions)))
            else:
                assert isinstance(solutions, Solution)
                solution = solutions

            # Report claimed solution state.
            if verbose:
                print("Solver claims {} solution for {} problem.".format(
                    solution.claimedStatus, solution.problemStatus))

            # Validate the primal solution.
            if options.primals:
                vars_ = self._variables.values()
                if solution.primalStatus != SS_OPTIMAL:
                    raise SolutionFailure(3, "Primal solution state claimed {} "
                        "but optimality is required (primals=True)."
                        .format(solution.primalStatus))
                elif None in solution.primals.values() \
                or any(var not in solution.primals for var in vars_):
                    raise SolutionFailure(3, "The primal solution is incomplete"
                        " but full primals are required (primals=True).")

            # Validate the dual solution.
            if options.duals:
                cons = self._constraints.values()
                if solution.dualStatus != SS_OPTIMAL:
                    raise SolutionFailure(4, "Dual solution state claimed {} "
                        "but optimality is required (duals=True).".format(
                        solution.dualStatus))
                elif None in solution.duals.values() \
                or any(con not in solution.duals for con in cons):
                    raise SolutionFailure(4, "The dual solution is incomplete "
                        "but full duals are required (duals=True).")

            if options.apply_solution:
                if verbose:
                    print("Applying the solution.")

                # Apply the (first) solution.
                solution.apply(snapshotStatus=True)

                # Store all solutions produced by the solver.
                self._last_solution = solutions

                # Report verified solution state.
                if verbose:
                    print("Applied solution is {}.".format(solution.lastStatus))

            endTime = time.time()
            solveTime = endTime - startTime
            searchTime = solution.searchTime

            if searchTime:
                overhead = (solveTime - searchTime) / searchTime
            else:
                overhead = float("inf")

            if verbose:
                print("Search {:.1e}s, solve {:.1e}s, overhead {:.0%}."
                    .format(searchTime, solveTime, overhead))

        if settings.RETURN_SOLUTION:
            return solutions

    def write_to_file(self, filename, writer="picos"):
        """See :func:`picos.modeling.file_out.write`."""
        write(self, filename, writer)

    # --------------------------------------------------------------------------
    # Methods to query the problem.
    # TODO: Document removal of is_complex, is_real (also for constraints).
    # TODO: Revisit #14: "Interfaces to get primal/dual objective values and
    #       primal/dual feasiblity (amount of violation).""
    # --------------------------------------------------------------------------

    def check_current_value_feasibility(self, tol=1e-5, inttol=None):
        """Check if the problem is feasibly valued.

        Checks whether all variables that appear in constraints are valued and
        satisfy both their bounds and the constraints up to the given tolerance.

        :param float tol:
            Largest tolerated absolute violation of a constraint or variable
            bound. If ``None``, then the ``abs_prim_fsb_tol`` solver option is
            used.

        :param inttol:
            DEPRECATED

        :returns:
            A tuple ``(feasible, violation)`` where ``feasible`` is a bool
            stating whether the solution is feasible and ``violation`` is either
            ``None``, if ``feasible == True``, or the amount of violation,
            otherwise.

        :raises picos.uncertain.IntractableWorstCase:
            When computing the worst-case (expected) value of the constrained
            expression is not supported.
        """
        if inttol is not None:
            throw_deprecation_warning("Variable integrality is now ensured on "
                "assignment of a value, so it does not need to be checked via "
                "check_current_value_feasibility's old 'inttol' parameter.")

        if tol is None:
            tol = self._options.abs_prim_fsb_tol

        all_cons = list(self._constraints.values())
        all_cons += [
            variable.bound_constraint for variable in self._variables.values()
            if variable.bound_constraint]

        largest_violation = 0.0

        for constraint in all_cons:
            try:
                slack = constraint.slack
            except IntractableWorstCase as error:
                raise IntractableWorstCase("Failed to check worst-case or "
                    "expected feasibility of {}: {}".format(constraint, error))\
                    from None

            assert isinstance(slack, (float, cvx.matrix, cvx.spmatrix))
            if isinstance(slack, (float, cvx.spmatrix)):
                slack = cvx.matrix(slack)  # Allow min, max.

            # HACK: The following works around the fact that the slack of an
            #       uncertain conic constraint is returned as a vector, even
            #       when the cone is that of the positive semidefinite matrices,
            #       in which case the vectorization used is nontrivial (svec).
            # FIXME: A similar issue should arise when a linear matrix
            #        inequality is integrated in a product cone; The product
            #        cone's slack can then have negative entries but still be
            #        feasible and declared infeasible here.
            # TODO: Add a "violation" interface to Constraint that replaces all
            #       the logic below.
            from ..expressions import Constant, PositiveSemidefiniteCone
            if isinstance(constraint,
                constraints.uncertain.ScenarioUncertainConicConstraint) \
            and isinstance(constraint.cone, PositiveSemidefiniteCone):
                hack = True
                slack = Constant(slack).desvec.safe_value
            else:
                hack = False

            if isinstance(constraint, constraints.LMIConstraint) or hack:
                # Check hermitian-ness of slack.
                violation = float(max(abs(slack - slack.H)))
                if violation > tol:
                    largest_violation = max(largest_violation, violation)

                # Check positive semidefiniteness of slack.
                violation = -float(min(np.linalg.eigvalsh(cvx2np(slack))))
                if violation > tol:
                    largest_violation = max(largest_violation, violation)
            else:
                violation = -float(min(slack))
                if violation > tol:
                    largest_violation = max(largest_violation, violation)

        return (not largest_violation, largest_violation)

    # --------------------------------------------------------------------------
    # Abstract method implementations for the Valuable base class.
    # --------------------------------------------------------------------------

    def _get_valuable_string(self):
        return "problem with {}".format(self._objective._get_valuable_string())

    def _get_value(self):
        return self._objective._get_value()

    # --------------------------------------------------------------------------
    # Legacy methods and properties.
    # --------------------------------------------------------------------------

    _LEGACY_PROPERTY_REASON = "Still used internally by legacy code; will be " \
        "removed together with that code."

    @property
    @deprecated("2.0", reason=_LEGACY_PROPERTY_REASON)
    def countVar(self):
        """The same as :func:`len` applied to :attr:`variables`."""
        return len(self._variables)

    @property
    @deprecated("2.0", reason=_LEGACY_PROPERTY_REASON)
    def countCons(self):
        """The same as :func:`len` applied to :attr:`constraints`."""
        return len(self._variables)

    @property
    @deprecated("2.0", reason=_LEGACY_PROPERTY_REASON)
    def numberOfVars(self):
        """The sum of the dimensions of all referenced variables."""
        return sum(variable.dim for variable in self._variables.values())

    @property
    @deprecated("2.0", reason=_LEGACY_PROPERTY_REASON)
    def numberLSEConstraints(self):
        """Number of :class:`~picos.constraints.LogSumExpConstraint` stored."""
        return len([c for c in self._constraints.values()
            if isinstance(c, constraints.LogSumExpConstraint)])

    @property
    @deprecated("2.0", reason=_LEGACY_PROPERTY_REASON)
    def numberSDPConstraints(self):
        """Number of :class:`~picos.constraints.LMIConstraint` stored."""
        return len([c for c in self._constraints.values()
            if isinstance(c, constraints.LMIConstraint)])

    @property
    @deprecated("2.0", reason=_LEGACY_PROPERTY_REASON)
    def numberQuadConstraints(self):
        """Number of quadratic constraints stored."""
        return len([c for c in self._constraints.values() if isinstance(c, (
            constraints.ConvexQuadraticConstraint,
            constraints.ConicQuadraticConstraint,
            constraints.NonconvexQuadraticConstraint))])

    @property
    @deprecated("2.0", reason=_LEGACY_PROPERTY_REASON)
    def numberConeConstraints(self):
        """Number of quadratic conic constraints stored."""
        return len([c for c in self._constraints.values() if isinstance(
            c, (constraints.SOCConstraint, constraints.RSOCConstraint))])

    @deprecated("2.0", useInstead="value")
    def obj_value(self):
        """Objective function value.

        :raises AttributeError:
            If the problem is a feasibility problem or if the objective function
            is not valued. This is legacy behavior. Note that :attr:`value` just
            returns :obj:`None` while functions that **do** raise an exception
            to denote an unvalued expression would raise
            :exc:`~picos.expressions.NotValued` instead.
        """
        if self._objective.feasibility:
            raise AttributeError(
                "A feasibility problem has no objective value.")

        value = self.value

        if self.value is None:
            raise AttributeError("The objective {} is not fully valued."
                .format(self._objective.function.string))
        else:
            return value

    @deprecated("2.0", useInstead="continuous")
    def is_continuous(self):
        """Whether all variables are of continuous types."""
        return self.continuous

    @deprecated("2.0", useInstead="pure_integer")
    def is_pure_integer(self):
        """Whether all variables are of integral types."""
        return self.pure_integer

    @deprecated("2.0", useInstead="Problem.options")
    def set_all_options_to_default(self):
        """Set all solver options to their default value."""
        self._options.reset()

    @deprecated("2.0", useInstead="Problem.options")
    def set_option(self, key, val):
        """Set a single solver option to the given value.

        :param str key: String name of the option, see below for a list.
        :param val: New value for the option.
        """
        key, val = map_legacy_options({key: val}).popitem()
        self._options[key] = val

    @deprecated("2.0", useInstead="Problem.options")
    def update_options(self, **options):
        """Set multiple solver options at once.

        :param options: A parameter sequence of options to set.
        """
        options = map_legacy_options(**options)
        for key, val in options.items():
            self._options[key] = val

    @deprecated("2.0", useInstead="Problem.options")
    def verbosity(self):
        """Return the problem's current verbosity level."""
        return self._options.verbosity

    @deprecated("2.0", reason="Variables can now be created independent of "
        "problems, and do not need to be added to any problem explicitly.")
    def add_variable(
            self, name, size=1, vtype='continuous', lower=None, upper=None):
        r"""Legacy method to create a PICOS variable.

        :param str name: The name of the variable.

        :param size:
            The shape of the variable.
        :type size:
            anything recognized by :func:`~picos.expressions.data.load_shape`

        :param str vtype:
            Domain of the variable. Can be any of

            - ``'continuous'`` -- real valued,
            - ``'binary'`` -- either zero or one,
            - ``'integer'`` -- integer valued,
            - ``'symmetric'`` -- symmetric matrix,
            - ``'antisym'`` or ``'skewsym'`` -- skew-symmetric matrix,
            - ``'complex'`` -- complex matrix,
            - ``'hermitian'`` -- complex hermitian matrix.

        :param lower:
            A lower bound on the variable.
        :type lower:
            anything recognized by :func:`~picos.expressions.data.load_data`

        :param upper:
            An upper bound on the variable.
        :type upper:
            anything recognized by :func:`~picos.expressions.data.load_data`

        :returns:
            A :class:`~picos.expressions.BaseVariable` instance.

        :Example:

        >>> from picos import Problem
        >>> P = Problem()
        >>> x = P.add_variable("x", 3)
        >>> x
        <3×1 Real Variable: x>
        >>> # Variable are not stored inside the problem any more:
        >>> P.variables
        mappingproxy(OrderedDict())
        >>> # They are only part of the problem if they actually appear:
        >>> P.set_objective("min", abs(x)**2)
        >>> P.variables
        mappingproxy(OrderedDict([('x', <3×1 Real Variable: x>)]))
        """
        if vtype == "continuous":
            return expressions.RealVariable(name, size, lower, upper)
        elif vtype == "binary":
            return expressions.BinaryVariable(name, size)
        elif vtype == "integer":
            return expressions.IntegerVariable(name, size, lower, upper)
        elif vtype == "symmetric":
            return expressions.SymmetricVariable(name, size, lower, upper)
        elif vtype in ("antisym", "skewsym"):
            return expressions.SkewSymmetricVariable(name, size, lower, upper)
        elif vtype == "complex":
            return expressions.ComplexVariable(name, size)
        elif vtype == "hermitian":
            return expressions.HermitianVariable(name, size)
        elif vtype in ("semiint", "semicont"):
            raise NotImplementedError("Variables with legacy types 'semiint' "
                "and 'semicont' are not supported anymore as of PICOS 2.0. "
                "If you need this functionality back, please open an issue.")
        else:
            raise ValueError("Unknown legacy variable type '{}'.".format(vtype))

    @deprecated("2.0", reason="Whether a problem references a variable is now"
        " determined dynamically, so this method has no effect.")
    def remove_variable(self, name):
        """Does nothing."""
        pass

    @deprecated("2.0", useInstead="variables")
    def set_var_value(self, name, value):
        """Set the :attr:`~.valuable.Valuable.value` of a variable.

        For a :class:`Problem` ``P``, this is the same as
        ``P.variables[name] = value``.

        :param str name:
            Name of the variable to be valued.

        :param value:
            The value to be set.
        :type value:
            anything recognized by :func:`~picos.expressions.data.load_data`
        """
        try:
            variable = self._variables[name]
        except KeyError:
            raise KeyError("The problem references no variable named '{}'."
                .format(name)) from None
        else:
            variable.value = value

    @deprecated("2.0", useInstead="dual")
    def as_dual(self):
        """Return the Lagrangian dual problem of the standardized problem."""
        return self.dual


# --------------------------------------
__all__ = api_end(_API_START, globals())
