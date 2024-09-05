# ------------------------------------------------------------------------------
# Copyright (C) 2019-2020 Maximilian Stahlberg
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

"""Optimization solver parameter handling."""

# ------------------------------------------------------------------------------
# NOTE: When modifying tolerance options, be sure to also modify tolerances.rst.
# ------------------------------------------------------------------------------

import fnmatch
import types

from ..apidoc import api_end, api_start
from ..solvers import all_solvers, Solver

_API_START = api_start(globals())
# -------------------------------


OPTIONS = [
    # General options.
    # --------------------------------------------------------------------------

    ("strict_options", bool, False, """
        Whether unsupported general options will raise an
        :class:`~.solver.UnsupportedOptionError` exception, instead of printing
        a warning."""),

    ("verbosity", int, 0, """
        Verbosity level.

        - ``-1`` attempts to suppress all output, even errros.
        - ``0`` only generates warnings and errors.
        - ``1`` generates standard informative output.
        - ``2`` or larger prints additional information for debugging purposes.
        """, lambda n: -1 <= n),

    ("license_warnings", bool, True, """
        Whether solvers are allowed to ignore the :ref:`verbosity
        <option_verbosity>` option to print licensing related warnings.

        See also the global setting :data:`~.settings.LICENSE_WARNINGS`.
        """),

    ("solver", str, None, """
        The solver to use.

        See also the global settings :data:`~.settings.SOLVER_BLACKLIST`,
        :data:`~.settings.SOLVER_WHITELIST` and
        :data:`~.settings.NONFREE_SOLVERS`.

        - :obj:`None` to let PICOS choose.
        - """ + """
        - """.join('``"{0}"`` for :class:`{1} <picos.solvers.{1}>`.'
            .format(name, solver.__name__)
            for name, solver in all_solvers().items()) + """

        This option is ignored when :ref:`ad_hoc_solver <option_ad_hoc_solver>`
        is set.

        .. note::

            :func:`picos.available_solvers() <picos.available_solvers>` returns
            a list of names of solvers that are available at runtime.
        """, lambda value: value is None or value in all_solvers().keys()),

    ("ad_hoc_solver", type, None, """
        The solver to use as a :class:`~.solvers.solver.Solver` subclass.

        This allows solver implementations to be shipped independent of PICOS.

        If set, takes precedence over :ref:`solver <option_solver>`.""",
        lambda value: value is None or issubclass(value, Solver)),

    ("primals", bool, True, """
        Whether to request a primal solution.

        - :obj:`True` will raise an exception if no optimal primal solution is
          found.
        - :obj:`None` will accept and apply also incomplete, infeasible or
          suboptimal primal solutions.
        - :obj:`False` will not ask for a primal solution and throw away any
          primal solution returned by the solver.
        """, None),

    ("duals", bool, None, """
        Whether to request a dual solution.

        - :obj:`True` will raise an exception if no optimal dual solution is
          found.
        - :obj:`None` will accept and apply also incomplete, infeasible or
          suboptimal dual solutions.
        - :obj:`False` will not ask for a dual solution and throw away any
          dual solution returned by the solver.
        """, None),

    ("dualize", bool, False, """
        Whether to dualize the problem as part of the solution strategy.

        This can sometimes lead to a significant solution search speedup.
        """),

    ("assume_conic", bool, True, r"""
        Determines how :class:`~picos.constraints.ConicQuadraticConstraint`
        instances, which correspond to nonconvex constraints of the form
        :math:`x^TQx + p^Tx + q \leq (a^Tx + b)(c^Tx + d)` with
        :math:`x^TQx + p^Tx + q` representable as a squared norm, are processed:

        - :obj:`True` strengthens them into convex conic constraints by assuming
          the additional constraints :math:`a^Tx + b \geq 0` and
          :math:`c^Tx + d \geq 0`.
        - :obj:`False` takes them verbatim and also considers solutions with
          :math:`(a^Tx + b) < 0` or :math:`(c^Tx + d) < 0`. This requires a
          solver that accepts nonconvex quadratic constraints.

        .. warning::

            :class:`~picos.constraints.ConicQuadraticConstraint` are also used
            in the case of :math:`Q = 0`. For instance, :math:`x^2 \geq 1` is
            effectively ransformed to :math:`x \geq 1` if this is :obj:`True`.
        """),

    ("apply_solution", bool, True, """
        Whether to immediately apply the solution returned by a solver to the
        problem's variables and constraints.

        If multiple solutions are returned by the solver, then the first one
        will be applied. If this is ``False``, then solutions can be applied
        manually via their :meth:`~.solution.Solution.apply` method.
        """),

    ("abs_prim_fsb_tol", float, 1e-8, """
        Absolute primal problem feasibility tolerance.

        A primal solution is feasible if some norm over the vector of primal
        constraint violations is smaller than this value.

        :obj:`None` lets the solver use its own default value.
        """, lambda tol: tol is None or tol > 0.0),

    ("rel_prim_fsb_tol", float, 1e-8, """
        Relative primal problem feasibility tolerance.

        Like :ref:`abs_prim_fsb_tol <option_abs_prim_fsb_tol>`, but the norm is
        divided by the norm of the constraints' right hand side vector.

        If the norm used is some nested norm (e.g. the maximum over the norms of
        the equality and inequality violations), then solvers might divide the
        inner violation norms by the respective right hand side inner norms (see
        e.g. `CVXOPT
        <https://cvxopt.org/userguide/coneprog.html#algorithm-parameters>`__).

        To prevent that the right hand side vector norm is zero (or small),
        solvers would either add some small constant or use a fixed lower bound,
        which may be as large as :math:`1`.

        :obj:`None` lets the solver use its own default value.
        """, lambda tol: tol is None or tol > 0.0),

    ("abs_dual_fsb_tol", float, 1e-8, """
        Absolute dual problem feasibility tolerance.

        A dual solution is feasible if some norm over the vector of dual
        constraint violations is smaller than this value.

        Serves as an optimality criterion for the Simplex algorithm.

        :obj:`None` lets the solver use its own default value.
        """, lambda tol: tol is None or tol > 0.0),

    ("rel_dual_fsb_tol", float, 1e-8, """
        Relative dual problem feasibility tolerance.

        Like :ref:`abs_dual_fsb_tol <option_abs_dual_fsb_tol>`, but the norm is
        divided by the norm of the constraints' right hand side vector. (See
        :ref:`rel_prim_fsb_tol <option_rel_prim_fsb_tol>` for exceptions.)

        Serves as an optimality criterion for the Simplex algorithm.

        :obj:`None` lets the solver use its own default value.
        """, lambda tol: tol is None or tol > 0.0),

    ("abs_ipm_opt_tol", float, 1e-8, """
        Absolute optimality tolerance for interior point methods.

        Depending on the solver, a fesible primal/dual solution pair is
        considered optimal if this value upper bounds either

        - the absolute difference between the primal and dual objective values,
          or
        - the violation of the complementary slackness condition.

        The violation is computed as some norm over the vector that contains the
        products of each constraint's slack with its corresponding dual value.
        If the norm is the 1-norm, then the two conditions are equal. Otherwise
        they can differ by a factor that depends on the number and type of
        constraints.

        :obj:`None` lets the solver use its own default value.
        """, lambda tol: tol is None or tol > 0.0),

    ("rel_ipm_opt_tol", float, 1e-8, """
        Relative optimality tolerance for interior point methods.

        Like :ref:`abs_ipm_opt_tol <option_abs_ipm_opt_tol>`, but the
        suboptimality measure is divided by a convex combination of the absolute
        primal and dual objective function values.

        :obj:`None` lets the solver use its own default value.
        """, lambda tol: tol is None or tol > 0.0),

    ("abs_bnb_opt_tol", float, 1e-6, """
        Absolute optimality tolerance for branch-and-bound solution strategies
        to mixed integer problems.

        A solution is optimal if the absolute difference between the objective
        function value of the current best integer solution and the current best
        bound obtained from a continuous relaxation is smaller than this value.

        :obj:`None` lets the solver use its own default value.
        """, lambda tol: tol is None or tol > 0.0),

    ("rel_bnb_opt_tol", float, 1e-4, """
        Relative optimality tolerance for branch-and-bound solution strategies
        to mixed integer problems.

        Like :ref:`abs_bnb_opt_tol <option_abs_bnb_opt_tol>`, but the difference
        is divided by a convex combination of the absolute values of the two
        objective function values.

        :obj:`None` lets the solver use its own default value.
        """, lambda tol: tol is None or tol > 0.0),

    ("integrality_tol", float, 1e-5, r"""
        Integrality tolerance.

        A number :math:`x \in \mathbb{R}` is considered integral if
        :math:`\min_{z \in \mathbb{Z}}{|x - z|}` is at most this value.

        :obj:`None` lets the solver use its own default value.
        """, lambda tol: tol is None or (tol > 0.0 and tol < 0.5)),

    ("markowitz_tol", float, None, """
        Markowitz threshold used in the Simplex algorithm.

        :obj:`None` lets the solver use its own default value.
        """, lambda tol: tol is None or (tol > 0.0 and tol < 1.0)),

    ("max_iterations", int, None, """
        Maximum number of iterations allowed for iterative solution strategies.

        :obj:`None` means no limit.
        """, None),

    ("max_fsb_nodes", int, None, """
        Maximum number of feasible solution nodes visited for branch-and-bound
        solution strategies.

        :obj:`None` means no limit.

        .. note::

            If you want to obtain all feasible solutions that the solver
            encountered, use the :ref:`pool_size <option_pool_size>` option.
        """, None),

    ("timelimit", int, None, """
        Maximum number of seconds spent searching for a solution.

        :obj:`None` means no limit.
        """, None),

    ("lp_root_method", str, None, """
        Algorithm used to solve continuous linear problems, including the root
        relaxation of mixed integer problems.

        - :obj:`None` lets PICOS or the solver select it for you.
        - ``"psimplex"`` for Primal Simplex.
        - ``"dsimplex"`` for Dual Simplex.
        - ``"interior"`` for the interior point method.
        """, lambda value: value in (None, "psimplex", "dsimplex", "interior")),

    ("lp_node_method", str, None, """
        Algorithm used to solve continuous linear problems at non-root nodes of
        the branching tree built when solving mixed integer programs.

        - :obj:`None` lets PICOS or the solver select it for you.
        - ``"psimplex"`` for Primal Simplex.
        - ``"dsimplex"`` for Dual Simplex.
        - ``"interior"`` for the interior point method.
        """, lambda value: value in (None, "psimplex", "dsimplex", "interior")),

    ("treememory", int, None, """
        Bound on the memory used by the branch-and-bound tree, in Megabytes.

        :obj:`None` means no limit.
        """, None),

    ("pool_size", int, None, """
        Maximum number of mixed integer feasible solutions returned.

        If this is not :obj:`None`, :meth:`~.problem.Problem.solve`
        returns a list of :class:`~.solution.Solution` objects instead of just a
        single one.

        :obj:`None` lets the solver return only the best solution.
        """, lambda value: value is None or value >= 1),

    ("pool_rel_gap", float, None, """
        Discards solutions from the :ref:`solution pool <option_pool_size>` as
        soon as a better solution is found that beats it by the given relative
        objective function gap.

        :obj:`None` is the solver's choice, which may be *never discard*.
        """, None),

    ("pool_abs_gap", float, None, """
        Discards solutions from the :ref:`solution pool <option_pool_size>` as
        soon as a better solution is found that beats it by the given absolute
        objective function gap.

        :obj:`None` is the solver's choice, which may be *never discard*.
        """, None),

    ("hotstart", bool, False, """
        Tells the solver to start from the (partial) solution that is stored in
        the :class:`variables <.variables.BaseVariable>` assigned to the
        problem."""),

    ("verify_prediction", bool, True, """
        Whether PICOS should validate that problem reformulations produce a
        problem that matches their predicted outcome.

        If a mismatch is detected, a :class:`RuntimeError` is thrown as there is
        a chance that it is caused by a bug in the reformulation, which could
        affect the correctness of the solution. By disabling this option you are
        able to retrieve a correct solution given that the error is only in the
        prediction, and given that the solution strategy remains valid for the
        actual outcome."""),

    ("max_footprints", int, 1024, """
        Maximum number of different predicted problem formulations (footprints)
        to consider before deciding on a formulation and solver to use.

        :obj:`None` lets PICOS exhaust all reachable problem formulations.
        """, None),

    # Solver-specific options.
    # --------------------------------------------------------------------------

    ("cplex_params", dict, {}, """
        A dictionary of CPLEX parameters to be set after general options are
        passed and before the search is started.

        For example, ``{"mip.limits.cutpasses": 5}`` limits the number of
        cutting plane passes when solving the root node to :math:`5`."""),

    ("cplex_vmconfig", str, None, """
        Load a CPLEX virtual machine configuration file.
        """, None),

    ("cplex_lwr_bnd_limit", float, None, """
        Tells CPLEX to stop MIP optimization if a lower bound below this value
        is found.
        """, None),

    ("cplex_upr_bnd_limit", float, None, """
        Tells CPLEX to stop MIP optimization if an upper bound above this value
        is found.
        """, None),

    ("cplex_bnd_monitor", bool, False, """
        Tells CPLEX to store information about the evolution of the bounds
        during the MIP solution search process. At the end of the computation, a
        list of triples ``(time, lowerbound, upperbound)`` will be provided in
        the field ``bounds_monitor`` of the dictionary returned by
        :meth:`~.problem.Problem.solve`.
        """),

    ("cvxopt_kktsolver", (str, types.FunctionType), None, """
        The KKT solver used by CVXOPT internally.

        See `CVXOPT's guide on exploiting structure
        <https://cvxopt.org/userguide/coneprog.html#exploiting-structure>`_.

        :obj:`None` denotes PICOS' choice: Try first with the faster ``"chol"``,
        then with the more reliable ``"ldl"`` solver.
        """, None),

    ("cvxopt_kktreg", float, 1e-9, """
        The KKT solver regularization term used by CVXOPT internally.

        This is an undocumented feature of CVXOPT, see `here
        <https://github.com/cvxopt/cvxopt/issues/36#issuecomment-125165634>`_.

        End of 2020, this option only affected the LDL KKT solver.

        :obj:`None` denotes CVXOPT's default value.
        """, None),

    ("gurobi_params", dict, {}, """
        A dictionary of Gurobi parameters to be set after general options are
        passed and before the search is started.

        For example, ``{"NodeLimit": 25}`` limits the number of nodes visited by
        the MIP optimizer to :math:`25`."""),

    ("gurobi_matint", bool, None, """
        Whether to use Gurobi's matrix interface.

        This requires Gurobi 9 or later and SciPy.

        :obj:`None` with :data:`~picos.settings.PREFER_GUROBI_MATRIX_INTERFACE`
        enabled means *use it if possible*. :obj:`None` with that setting
        disabled behaves like :obj:`False`.
        """, None),

    ("mosek_params", dict, {}, """
        A dictionary of MOSEK (Optimizer) parameters to be set after general
        options are passed and before the search is started.

        See the `list of MOSEK (Optimizer) 8.1 parameters
        <https://docs.mosek.com/8.1/pythonapi/parameters.html>`_."""),

    ("mosek_server", str, None, """
        Address of a MOSEK remote optimization server to use.

        This option affects both MOSEK (Optimizer) and MOSEK (Fusion).
        """, None),

    ("mosek_basic_sol", bool, False, """
        Return a basic solution when solving LPs with MOSEK (Optimizer).
        """),

    ("mskfsn_params", dict, {}, """
        A dictionary of MOSEK (Fusion) parameters to be set after general
        options are passed and before the search is started.

        See the `list of MOSEK (Fusion) 8.1 parameters
        <https://docs.mosek.com/8.1/pythonfusion/parameters.html>`_."""),

    ("osqp_params", dict, {}, """
        A dictionary of OSQP parameters to be set after general options are
        passed and before the search is started.

        See the `list of OSQP parameters
        <https://osqp.org/docs/interfaces/solver_settings.html>`_."""),

    ("scip_params", dict, {}, """
        A dictionary of SCIP parameters to be set after general options are
        passed and before the search is started.

        For example, ``{"lp/threads": 4}`` sets the number of threads to solve
        LPs with to :math:`4`."""),
]
"""The table of available solver options.

Each entry is a tuple representing a single solver option. The tuple's entries
are, in order:

- Name of the option. Must be a valid Python attribute name.
- The option's argument type. Will be cast on any argument that is not already
  an instance of the type, except for :obj:`None`.
- The option's default value. Must already be of the proper type, or
  :obj:`None`, and must pass the optional check.
- The option's description, which is used as part of the docstring of
  :class:`Options`. In the case of a multi-line text, leading and trailing
  empty lines as well as the overall indentation are ignored.
- Optional: A boolean function used on every argument that passes the type
  conversion (so either an argument of the proper type, or :obj:`None`). If the
  function returns ``False``, then the argument is rejected. The default
  function rejects exactly :obj:`None`. Supplying :obj:`None` instead of a
  function accepts all arguments (in particular, accepts :obj:`None`).
"""

# Add per-solver options.
for name, solver in all_solvers().items():
    OPTIONS.append(("penalty_{}".format(name), float, solver.default_penalty(),
        """
        Penalty for using the {} solver.

        If solver :math:`A` has a penalty of :math:`p` and solver :math:`B` has
        a larger penality of :math:`p + x`, then :math:`B` is be chosen over
        :math:`A` only if the problem as passed to :math:`A` would be
        :math:`10^x` times larger as when passed to :math:`B`.
        """.format(name.upper())))

del name, solver

OPTIONS = sorted(OPTIONS)


class Option():
    """Optimization solver option.

    A single option that affects how a :class:`~.problem.Problem` is solved.

    An initial instance of this class is built from each entry of the
    :data:`OPTIONS` table to obtain the :data:`OPTION_OBJS` tuple.
    """

    # Define __new__ in addition to __init__ so that copy can bypass __init__.
    def __new__(cls, *args, **kwargs):
        """Create a blank :class:`Option` to be filled in by :meth:`copy`."""
        return super(Option, cls).__new__(cls)

    def __init__(self, name, argType, default, description,
            check=(lambda x: x is not None)):
        """Initialize an :class:`Option`.

        See :data:`OPTIONS`.
        """
        assert default is None or isinstance(default, argType)
        assert check is None or check(default)

        self.name        = name
        self.argType     = argType
        self.default     = default
        self._value      = default
        self.description = self._normalize_description(description)
        self.check       = check

    def _normalize_description(self, description):
        lines = description.splitlines()
        notSpace = [n for n, line in enumerate(lines) if line.strip()]
        if not notSpace:
            return ""
        first, last = min(notSpace), max(notSpace)
        i = len(lines[first]) - len(lines[first].lstrip())
        return "\n".join(line[i:].rstrip() for line in lines[first:last+1])

    def _set_value(self, value):
        if value is not None and not isinstance(value, self.argType):
            if isinstance(self.argType, type):
                try:
                    value = self.argType(value)
                except Exception as error:
                    raise TypeError("Failed to convert argument {} to option "
                        "'{}' to type {}.".format(repr(value), self.name,
                        self.argType.__name__)) from error
            else:
                assert isinstance(self.argType, (tuple, list))
                assert all(isinstance(t, type) for t in self.argType)

                raise TypeError("Argument {} to option '{}' does not match "
                    "permissible types {}.".format(repr(value), self.name,
                    ", ".join(t.__name__ for t in self.argType)))

        if self.check is not None and not self.check(value):
            raise ValueError("The option '{}' does not accept the value {}."
                .format(self.name, repr(value)))

        self._value = value

    value = property(lambda self: self._value, _set_value)

    def reset(self):
        """Reset the option to its default value."""
        self.value = self.default

    def is_default(self):
        """Whether the option has its default value."""
        return self.value == self.default

    def copy(self):
        """Return an independent copy of the option."""
        theCopy = self.__class__.__new__(self.__class__)
        theCopy.name        = self.name
        theCopy.argType     = self.argType
        theCopy.default     = self.default
        theCopy._value      = self._value
        theCopy.description = self.description
        theCopy.check       = self.check
        return theCopy


OPTION_OBJS = tuple(Option(*args) for args in OPTIONS)
"""The initial solver options as :class:`Option` objects."""


def _tablerow(option, indentaion=0):
    """Return a reST list-table row describing an :class:`Option`."""
    spaces = " "*indentaion
    return (
        "{}- * {{0}}\n"
        "{}  * ``{{1}}``\n"
        "{}  * .. _option_{{0}}:\n\n"
        "{}    {{2}}"
    ).format(
        *(4*(spaces,))).format(option.name, repr(option.default),
        "\n{}    ".format(spaces).join(option.description.splitlines()))


def _jumplabel(option):
    """Return a reStructuredText jumplabel describing an :class:`Option`."""
    return ":ref:`{0} <option_{0}>`".format(option.name)


class Options():
    """Collection of optimization solver options.

    A collection of options that affect how a :class:`~.problem.Problem` is
    solved. :attr:`Problem.options <.problem.Problem.options>` is an instance of
    this class.

    The options can be accessed as an attribute or as an item. The latter
    approach supports Unix shell-style wildcard characters:

    >>> import picos
    >>> P = picos.Problem()
    >>> P.options.verbosity = 2
    >>> P.options["primals"] = False
    >>> # Set all absolute tolerances at once.
    >>> P.options["abs_*_tol"] = 1e-6

    There are two corresponding ways to reset an option to its default value:

    >>> del P.options.verbosity
    >>> P.options.reset("primals", "*_tol")

    Options can also be passed as a keyword argument sequence when the
    :class:`Problem <picos.Problem>` is created and whenever a solution is
    searched:

    >>> # Use default options except for verbosity.
    >>> P = picos.Problem(verbosity = 1)
    >>> x = P.add_variable("x", lower = 0); P.set_objective("min", x)
    >>> # Only for the next search: Don't be verbose anyway.
    >>> solution = P.solve(solver = "cvxopt", verbosity = 0)
    """

    # Document the individual options.
    __doc__ += \
    """
    .. rubric:: Available Options

    Jump to option: ➥\xa0{}

    .. list-table::
      :header-rows: 1
      :widths: 10 10 80

      - * Option
        * Default
        * Description
    """.format(" ➥\xa0".join(_jumplabel(option) for option in OPTION_OBJS)) \
    .rstrip() + "\n" + "\n".join(_tablerow(option, 6) for option in OPTION_OBJS)

    # Define __new__ in addition to __init__ so that
    # 1. __init__ does not take the static default options as an argument,
    #    hiding them from the user and the documentation while
    # 2. Options.copy can still bypass copying the default options (by bypassing
    #    __init__) so that options aren't copied twice.
    def __new__(cls, *args, **kwargs):
        """Create an empty options set."""
        instance = super(Options, cls).__new__(cls)
        # Options overwrites __setattr__, so we need to call object.__setattr__.
        super(Options, cls).__setattr__(instance, "_options", {})
        return instance

    def __init__(self, **options):
        """Create a default option set and set the given options on top."""
        for option in OPTION_OBJS:
            self._options[option.name] = option.copy()

        self.update(**options)

    def __str__(self):
        defaults = sorted(
            (o for o in self._options.values() if o.is_default()),
            key=(lambda o: o.name))
        modified = sorted(
            (o for o in self._options.values() if not o.is_default()),
            key=(lambda o: o.name))

        nameLength  = max(len(o.name)       for o in self._options.values())
        valueLength = max(len(str(o.value)) for o in self._options.values())

        string = ""

        if modified:
            defaultLength = max(len(str(o.default)) for o in modified)

            string += "Modified solver options:\n" + "\n".join((
                "  {{:{}}} = {{:{}}} (default: {{:{}}})".format(
                    nameLength, valueLength, defaultLength
                ).format(
                    option.name, str(option.value), str(option.default))
                for num, option in enumerate(modified)))

        if defaults:
            if modified:
                string += "\n\n"

            string += "Default solver options:\n" + "\n".join((
                "  {{:{}}} = {{}}".format(nameLength).format(
                    option.name, str(option.value))
                for num, option in enumerate(defaults)))

        return string

    def __eq__(self, other):
        """Report whether two sets of options are equal."""
        if self is other:
            return True

        for name in self._options:
            if self._options[name].value != other._options[name].value:
                return False

        return True

    def _fuzzy(returnsSomething):
        """Allow wildcards in option names."""
        def decorator(method):
            def wrapper(self, pattern, *extraArgs):
                if any(char in pattern for char in "*?[!]"):
                    matching = fnmatch.filter(self._options.keys(), pattern)

                    if not matching:
                        raise LookupError("No option matches '{}'."
                            .format(pattern))

                    if returnsSomething:
                        return {name: method(self, name, *extraArgs)
                            for name in matching}
                    else:
                        for name in matching:
                            method(self, name, *extraArgs)
                else:
                    if returnsSomething:
                        return method(self, pattern, *extraArgs)
                    else:
                        method(self, pattern, *extraArgs)
            return wrapper
        return decorator

    @_fuzzy(True)
    def __getattr__(self, name):
        if name in self._options:
            return self._options[name].value
        else:
            raise AttributeError("Unknown option '{}'.".format(name))

    @_fuzzy(False)
    def __setattr__(self, name, value):
        if name in self._options:
            self._options[name].value = value
        else:
            raise AttributeError("Unknown option '{}'.".format(name))

    @_fuzzy(False)
    def __delattr__(self, name):
        if name in self._options:
            self._options[name].reset()
        else:
            raise AttributeError("Unknown option '{}'.".format(name))

    @_fuzzy(True)
    def __getitem__(self, name):
        if name in self._options:
            return self._options[name].value
        else:
            raise LookupError("Unknown option '{}'.".format(name))

    @_fuzzy(False)
    def __setitem__(self, name, value):
        if name in self._options:
            self._options[name].value = value
        else:
            raise LookupError("Unknown option '{}'.".format(name))

    def __contains__(self, name):
        return name in self._options

    def __dir__(self):
        optionNames = [name for name in self._options.keys()]
        list_ = super(Options, self).__dir__() + optionNames
        return sorted(list_)

    def copy(self):
        """Return an independent copy of the current options set."""
        theCopy = self.__class__.__new__(self.__class__)
        for option in self._options.values():
            theCopy._options[option.name] = option.copy()
        return theCopy

    def update(self, **options):
        """Set multiple options at once.

        This method is called with the keyword arguments supplied to the
        :class:`Options` constructor, so the following two are the same:

        >>> import picos
        >>> a = picos.Options(verbosity = 1, primals = False)
        >>> b = picos.Options()
        >>> b.update(verbosity = 1, primals = False)
        >>> a == b
        True

        :param options: A parameter sequence of options to set.
        """
        for key, val in options.items():
            self[key] = val

    def updated(self, **options):
        """Return a modified copy."""
        theCopy = self.copy()
        if options:
            theCopy.update(**options)
        return theCopy

    def self_or_updated(self, **options):
        """Return either a modified copy or self, depending on given options."""
        if options:
            theCopy = self.copy()
            theCopy.update(**options)
            return theCopy
        else:
            return self

    @_fuzzy(False)
    def _reset_single(self, name):
        self._options[name].reset()

    def reset(self, *options):
        """Reset all or a selection of options to their default values.

        :param options: The names of the options to reset, may contain wildcard
            characters. If no name is given, all options are reset.
        """
        if options:
            for name in options:
                self._reset_single(name)
        else:
            for option in self._options.values():
                option.reset()

    @_fuzzy(True)
    def _help_single(self, name):
        option = self._options[name]
        return (
            "Option:  {}\n"
            "Default: {}\n"
            "\n  {}"
        ).format(option.name, str(option.default),
            "\n  ".join(option.description.splitlines()))

    def help(self, *options):
        """Print text describing selected options.

        :param options: The names of the options to describe, may contain
            wildcard characters.
        """
        for i, name in enumerate(options):
            if i != 0:
                print("\n\n")
            retval = self._help_single(name)
            if isinstance(retval, str):
                print(retval)
            else:
                assert isinstance(retval, dict)
                print("\n\n".join(retval.values()))

    @property
    def nondefaults(self):
        """A dictionary mapping option names to nondefault values.

        :Example:

        >>> from picos import Options
        >>> o = Options()
        >>> o.verbosity = 2
        >>> o.nondefaults
        {'verbosity': 2}
        >>> Options(**o.nondefaults) == o
        True
        """
        return {name: option._value for name, option in self._options.items()
            if option._value != option.default}


# --------------------------------------
__all__ = api_end(_API_START, globals())
