# ------------------------------------------------------------------------------
# Copyright (C) 2017-2018 Maximilian Stahlberg
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

"""Optimization solver interfaces.

This package contains the interfaces to the optimization solvers that PICOS uses
as its backend. You do not need to instanciate any of the solver classes
directly; if you want to select a particular solver, it is most convenient to
name it to :meth:`~.problem.Problem.solve` via the ``solver`` keyword argument.
"""

from ..apidoc import api_end, api_start
from ..legacy import throw_deprecation_warning


_API_START = api_start(globals())
# -------------------------------

# Import the solver base class and exceptions.
from .solver import (Solver, SolverError, ProblemUpdateError,  # noqa
    OptionError, UnsupportedOptionError,
    ConflictingOptionsError, DependentOptionError, OptionValueError)

# Import all solvers.
from .solver_cplex  import CPLEXSolver  # noqa
from .solver_cvxopt import CVXOPTSolver  # noqa
from .solver_ecos   import ECOSSolver  # noqa
from .solver_glpk   import GLPKSolver  # noqa
from .solver_gurobi import GurobiSolver  # noqa
from .solver_mosek  import MOSEKSolver  # noqa
from .solver_mskfsn import MOSEKFusionSolver  # noqa
from .solver_osqp   import OSQPSolver  # noqa
from .solver_scip   import SCIPSolver  # noqa
from .solver_smcp   import SMCPSolver  # noqa


# Map solver names to their implementation classes.
_solvers = {solver.names()[0]: solver for solver in (
    CPLEXSolver,
    CVXOPTSolver,
    ECOSSolver,
    GLPKSolver,
    GurobiSolver,
    MOSEKSolver,
    MOSEKFusionSolver,
    OSQPSolver,
    SCIPSolver,
    SMCPSolver
)}

# Make sure all solvers inherit from their abstract base class.
assert all(issubclass(solver, Solver) for solver in _solvers.values())


def get_solver(name):
    """Return the implementation class of the solver with the given name."""
    return _solvers[name]


def get_solver_name(solver):
    """Return the registry name of a solver instance."""
    for name, solverClass in _solvers.items():
        if isinstance(solver, solverClass):
            return name
    raise LookupError("The given object's type is not regisered as a solver.")


def all_solvers():
    """Return a dictionary mapping solver names to implementation classes."""
    return _solvers.copy()


def available_solvers(problem=None):
    """Return a sorted list of names of available solvers.

    :param problem: DEPRECATED
    """
    if problem is not None:
        throw_deprecation_warning(
            "Arguments 'problem' to picos.available_solvers is deprecated.")

    return sorted(
        name for name, solver in _solvers.items() if solver.available())


# --------------------------------------
__all__ = api_end(_API_START, globals())
