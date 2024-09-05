# ------------------------------------------------------------------------------
# Copyright (C) 2021-2022 Maximilian Stahlberg
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

"""Implementation of :class:`OSQPSolver`."""

import cvxopt
import numpy

from ..apidoc import api_end, api_start
from ..constraints import AffineConstraint, DummyConstraint
from ..expressions import (CONTINUOUS_VARTYPES, AffineExpression,
                           QuadraticExpression)
from ..expressions.data import cvx2csc
from ..modeling.footprint import Specification
from ..modeling.solution import (PS_FEASIBLE, PS_INFEASIBLE, PS_UNBOUNDED,
                                 PS_UNKNOWN, SS_FAILURE, SS_INFEASIBLE,
                                 SS_OPTIMAL, SS_PREMATURE, SS_UNKNOWN)
from .solver import Solver

_API_START = api_start(globals())
# -------------------------------


class OSQPSolver(Solver):
    """Interface to the OSQP solver."""

    SUPPORTED = Specification(
        objectives=[
            AffineExpression,
            QuadraticExpression],
        variables=CONTINUOUS_VARTYPES,
        constraints=[
            DummyConstraint,
            AffineConstraint])

    @classmethod
    def supports(cls, footprint, explain=False):
        """Implement :meth:`~.solver.Solver.supports`."""
        result = Solver.supports(footprint, explain)
        if not result or (explain and not result[0]):
            return result

        if footprint.nonconvex_quadratic_objective:
            if explain:
                return (False, "QPs with a nonconvex objective.")
            else:
                return False

        if footprint not in cls.SUPPORTED:
            if explain:
                return False, cls.SUPPORTED.mismatch_reason(footprint)
            else:
                return False

        return (True, None) if explain else True

    @classmethod
    def default_penalty(cls):
        """Implement :meth:`~.solver.Solver.default_penalty`."""
        # OSQP is an established free/open source solver but it has issues with
        # moderate numbers of affine inequalities, so leave it to user choice.
        return 2.0

    @classmethod
    def test_availability(cls):
        """Implement :meth:`~.solver.Solver.test_availability`."""
        cls.check_import("osqp")

    @classmethod
    def names(cls):
        """Implement :meth:`~.solver.Solver.names`."""
        return "osqp", "OSQP", "Operator Splitting QP Solver", None

    @classmethod
    def is_free(cls):
        """Implement :meth:`~.solver.Solver.is_free`."""
        return True

    def __init__(self, problem):
        """Initialize a OSQP solver interface.

        :param ~picos.Problem problem: The problem to be solved.
        """
        super(OSQPSolver, self).__init__(problem)

        self._numVars = 0
        """Total number of scalar variables passed to OSQP."""

        self._osqpVarOffset = {}
        """Maps a PICOS variable to its column in the constraint matrix."""

        self._osqpConOffset = {}
        """Maps a PICOS constraint to its row in the constraint matrix."""

        self._objectiveOffset = 0.0
        """Objective function constant offset."""

    def reset_problem(self):
        """Implement :meth:`~.solver.Solver.reset_problem`."""
        self.int = None

        self._numVars = 0
        self._osqpVarOffset.clear()
        self._osqpConOffset.clear()
        self._objectiveOffset = 0.0

    def _affine_expression_to_G_and_h(self, expression):
        assert isinstance(expression, AffineExpression)

        return expression.scipy_sparse_matrix_form(
            varOffsetMap=self._osqpVarOffset, dense_b=True)

    _Gh = _affine_expression_to_G_and_h

    def _import_variables(self):
        offset = 0
        for variable in self.ext.variables.values():
            dim = variable.dim

            # Register the variable.
            self._osqpVarOffset[variable] = offset
            offset += dim

        assert offset == self._numVars

        # Add variable bounds as affine constraints.
        for variable in self.ext.variables.values():
            # TODO: Import lower and upper bound in a single constraint instead.
            bounds = variable.bound_constraint
            if bounds:
                self._import_affine_constraint(bounds)

    def _import_objective(self):
        direction, objective = self.ext.no

        # OSQP only supports minimization; flip the sign for maximization.
        if direction == "max":
            objective = -objective

        # Split objective into quadratic and affine part.
        if isinstance(objective, AffineExpression):
            sparse_quads = {}
            affine_part = objective
        elif isinstance(objective, QuadraticExpression):
            sparse_quads = objective._sparse_quads
            affine_part = objective.aff
        else:
            assert False, "Unexpected objective."

        # Import quadratic part.
        for xy, Q in sparse_quads.items():
            x, y = xy
            m, n = x.dim, y.dim

            dx = self._osqpVarOffset[x]
            dy = self._osqpVarOffset[y]

            # OSQP reads only the upper triangular part.
            if dx > dy:
                dx, dy = dy, dx
                m, n = n, m
                Q = Q.T

            # OSQP adds a factor of 0.5; cancel it.
            Q = 2*Q

            # Convert from cvxopt sparse to scipy sparse matrix.
            Q = cvx2csc(Q)

            self.int["P"][dx:dx + m, dy:dy + n] = Q

        # Import linear part.
        q, c = self._Gh(affine_part)

        self.int["q"] = numpy.ravel(q.todense())
        self._objectiveOffset = float(c[0])

    def _import_affine_constraint(self, constraint):
        from scipy import sparse

        assert isinstance(constraint, AffineConstraint)

        A, minus_b = self._Gh(constraint.lmr)
        b = -minus_b

        if constraint.is_equality():
            l = u = b
        elif constraint.is_increasing():
            l = numpy.full(len(b), float("-inf"))
            u = b
        elif constraint.is_decreasing():
            l = b
            u = numpy.full(len(b), float("+inf"))

        self._osqpConOffset[constraint] = len(self.int["l"])

        # TODO: Add matrices to a list and concatenate them at once later.
        self.int["A"] = sparse.vstack([self.int["A"], A], format="csc")
        self.int["l"] = numpy.concatenate([self.int["l"], l])
        self.int["u"] = numpy.concatenate([self.int["u"], u])

    def _import_constraint(self, constraint):
        if isinstance(constraint, AffineConstraint):
            self._import_affine_constraint(constraint)
        else:
            assert isinstance(constraint, DummyConstraint), \
                "Unexpected constraint type: {}".format(
                constraint.__class__.__name__)

    def _import_problem(self):
        from scipy import sparse

        self._numVars = n = sum(var.dim for var in self.ext.variables.values())

        # OSQP's internal problem representation is stateful but supports
        # updates only by setting or replacing whole matrices and not on a
        # per-variable or per-constraint basis. We thus pretend it was stateless
        # and use the osqp.solve function in a similar manner as with CVXOPT.
        # TODO: Consider supporting the limited update capabilities of OSQP.
        self.int = {
            # Objective function quadratic form.
            # NOTE: lil_matrix for cheap updates, converted to csc_matrix later.
            "P": sparse.lil_matrix((n, n)),

            # Objective function linear coefficients.
            "q": numpy.zeros(n),

            # Linear inequality coefficient matrix.
            "A": sparse.csc_matrix((0, n)),

            # Linear inequality lower bound.
            "l": numpy.zeros(0),

            # Linear inequality upper bound.
            "u": numpy.zeros(0),
        }

        # Import variables without their bounds.
        self._import_variables()

        # Set objective.
        self._import_objective()

        # Import constraints.
        for constraint in self.ext.constraints.values():
            self._import_constraint(constraint)

        # Convert from LIL to CSC manually to avoid a warning by OSQP.
        self.int["P"] = self.int["P"].tocsc()

    def _update_problem(self):
        raise NotImplementedError

    def _solve(self):
        import osqp

        options = {}

        # verbosity
        options["verbose"] = (self.verbosity() >= 1)

        # abs_prim_fsb_tol
        if self.ext.options.abs_prim_fsb_tol is not None:
            options["eps_prim_inf"] = self.ext.options.abs_prim_fsb_tol

        # abs_dual_fsb_tol
        if self.ext.options.abs_dual_fsb_tol is not None:
            options["eps_dual_inf"] = self.ext.options.abs_dual_fsb_tol

        # abs_ipm_opt_tol
        if self.ext.options.abs_ipm_opt_tol is not None:
            options["eps_abs"] = self.ext.options.abs_ipm_opt_tol

        # rel_ipm_opt_tol
        if self.ext.options.rel_ipm_opt_tol is not None:
            options["eps_rel"] = self.ext.options.rel_ipm_opt_tol

        # max_iterations
        # NOTE: OSQP can hang long already for moderately sized LPs but we still
        #       "remove" the iteration limit to obey the PICOS setting. This is
        #       fine as long as OSQP requires user selection.
        if self.ext.options.max_iterations is not None:
            options["max_iter"] = self.ext.options.max_iterations
        else:
            options["max_iter"] = int(1e9)

        # timelimit
        if self.ext.options.timelimit is not None:
            options["time_limit"] = self.ext.options.timelimit

        # Enable polishing to increase chance of obeying precision limits.
        options["polish"] = True

        # Handle OSQP-specific options.
        options.update(self.ext.options.osqp_params)

        # Handle unsupported options.
        # TODO: Support hotstart.
        self._handle_unsupported_options("lp_root_method", "lp_node_method",
            "treememory", "max_fsb_nodes", "hotstart")

        # Attempt to solve the problem.
        with self._header(), self._stopwatch():
            # NOTE: There is supposed to be a direct function osqp.solve but it
            #       does not exist for my installation of 0.6.2.
            # result = osqp.solve(**self.int, **options)

            model = osqp.OSQP()
            model.setup(**self.int, **options)

            try:
                result = model.solve()
            except ValueError as error:
                if str(error) == "OSQP solve error!":
                    result = None
                else:
                    raise

        # Retrieve primals.
        primals = {}
        if result and self.ext.options.primals is not False:
            for variable in self.ext.variables.values():
                offset = self._osqpVarOffset[variable]
                primal = list(result.x[offset:offset + variable.dim])

                if None in primal:
                    primal = None
                else:
                    primal = cvxopt.matrix(primal)

                primals[variable] = primal

        # Retrieve duals.
        duals = {}
        if result and self.ext.options.duals is not False:
            for constraint in self.ext.constraints.values():
                if isinstance(constraint, DummyConstraint):
                    duals[constraint] = cvxopt.spmatrix(
                        [], [], [], constraint.size)
                    continue

                assert isinstance(constraint, AffineConstraint)

                offset = self._osqpConOffset[constraint]
                length = len(constraint)
                dual = list(result.y[offset:offset + length])

                if None in dual:
                    dual = None
                else:
                    dual = cvxopt.matrix(dual)

                    if not constraint.is_increasing():
                        dual = -dual

                duals[constraint] = dual

        # Retrieve objective value.
        value = result.info.obj_val if result else None
        if value is not None:
            # Add back the constant part.
            value += self._objectiveOffset

            # Flip back the sign for maximization.
            if self.ext.no.direction == "max":
                value = -value

        # Retrieve solution status.
        status = result.info.status if result else None
        if status is None:
            primalStatus = SS_FAILURE
            dualStatus = SS_FAILURE
            problemStatus = PS_UNKNOWN
        elif status == "solved":
            primalStatus = SS_OPTIMAL
            dualStatus = SS_OPTIMAL
            problemStatus = PS_FEASIBLE
        elif status == "primal infeasible":
            primalStatus = SS_INFEASIBLE
            dualStatus = SS_UNKNOWN
            problemStatus = PS_INFEASIBLE
        elif status == "dual infeasible":
            primalStatus = SS_UNKNOWN
            dualStatus = SS_INFEASIBLE
            problemStatus = PS_UNBOUNDED
        elif status in ("maximum iterations reached", "run time limit reached"):
            primalStatus = SS_PREMATURE
            dualStatus = SS_PREMATURE
            problemStatus = PS_UNKNOWN
        else:
            assert False, "Unknown solver status '{}'".format(status)

        return self._make_solution(
            value, primals, duals, primalStatus, dualStatus, problemStatus,
            {"osqp_info": result.info if result else None})


# --------------------------------------
__all__ = api_end(_API_START, globals())
