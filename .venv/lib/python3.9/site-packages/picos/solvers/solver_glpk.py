# ------------------------------------------------------------------------------
# Copyright (C) 2017-2019 Maximilian Stahlberg
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

"""Implementation of :class:`GLPKSolver`."""

from collections import namedtuple

import cvxopt

from ..apidoc import api_end, api_start
from ..constraints import AffineConstraint, DummyConstraint
from ..expressions import AffineExpression, BinaryVariable, IntegerVariable
from ..modeling.footprint import Specification
from ..modeling.solution import (PS_FEASIBLE, PS_INFEASIBLE, PS_UNBOUNDED,
                                 PS_UNKNOWN, SS_EMPTY, SS_FEASIBLE,
                                 SS_INFEASIBLE, SS_OPTIMAL, SS_UNKNOWN)
from .solver import Solver

_API_START = api_start(globals())
# -------------------------------


class GLPKSolver(Solver):
    """Interface to the GLPK solver via swiglpk."""

    SUPPORTED = Specification(
        objectives=[
            AffineExpression],
        constraints=[
            DummyConstraint,
            AffineConstraint])

    UNUSED_VAR = namedtuple("UnusedVar", ("dim",))(dim=1)

    @classmethod
    def supports(cls, footprint, explain=False):
        """Implement :meth:`~.solver.Solver.supports`."""
        result = Solver.supports(footprint, explain)
        if not result or (explain and not result[0]):
            return result

        if footprint not in cls.SUPPORTED:
            if explain:
                return False, cls.SUPPORTED.mismatch_reason(footprint)
            else:
                return False

        return (True, None) if explain else True

    @classmethod
    def default_penalty(cls):
        """Implement :meth:`~.solver.Solver.default_penalty`."""
        return 1.0  # Stable free/open source solver.

    @classmethod
    def test_availability(cls):
        """Implement :meth:`~.solver.Solver.test_availability`."""
        cls.check_import("swiglpk")

    @classmethod
    def names(cls):
        """Implement :meth:`~.solver.Solver.names`."""
        return "glpk", "GLPK", "GNU Linear Programming Kit", "swiglpk"

    @classmethod
    def is_free(cls):
        """Implement :meth:`~.solver.Solver.is_free`."""
        return True

    def __init__(self, problem):
        """Initialize a GLPK solver interface.

        :param ~picos.Problem problem: The problem to be solved.
        """
        self._glpkVarOffset = {}
        self._glpkConOffset = {}

        super(GLPKSolver, self).__init__(problem)

    def __del__(self):
        try:
            import swiglpk as glpk
        except ImportError:
            # Happens when python is shutting down. In this case, no garbage
            # collection is necessary, anyway.
            return

        if self.int is not None:
            glpk.glp_delete_prob(self.int)

    def reset_problem(self):
        """Implement :meth:`~.solver.Solver.reset_problem`."""
        import swiglpk as glpk

        self._glpkVarOffset.clear()
        self._glpkConOffset.clear()

        if self.int is not None:
            glpk.glp_delete_prob(self.int)
            self.int = None

    def _import_variable(self, picosVar):
        import swiglpk as glpk

        p = self.int
        dim = picosVar.dim

        # Append columns to the constraint matrix.
        offset = glpk.glp_add_cols(p, dim)

        # Retrieve the type.
        if isinstance(picosVar, IntegerVariable):
            varType = glpk.GLP_IV
        elif isinstance(picosVar, BinaryVariable):
            varType = glpk.GLP_BV
        else:
            varType = glpk.GLP_CV

        # Retrieve bounds.
        if not isinstance(picosVar, BinaryVariable):
            boundKeys   = [glpk.GLP_FR]*dim
            lowerBounds = [0]*dim
            upperBounds = [0]*dim

            lower, upper = picosVar.bound_dicts

            for i, b in lower.items():
                boundKeys[i] = glpk.GLP_LO
                lowerBounds[i] = b

            for i, b in upper.items():
                if boundKeys[i] == glpk.GLP_FR:
                    boundKeys[i] = glpk.GLP_UP
                else:  # Also has a lower bound.
                    if lowerBounds[i] == b:
                        boundKeys[i] = glpk.GLP_FX
                    else:
                        boundKeys[i] = glpk.GLP_DB

                upperBounds[i] = b

        # Import scalar variables.
        for i in range(picosVar.dim):
            glpkIndex = offset + i

            # Assign a name.
            glpk.glp_set_col_name(
                p, glpkIndex, "{}_{}".format(picosVar.name, i))

            # Set the type.
            glpk.glp_set_col_kind(p, glpkIndex, varType)

            # Assign the bounds.
            if not isinstance(picosVar, BinaryVariable):
                glpk.glp_set_col_bnds(
                    p, glpkIndex, boundKeys[i], lowerBounds[i], upperBounds[i])

        self._glpkVarOffset[picosVar] = offset

    def _remove_variable(self, picosVar):
        import swiglpk as glpk

        dim = picosVar.dim

        offset = self._glpkVarOffset.pop(picosVar)

        glpkVars = glpk.intArray(dim + 1)
        for i in range(dim):
            glpkVars[i + 1] = offset + i  # Index 0 is unused.

        glpk.glp_del_cols(self.int, dim, glpkVars)

        for otherVar, otherOffset in self._glpkVarOffset.items():
            if otherOffset > offset:
                self._glpkVarOffset[otherVar] -= dim

    def _import_constraint(self, picosCon):
        import swiglpk as glpk

        p = self.int

        # Append rows to the constraint matrix.
        rowOffset = glpk.glp_add_rows(p, len(picosCon))

        # Import scalar constraints.
        lmr_rows = picosCon.lmr.sparse_rows(self._glpkVarOffset)
        for localConIndex, (glpkVarIndices, coefs, c) in enumerate(lmr_rows):
            rhs          = -c
            glpkConIndex = rowOffset + localConIndex
            numColumns   = len(glpkVarIndices)

            # Assign a name.
            glpk.glp_set_row_name(
                p, glpkConIndex, "{}_{}".format(picosCon.id, localConIndex))

            # Set the constant term.
            if picosCon.is_equality():
                glpk.glp_set_row_bnds(p, glpkConIndex, glpk.GLP_FX, rhs, rhs)
            elif picosCon.is_increasing():
                glpk.glp_set_row_bnds(p, glpkConIndex, glpk.GLP_UP, 0, rhs)
            elif picosCon.is_decreasing():
                glpk.glp_set_row_bnds(p, glpkConIndex, glpk.GLP_LO, rhs, 0)
            else:
                assert False, "Unexpected constraint relation."

            # Set coefficients.
            # NOTE: GLPK requires a glpk.intArray containing column indices and
            #       a glpk.doubleArray of same size containing the coefficients
            #       for the listed column index. The first element of both
            #       arrays (with index 0) is skipped by GLPK.
            glpkVarIndicesArray = glpk.intArray(numColumns + 1)
            for i in range(numColumns):
                glpkVarIndicesArray[i + 1] = glpkVarIndices[i]

            coefficientsArray = glpk.doubleArray(numColumns + 1)
            for i in range(numColumns):
                coefficientsArray[i + 1] = coefs[i]

            glpk.glp_set_mat_row(p, glpkConIndex, numColumns,
                glpkVarIndicesArray, coefficientsArray)

        self._glpkConOffset[picosCon] = rowOffset

    def _remove_constraint(self, picosCon):
        import swiglpk as glpk

        length = len(picosCon)

        offset = self._glpkConOffset.pop(picosCon)

        glpkCons = glpk.intArray(length + 1)
        for i in range(length):
            glpkCons[i + 1] = offset + i  # Index 0 is unused.

        glpk.glp_del_rows(self.int, length, glpkCons)

        for otherCon, otherOffset in self._glpkConOffset.items():
            if otherOffset > offset:
                self._glpkConOffset[otherCon] -= length

    def _import_objective(self):
        import swiglpk as glpk

        p = self.int
        direction, objective = self.ext.no

        # Set optimization direction.
        if direction == "min":
            glpk.glp_set_obj_dir(p, glpk.GLP_MIN)
        else:
            assert direction == "max"
            glpk.glp_set_obj_dir(p, glpk.GLP_MAX)

        # Set objective function shift (index 0).
        glpk.glp_set_obj_coef(p, 0, objective._constant_coef[0])

        # Set objective function coefficient of the scalar variable.
        for picosVar, picosCoef in objective._linear_coefs.items():
            for localIndex in range(picosVar.dim):
                if picosCoef[localIndex]:
                    glpkIndex = self._glpkVarOffset[picosVar] + localIndex
                    glpk.glp_set_obj_coef(p, glpkIndex, picosCoef[localIndex])

    def _reset_objective(self):
        import swiglpk as glpk

        p = self.int

        # Zero the objective.
        scalarVars = glpk.glp_get_num_cols(p)
        for i in range(scalarVars + 1):  # Index 0 refers to the constant term.
            glpk.glp_set_obj_coef(p, i, 0)

    def _import_problem(self):
        import swiglpk as glpk

        if self.verbosity() >= 1:
            glpk.glp_term_out(glpk.GLP_ON)
        else:
            glpk.glp_term_out(glpk.GLP_OFF)

        # Create a problem instance.
        self.int = glpk.glp_create_prob()

        # Add a dummy variable since index zero is unused.
        # This is necessary for ComplexAffineExpression.sparse_row to work.
        self._glpkVarOffset[self.UNUSED_VAR] = 0

        # Import variables.
        for variable in self.ext.variables.values():
            self._import_variable(variable)

        # Import constraints.
        for constraint in self.ext.constraints.values():
            if not isinstance(constraint, DummyConstraint):
                self._import_constraint(constraint)

        # Set objective.
        self._import_objective()

    def _update_problem(self):
        import swiglpk as glpk

        resetBasis = False

        for oldConstraint in self._removed_constraints():
            self._remove_constraint(oldConstraint)
            resetBasis = True

        for oldVariable in self._removed_variables():
            self._remove_variable(oldVariable)
            resetBasis = True

        for newVariable in self._new_variables():
            self._import_variable(newVariable)

        for newConstraint in self._new_constraints():
            self._import_constraint(newConstraint)

        if self._objective_has_changed():
            self._reset_objective()
            self._import_objective()

        if resetBasis:
            # TODO: Repair the basis in _remove_constraint, _remove_variable.
            glpk.glp_cpx_basis(self.int)

    def _solve(self):
        import swiglpk as glpk

        p = self.int

        continuous = self.ext.is_continuous()
        minimizing = glpk.glp_get_obj_dir(p) == glpk.GLP_MIN

        # Select LP solver (Simplex or Interior Point Method).
        if continuous:
            if self.ext.options.lp_root_method == "interior":
                interior = True
            else:
                # Default to Simplex.
                interior = False
            simplex = not interior
        else:
            simplex = interior = False

        # Select appropriate options container.
        if simplex:
            options = glpk.glp_smcp()
            glpk.glp_init_smcp(options)
        elif interior:
            options = glpk.glp_iptcp()
            glpk.glp_init_iptcp(options)
        else:
            options = glpk.glp_iocp()
            glpk.glp_init_iocp(options)

        # verbosity
        verbosity = self.verbosity()
        if verbosity < 0:
            options.msg_lev = glpk.GLP_MSG_OFF
        elif verbosity == 0:
            options.msg_lev = glpk.GLP_MSG_ERR
        elif verbosity == 1:
            options.msg_lev = glpk.GLP_MSG_ON
        elif verbosity >= 2:
            options.msg_lev = glpk.GLP_MSG_ALL

        # abs_prim_fsb_tol
        if self.ext.options.abs_prim_fsb_tol is not None:
            options.tol_bnd = self.ext.options.abs_prim_fsb_tol

        # abs_dual_fsb_tol
        if self.ext.options.abs_dual_fsb_tol is not None:
            options.tol_dj = self.ext.options.abs_dual_fsb_tol

        # rel_bnb_opt_tol
        # Note that the option is silently ignored if passed alongside an LP;
        # while the solver does not allow us to pass the option in that case, it
        # is still technically a valid option as every LP is also a MIP.
        if self.ext.options.rel_bnb_opt_tol is not None:
            if not continuous:
                options.mip_gap = self.ext.options.rel_bnb_opt_tol

        # max_iterations
        if not simplex:
            self._handle_unsupported_option("max_iterations",
                "GLPK supports the 'max_iterations' option only with Simplex.")
        elif self.ext.options.max_iterations is not None:
            options.it_lim = int(self.ext.options.max_iterations)

        # lp_root_method
        # Note that the PICOS option is explicitly also meant for the MIP
        # preprocessing step but GLPK does not support it in that scenario.
        if not continuous:
            self._handle_unsupported_option("lp_root_method",
                "GLPK supports the 'lp_root_method' option only for LPs.")
        elif self.ext.options.lp_root_method is not None:
            if self.ext.options.lp_root_method == "interior":
                # Handled above.
                pass
            elif self.ext.options.lp_root_method == "psimplex":
                assert simplex
                options.meth = glpk.GLP_PRIMAL
            elif self.ext.options.lp_root_method == "dsimplex":
                assert simplex
                options.meth = glpk.GLP_DUAL
            else:
                assert False, "Unexpected lp_root_method value."

        # timelimit
        if interior:
            self._handle_unsupported_option("timelimit",
                "GLPK does not support the 'timelimit' option with the "
                "Interior Point Method.")
        elif self.ext.options.timelimit is not None:
            options.tm_lim = int(1000 * self.ext.options.timelimit)

        # Handle unsupported options.
        self._handle_unsupported_options(
            "lp_node_method", "treememory", "max_fsb_nodes", "hotstart")

        # TODO: Add GLPK-sepcific options. Candidates are:
        #       For both Simplex and MIPs:
        #           tol_*, out_*
        #       For Simplex:
        #           pricing, r_test, obj_*
        #       For the Interior Point Method:
        #           ord_alg
        #       For MIPs:
        #           *_tech, *_heur, ps_tm_lim, *_cuts, cb_size, binarize

        # Attempt to solve the problem.
        with self._header():
            with self._stopwatch():
                if simplex:
                    # TODO: Support glp_exact.
                    error = glpk.glp_simplex(p, options)
                elif interior:
                    error = glpk.glp_interior(p, options)
                else:
                    options.presolve = glpk.GLP_ON
                    error = glpk.glp_intopt(p, options)

            # Conert error codes to text output.
            # Note that by printing it above the footer, this output is made to
            # look like it's coming from GLPK, which is technically wrong but
            # semantically correct.
            if error == glpk.GLP_EBADB:
                self._warn("Unable to start the search, because the initial "
                    "basis specified in the problem object is invalid.")
            elif error == glpk.GLP_ESING:
                self._warn("Unable to start the search, because the basis "
                    "matrix corresponding to the initial basis is singular "
                    "within the working precision.")
            elif error == glpk.GLP_ECOND:
                self._warn("Unable to start the search, because the basis "
                    "matrix corresponding to the initial basis is "
                    "ill-conditioned.")
            elif error == glpk.GLP_EBOUND:
                self._warn("Unable to start the search, because some double-"
                    "bounded variables have incorrect bounds.")
            elif error == glpk.GLP_EFAIL:
                self._warn("The search was prematurely terminated due to a "
                    "solver failure.")
            elif error == glpk.GLP_EOBJLL:
                self._warn("The search was prematurely terminated, because the "
                    "objective function being maximized has reached its lower "
                    "limit and continues decreasing.")
            elif error == glpk.GLP_EOBJUL:
                self._warn("The search was prematurely terminated, because the "
                    "objective function being minimized has reached its upper "
                    "limit and continues increasing.")
            elif error == glpk.GLP_EITLIM:
                self._warn("The search was prematurely terminated, because the "
                    "simplex iteration limit has been exceeded.")
            elif error == glpk.GLP_ETMLIM:
                self._warn("The search was prematurely terminated, because the "
                    "time limit has been exceeded.")
            elif error == glpk.GLP_ENOPFS:
                self._verbose("The LP has no primal feasible solution.")
            elif error == glpk.GLP_ENODFS:
                self._verbose("The LP has no dual feasible solution.")
            elif error != 0:
                self._warn("GLPK error {:d}.".format(error))

        # Retrieve primals.
        primals = {}
        if self.ext.options.primals is not False:
            for variable in self.ext.variables.values():
                value = []

                for localIndex in range(variable.dim):
                    glpkIndex = self._glpkVarOffset[variable] + localIndex

                    if simplex:
                        localValue = glpk.glp_get_col_prim(p, glpkIndex)
                    elif interior:
                        localValue = glpk.glp_ipt_col_prim(p, glpkIndex)
                    else:
                        localValue = glpk.glp_mip_col_val(p, glpkIndex)

                    value.append(localValue)

                primals[variable] = value

        # Retrieve duals.
        duals = {}
        if self.ext.options.duals is not False and continuous:
            for constraint in self.ext.constraints.values():
                if isinstance(constraint, DummyConstraint):
                    duals[constraint] = cvxopt.spmatrix(
                        [], [], [], constraint.size)
                    continue

                value = []

                for localIndex in range(len(constraint)):
                    glpkIndex = self._glpkConOffset[constraint] + localIndex

                    if simplex:
                        localValue = glpk.glp_get_row_dual(p, glpkIndex)
                    elif interior:
                        localValue = glpk.glp_ipt_row_dual(p, glpkIndex)
                    else:
                        assert False

                    value.append(localValue)

                dual = cvxopt.matrix(value, constraint.size)
                if (not constraint.is_increasing()) ^ minimizing:
                    dual = -dual

                duals[constraint] = dual

        # Retrieve objective value.
        if simplex:
            value = glpk.glp_get_obj_val(p)
        elif interior:
            value = glpk.glp_ipt_obj_val(p)
        else:
            value = glpk.glp_mip_obj_val(p)

        # Retrieve solution status.
        if simplex:
            probStatusCode = glpk.glp_get_status(p)
        elif interior:
            probStatusCode = glpk.glp_ipt_status(p)
        else:
            probStatusCode = glpk.glp_mip_status(p)

        if probStatusCode == glpk.GLP_OPT:
            # simplex, interior, mip
            problemStatus = PS_FEASIBLE
        elif probStatusCode == glpk.GLP_FEAS:
            # simplex, mip
            problemStatus = PS_FEASIBLE
        elif probStatusCode == glpk.GLP_INFEAS:
            # simplex, interior
            problemStatus = PS_UNKNOWN
        elif probStatusCode == glpk.GLP_NOFEAS:
            # simplex, interior, mip
            problemStatus = PS_INFEASIBLE
        elif probStatusCode == glpk.GLP_UNBND:
            # simplex
            problemStatus = PS_UNBOUNDED
        elif probStatusCode == glpk.GLP_UNDEF:
            # simplex, interior, mip
            problemStatus = PS_UNKNOWN
        else:
            problemStatus = PS_UNKNOWN

        if simplex:
            prmlStatusCode = glpk.glp_get_prim_stat(p)
            dualStatusCode = glpk.glp_get_dual_stat(p)

            if prmlStatusCode == glpk.GLP_FEAS:
                if probStatusCode == glpk.GLP_OPT:
                    prmlStatus = SS_OPTIMAL
                else:
                    prmlStatus = SS_FEASIBLE
            elif prmlStatusCode == glpk.GLP_INFEAS:
                prmlStatus = SS_INFEASIBLE
            elif prmlStatusCode == glpk.GLP_NOFEAS:
                prmlStatus = PS_INFEASIBLE
            elif prmlStatusCode == glpk.GLP_UNDEF:
                prmlStatus = SS_EMPTY
            else:
                prmlStatus = SS_UNKNOWN

            if dualStatusCode == glpk.GLP_FEAS:
                if probStatusCode == glpk.GLP_OPT:
                    dualStatus = SS_OPTIMAL
                else:
                    dualStatus = SS_FEASIBLE
            elif dualStatusCode == glpk.GLP_INFEAS:
                dualStatus = SS_INFEASIBLE
            elif dualStatusCode == glpk.GLP_NOFEAS:
                dualStatus = PS_INFEASIBLE
            elif dualStatusCode == glpk.GLP_UNDEF:
                dualStatus = SS_EMPTY
            else:
                dualStatus = SS_UNKNOWN
        elif interior:
            if probStatusCode == glpk.GLP_UNDEF:
                prmlStatus = SS_EMPTY
                dualStatus = SS_EMPTY
            elif probStatusCode == glpk.GLP_OPT:
                prmlStatus = SS_OPTIMAL
                dualStatus = SS_OPTIMAL
            elif probStatusCode == glpk.GLP_FEAS:
                prmlStatus = SS_FEASIBLE
                dualStatus = SS_FEASIBLE
            elif probStatusCode == glpk.GLP_NOFEAS:
                prmlStatus = SS_INFEASIBLE
                dualStatus = SS_INFEASIBLE
            else:
                prmlStatus = SS_UNKNOWN
                dualStatus = SS_UNKNOWN
        else:  # MIP
            if probStatusCode == glpk.GLP_UNDEF:
                prmlStatus = SS_EMPTY
            elif probStatusCode == glpk.GLP_OPT:
                prmlStatus = SS_OPTIMAL
            elif probStatusCode == glpk.GLP_FEAS:
                prmlStatus = SS_FEASIBLE
            elif probStatusCode == glpk.GLP_NOFEAS:
                prmlStatus = SS_INFEASIBLE
            else:
                prmlStatus = SS_UNKNOWN

            dualStatus = SS_EMPTY

        return self._make_solution(
            value, primals, duals, prmlStatus, dualStatus, problemStatus)


# --------------------------------------
__all__ = api_end(_API_START, globals())
