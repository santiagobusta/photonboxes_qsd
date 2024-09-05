# ------------------------------------------------------------------------------
# Copyright (C) 2012-2017 Guillaume Sagnol
# Copyright (C) 2017-2022 Maximilian Stahlberg
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

"""Implementation of :class:`CVXOPTSolver`."""

import cvxopt
import numpy

from ..apidoc import api_end, api_start
from ..constraints import (AffineConstraint, DummyConstraint, LMIConstraint,
                           LogSumExpConstraint, RSOCConstraint, SOCConstraint)
from ..expressions import CONTINUOUS_VARTYPES, AffineExpression, LogSumExp
from ..modeling.footprint import Specification
from ..modeling.solution import (PS_FEASIBLE, PS_INFEASIBLE, PS_UNBOUNDED,
                                 PS_UNKNOWN, SS_INFEASIBLE, SS_OPTIMAL,
                                 SS_UNKNOWN, Solution)
from .solver import Solver

_API_START = api_start(globals())
# -------------------------------


class CVXOPTSolver(Solver):
    """Interface to the CVXOPT solver.

    Also used as an interface to the
    :class:`SMCP solver <picos.solvers.solver_smcp.SMCPSolver>`.
    """

    SUPPORTED = Specification(
        objectives=[
            AffineExpression,
            LogSumExp],
        variables=CONTINUOUS_VARTYPES,
        constraints=[
            DummyConstraint,
            AffineConstraint,
            SOCConstraint,
            RSOCConstraint,
            LMIConstraint,
            LogSumExpConstraint])

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
        # CVXOPT is a dependency of PICOS, so it is always available.
        pass

    @classmethod
    def names(cls):
        """Implement :meth:`~.solver.Solver.names`."""
        return "cvxopt", "CVXOPT", "Python Convex Optimization Solver", None

    @classmethod
    def is_free(cls):
        """Implement :meth:`~.solver.Solver.is_free`."""
        return True

    @property
    def is_smcp(self):
        """Whether to implement SMCP instead of CVXOPT."""
        return False

    def __init__(self, problem):
        """Initialize a CVXOPT solver interface.

        :param ~picos.Problem problem: The problem to be solved.
        """
        super(CVXOPTSolver, self).__init__(problem)

        self._numVars = 0
        """Total number of scalar variables passed to CVXOPT."""

        self._cvxoptVarOffset = {}
        """Maps a PICOS variable to its offset in the constraint matrix."""

        # HACK: Setting this to false would result in variable bounds to be
        #       ignored, instead of added to the linear inequalities matrix.
        #       This is used by a Problem method that prints the problem to a
        #       file using a CVXOPTSolver instance.
        self.import_variable_bounds = True

        # SMCP currently offers no option reset, so keep a backup.
        if self.is_smcp:
            import smcp
            self._smcp_default_options = smcp.solvers.options.copy()

    def reset_problem(self):
        """Implement :meth:`~.solver.Solver.reset_problem`."""
        self.int = None

        self._numVars = 0
        self._cvxoptVarOffset.clear()

    def _is_geometric_program(self):
        return isinstance(self.ext.no.function, LogSumExp) or \
            any(isinstance(constraint, LogSumExpConstraint)
            for constraint in self.ext.constraints.values())

    def _affine_expression_to_G_and_h(self, expression):
        assert isinstance(expression, AffineExpression)

        return expression.sparse_matrix_form(
            varOffsetMap=self._cvxoptVarOffset, dense_b=True)

    _Gh = _affine_expression_to_G_and_h

    def _import_variables_without_bounds(self):
        offset = 0
        for variable in self.ext.variables.values():
            dim = variable.dim

            # Register the variable.
            self._cvxoptVarOffset[variable] = offset
            offset += dim

        assert offset == self._numVars

    def _import_variable_bounds(self):
        for variable in self.ext.variables.values():
            if self.import_variable_bounds:
                bounds = variable.bound_constraint
                if bounds:
                    self._import_affine_constraint(bounds)

    def _import_objective(self):
        direction, objective = self.ext.no

        if self._is_geometric_program():
            if isinstance(objective, LogSumExp):
                (F, g) = self._Gh(objective.x)
            else:
                assert isinstance(objective, AffineExpression)
                (F, g) = self._Gh(objective)

            # NOTE: Needs to happen before LogSumExpConstraint are added.
            self.int["K"] = [F.size[0]]

            if direction == "max":
                F, g = -F, -g

            self.int["F"] = F
            self.int["g"] = g
        else:
            (c, _) = self._Gh(objective)

            if direction == "max":
                c = -c

            # Must be a dense column-vector.
            c = cvxopt.matrix(c).T

            self.int["c"] = c

    def _import_affine_constraint(self, constraint):
        assert isinstance(constraint, AffineConstraint)

        (G_smaller, h_smaller) = self._Gh(constraint.smaller)
        (G_greater, h_greater) = self._Gh(constraint.greater)

        G = G_smaller - G_greater
        h = h_greater - h_smaller

        if constraint.is_equality():
            self.int["A"] = cvxopt.sparse([self.int["A"], G])
            self.int["b"] = cvxopt.matrix([self.int["b"], h])
        else:
            self.int["Gl"] = cvxopt.sparse([self.int["Gl"], G])
            self.int["hl"] = cvxopt.matrix([self.int["hl"], h])

    def _import_soc_constraint(self, constraint):
        assert isinstance(constraint, SOCConstraint)

        (A, b) = self._Gh(constraint.ne)
        (c, d) = self._Gh(constraint.ub)

        self.int["Gq"].append(cvxopt.sparse([-c, -A]))
        self.int["hq"].append(cvxopt.matrix([d, b]))

    def _import_rsoc_constraint(self, constraint):
        assert isinstance(constraint, RSOCConstraint)

        (A,  b)  = self._Gh(constraint.ne)
        (c1, d1) = self._Gh(constraint.ub1)
        (c2, d2) = self._Gh(constraint.ub2)

        self.int["Gq"].append(cvxopt.sparse([-c1 - c2, -2 * A, c2 - c1]))
        self.int["hq"].append(cvxopt.matrix([d1 + d2, 2 * b, d1 - d2]))

    def _import_lmi_constraint(self, constraint):
        assert isinstance(constraint, LMIConstraint)

        (G_smaller, h_smaller) = self._Gh(constraint.smaller)
        (G_greater, h_greater) = self._Gh(constraint.greater)

        self.int["Gs"].append(G_smaller - G_greater)
        self.int["hs"].append(h_greater - h_smaller)

    def _import_lse_constraint(self, constraint):
        assert isinstance(constraint, LogSumExpConstraint)

        (F, g) = self._Gh(constraint.le0.x)

        self.int["F"] = cvxopt.sparse([self.int["F"], F])
        self.int["g"] = cvxopt.matrix([self.int["g"], g])
        self.int["K"].append(F.size[0])

    def _import_constraint(self, constraint):
        if isinstance(constraint, AffineConstraint):
            self._import_affine_constraint(constraint)
        elif isinstance(constraint, SOCConstraint):
            self._import_soc_constraint(constraint)
        elif isinstance(constraint, RSOCConstraint):
            self._import_rsoc_constraint(constraint)
        elif isinstance(constraint, LMIConstraint):
            self._import_lmi_constraint(constraint)
        elif isinstance(constraint, LogSumExpConstraint):
            self._import_lse_constraint(constraint)
        else:
            assert isinstance(constraint, DummyConstraint), \
                "Unexpected constraint type: {}".format(
                constraint.__class__.__name__)

    def _import_problem(self):
        self._numVars = sum(var.dim for var in self.ext.variables.values())

        # CVXOPT's internal problem representation is stateless; a number of
        # matrices are supplied to the appropriate solver function each time a
        # search is started. These matrices are thus stored in self.int.
        self.int = {
            # Objective function coefficients.
            "c": None,

            # Linear equality left hand side.
            "A": cvxopt.spmatrix([], [], [], (0, self._numVars), tc="d"),

            # Linear equality right hand side.
            "b": cvxopt.matrix([], (0, 1), tc="d"),

            # Linear inequality left hand side.
            "Gl": cvxopt.spmatrix([], [], [], (0, self._numVars), tc="d"),

            # Linear inequality right hand side.
            "hl": cvxopt.matrix([], (0, 1), tc="d"),

            # Second order cone inequalities left hand sides.
            "Gq": [],

            # Second order cone inequalities right hand sides.
            "hq": [],

            # Semidefinite cone inequalities left hand sides.
            "Gs": [],

            # Semidefinite cone inequalities right hand sides.
            "hs": [],

            # Geometric program data.
            "F": None,
            "g": None,
            "K": None
        }

        # Import variables without their bounds.
        self._import_variables_without_bounds()

        # Set objective.
        # NOTE: This needs to happen before constraints are added as
        #       self.int["K"][0] refers to an LSE objective while
        #       self.int["K"][i] with i > 0 refers to LSE constraints.
        self._import_objective()

        # Import constraints.
        for constraint in self.ext.constraints.values():
            self._import_constraint(constraint)

        # Import variable bounds as additional affine constraints.
        # NOTE: This needs to happen after constraints are added due to how the
        #       dual values for constraints are extracted.
        self._import_variable_bounds()

    def _update_problem(self):
        raise NotImplementedError

    def _solve(self):
        if self.is_smcp:
            import smcp

        p = self.int
        isGP = self._is_geometric_program()

        # Clear all options set previously. This is necessary because CVXOPT
        # options are global, and might be changed even by another problem.
        cvxopt.solvers.options.clear()

        # verbosity
        cvxopt.solvers.options["show_progress"] = (self.verbosity() >= 1)

        # rel_prim_fsb_tol, rel_dual_fsb_tol
        feasibilityTols = [tol for tol in (self.ext.options.rel_prim_fsb_tol,
                self.ext.options.rel_dual_fsb_tol) if tol is not None]
        if feasibilityTols:
            cvxopt.solvers.options["feastol"] = min(feasibilityTols)

        # abs_ipm_opt_tol
        if self.ext.options.abs_ipm_opt_tol is not None:
            cvxopt.solvers.options["abstol"] = self.ext.options.abs_ipm_opt_tol

        # rel_ipm_opt_tol
        if self.ext.options.rel_ipm_opt_tol is not None:
            cvxopt.solvers.options["reltol"] = self.ext.options.rel_ipm_opt_tol

        # max_iterations
        if self.ext.options.max_iterations is not None:
            cvxopt.solvers.options["maxiters"] = self.ext.options.max_iterations
        else:
            cvxopt.solvers.options["maxiters"] = int(1e6)

        # cvxopt_kktsolver
        if self.ext.options.cvxopt_kktsolver is not None:
            userKKT = self.ext.options.cvxopt_kktsolver
        else:
            userKKT = None

        # cvxopt_kktreg
        if self.ext.options.cvxopt_kktreg is not None:
            cvxopt.solvers.options["kktreg"] = self.ext.options.cvxopt_kktreg

        # Handle unsupported options.
        self._handle_unsupported_options(
            "lp_root_method", "lp_node_method", "timelimit", "treememory",
            "max_fsb_nodes", "hotstart")

        # TODO: Add CVXOPT-sepcific options. Candidates are:
        #       - refinement

        # Set options for SMCP.
        if self.is_smcp:
            # Restore default options.
            smcp.solvers.options = self._smcp_default_options.copy()

            # Copy options also used by CVXOPT.
            smcp.solvers.options.update(cvxopt.solvers.options)

            # Further handle "verbose" option.
            smcp.solvers.options["debug"] = (self.verbosity() >= 2)

            # TODO: Add SMCP-sepcific options.

        if self._debug():
            from pprint import pformat
            self._debug("Setting options:\n{}\n".format(pformat(
                smcp.solvers.options if self.is_smcp
                else cvxopt.solvers.options)))

        # Print a header.
        if self.is_smcp:
            subsolverText = None
        else:
            if isGP:
                subsolverText = "internal GP solver"
            else:
                subsolverText = "internal CONELP solver"

        # Further prepare the problem for the CVXOPT/SMCP CONELP solvers.
        # TODO: This should be done during import.
        if not isGP:
            # Retrieve the structure of the cone, which is a cartesian product
            # of the non-negative orthant of dimension l, a number of second
            # order cones with dimensions in q and a number of positive
            # semidefinite cones with dimensions in s.
            dims = {
                "l": p["Gl"].size[0],
                "q": [Gqi.size[0] for Gqi in p["Gq"]],
                "s": [int(numpy.sqrt(Gsi.size[0])) for Gsi in p["Gs"]]
            }

            # Construct G and h to contain all conic inequalities, starting with
            # those with respect to the non-negative orthant.
            G = p["Gl"]
            h = p["hl"]

            # SMCP's ConeLP solver does not handle (linear) equalities, so cast
            # them as inequalities.
            if self.is_smcp:
                smcp_eps = min(feasibilityTols)
                if p["A"].size[0] > 0:
                    G = cvxopt.sparse([G, p["A"]])
                    G = cvxopt.sparse([G, -p["A"]])
                    h = cvxopt.matrix([h, p["b"]+smcp_eps])
                    h = cvxopt.matrix([h, smcp_eps-p["b"]])
                    dims["l"] += (2 * p["A"].size[0])

                # Remove the lines in G and h corresponding to 0==0 or 0<=0
                JP = list(set(G.I))
                IP = range(len(JP))
                VP = [1] * len(JP)

                if len(JP) != dims["l"]:
                    # is there a constraint of the form 0<=a, (a<0) ?
                    if any([b < -smcp_eps for (i, b) in enumerate(h)
                            if i not in JP]):
                        raise Exception(
                            'infeasible constraint of the form '
                            '0 <= a, with a<0')

                    # left-multiply with PPP-matrix to remove 0-constraints
                    PPP = cvxopt.spmatrix(VP, IP, JP, (len(IP), G.size[0]))
                    dims["l"] = len(JP)
                    G = PPP * G
                    h = PPP * h

            # Add second-order cone inequalities.
            for i in range(len(dims["q"])):
                G = cvxopt.sparse([G, p["Gq"][i]])
                h = cvxopt.matrix([h, p["hq"][i]])

            # Add semidefinite cone inequalities.
            for i in range(len(dims["s"])):
                G = cvxopt.sparse([G, p["Gs"][i]])
                h = cvxopt.matrix([h, p["hs"][i]])

            # Remove zero lines from linear equality constraint matrix, as
            # CVXOPT expects this matrix to have full row rank.
            JP = list(set(p["A"].I))
            IP = range(len(JP))
            VP = [1] * len(JP)

            # Skip solution on an infeasible constraint.
            if any([b for (i, b) in enumerate(p["b"]) if i not in JP]):
                return Solution(
                    primals=None, duals=None, problem=self.ext, solver="PICOS",
                    primalStatus=SS_INFEASIBLE, dualStatus=SS_UNKNOWN,
                    problemStatus=PS_INFEASIBLE, vectorizedPrimals=True)

            P = cvxopt.spmatrix(VP, IP, JP, (len(IP), p["A"].size[0]))
            A = P * p["A"]
            b = P * p["b"]

        # Attempt to solve the problem.
        with self._header(subsolverText), self._stopwatch():
            if self.is_smcp:
                if self._debug():
                    self._debug("Calling smcp.solvers.conelp(c, G, h, dims) "
                        "with\nc:\n{}\nG:\n{}\nh:\n{}\ndims:\n{}\n"
                        .format(p["c"], G, h, dims))
                try:
                    result = smcp.solvers.conelp(p["c"], G, h, dims)
                except TypeError:
                    # HACK: Work around "'NoneType' object is not subscriptable"
                    #       exception with infeasible/unbounded problems.
                    result = None
            else:
                kwargs = {}

                if userKKT:
                    kwargs["kktsolver"] = userKKT
                elif isGP:
                    # Use the more reliable LDL solver right away.
                    kwargs["kktsolver"] = "ldl"
                else:
                    # Try the fast but unreliable CHOL solver first.
                    kwargs["kktsolver"] = "chol"

                if isGP:
                    result = cvxopt.solvers.gp(p["K"], p["F"], p["g"], p["Gl"],
                        p["hl"], p["A"], p["b"], **kwargs)
                else:
                    try:
                        result = cvxopt.solvers.conelp(
                            p["c"], G, h, dims, A, b, **kwargs)

                        if not userKKT and result["status"] == "unknown":
                            raise ValueError("The first solution attempt with "
                                "CHOL as a KKT solver returnd a solution with "
                                "unknown status. This exception triggers "
                                "another solution attempt using LDL.")
                    except ValueError:
                        # NOTE: Apart from the one created by PICOS above, there
                        #       are at least two more ValueError produced by
                        #       CVXOPT when an unreliable KKT solver is used.
                        if userKKT:
                            raise  # Always respect the user's choice.
                        else:
                            # Re-solve using the LDL solver.
                            # TODO: Consider pre-solving on PICOS end to prevent
                            #       the "Rank(A) < p or Rank([G; A]) < n" error.
                            kwargs["kktsolver"] = "ldl"
                            result = cvxopt.solvers.conelp(
                                p["c"], G, h, dims, A, b, **kwargs)

        # Retrieve primals.
        primals = {}
        if self.ext.options.primals is not False and result is not None \
        and result["x"] is not None:
            for variable in self.ext.variables.values():
                offset = self._cvxoptVarOffset[variable]
                value = list(result["x"][offset:offset + variable.dim])
                primals[variable] = value

        # Retrieve duals.
        duals = {}
        if self.ext.options.duals is not False and result is not None:
            (indy, indzl, indzq, indznl, indzs) = (0, 0, 0, 0, 0)

            if isGP:
                zkey  = "zl"
                zqkey = "zq"
                zskey = "zs"
            else:
                zkey  = "z"
                zqkey = "z"
                zskey = "z"
                indzq = dims["l"]
                indzs = dims["l"] + sum(dims["q"])

            if self.is_smcp:
                # Equality constraints were cast as two inequalities.
                ieq = p["Gl"].size[0]
                neq = (dims["l"] - ieq) // 2
                soleq = result["z"][ieq:ieq + neq]
                soleq -= result["z"][ieq + neq:ieq + 2 * neq]
            else:
                soleq = result["y"]

            for constraint in self.ext.constraints.values():
                if isinstance(constraint, DummyConstraint):
                    duals[constraint] = cvxopt.spmatrix(
                        [], [], [], constraint.size)
                    continue

                dual   = None
                consSz = len(constraint)

                if isinstance(constraint, AffineConstraint):
                    if constraint.is_equality():
                        if soleq is not None:
                            dual = -(P.T * soleq)[indy:indy + consSz]
                            indy += consSz
                    else:
                        if result[zkey] is not None:
                            dual = result[zkey][indzl:indzl + consSz]
                            indzl += consSz
                elif isinstance(constraint, SOCConstraint) \
                or isinstance(constraint, RSOCConstraint):
                    if result[zqkey] is not None:
                        if isGP:
                            dual = result[zqkey][indzq]
                            dual[1:] = -dual[1:]
                            indzq += 1
                        else:
                            dual = result[zqkey][indzq:indzq + consSz]
                            if isinstance(constraint, RSOCConstraint):
                                # RScone were cast as a SOcone on import, so
                                # transform the dual to a proper RScone dual.
                                alpha = dual[0] + dual[-1]
                                beta  = dual[0] - dual[-1]
                                z     = 2.0 * dual[1:-1]
                                dual  = cvxopt.matrix([alpha, beta, z])
                            indzq += consSz
                elif isinstance(constraint, LMIConstraint):
                    if result[zskey] is not None:
                        matSz = constraint.size[0]
                        if isGP:
                            dual = cvxopt.matrix(
                                result[zskey][indzs], (matSz, matSz))
                            indzs += 1
                        else:
                            dual = cvxopt.matrix(
                                result[zskey][indzs:indzs + consSz],
                                (matSz, matSz))
                            indzs += consSz
                elif isinstance(constraint, LogSumExpConstraint):
                    # TODO: Retrieve LSE duals.
                    indznl += 1

                duals[constraint] = dual

        # Retrieve objective value.
        if result is None:
            value = None
        elif isGP:
            value = None
        else:
            p = result['primal objective']
            d = result['dual objective']

            if p is not None and d is not None:
                value = 0.5 * (p + d)
            elif p is not None:
                value = p
            elif d is not None:
                value = d
            else:
                value = None

        if value is not None and self.ext.no.direction == "max":
            value = -value

        if self.is_smcp:
            value = -value

        # Retrieve solution status.
        status = result["status"] if result else "unknown"
        if   status == "optimal":
            primalStatus   = SS_OPTIMAL
            dualStatus     = SS_OPTIMAL
            problemStatus  = PS_FEASIBLE
        elif status == "primal infeasible":
            primalStatus   = SS_INFEASIBLE
            dualStatus     = SS_UNKNOWN
            problemStatus  = PS_INFEASIBLE
        elif status == "dual infeasible":
            primalStatus   = SS_UNKNOWN
            dualStatus     = SS_INFEASIBLE
            problemStatus  = PS_UNBOUNDED
        else:
            primalStatus   = SS_UNKNOWN
            dualStatus     = SS_UNKNOWN
            problemStatus  = PS_UNKNOWN

        return self._make_solution(value, primals, duals, primalStatus,
            dualStatus, problemStatus, {"cvxopt_sol": result})


# --------------------------------------
__all__ = api_end(_API_START, globals())
