# ------------------------------------------------------------------------------
# Copyright (C) 2018-2022 Maximilian Stahlberg
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

"""Implementation of :class:`SCIPSolver`."""

import cvxopt

from .. import settings
from ..apidoc import api_end, api_start
from ..constraints import (AffineConstraint, ConvexQuadraticConstraint,
                           DummyConstraint, NonconvexQuadraticConstraint,
                           RSOCConstraint, SOCConstraint)
from ..expressions import AffineExpression, BinaryVariable, IntegerVariable
from ..modeling.footprint import Specification
from ..modeling.solution import (PS_FEASIBLE, PS_INFEASIBLE, PS_UNBOUNDED,
                                 PS_UNKNOWN, SS_INFEASIBLE, SS_OPTIMAL,
                                 SS_PREMATURE, SS_UNKNOWN)
from .solver import OptionValueError, Solver

_API_START = api_start(globals())
# -------------------------------


class SCIPSolver(Solver):
    """Interface to the SCIP solver via the PySCIPOpt Python interface."""

    SUPPORTED = Specification(
        objectives=[
            AffineExpression],
        constraints=[
            DummyConstraint,
            AffineConstraint,
            SOCConstraint,
            RSOCConstraint,
            ConvexQuadraticConstraint,
            NonconvexQuadraticConstraint])

    @classmethod
    def supports(cls, footprint, explain=False):
        """Implement :meth:`~.solver.Solver.supports`."""
        result = Solver.supports(footprint, explain)
        if not result or (explain and not result[0]):
            return result

        # HACK: SCIP has trouble with continuous problems, in particular it only
        #       produces dual values for LPs (which are sometimes incorrect).
        #       Nonconvex quadratics are supported as no other solver can deal
        #       with them at this point.
        if not settings.UNRELIABLE_STRATEGIES \
        and footprint.continuous \
        and not ("con", NonconvexQuadraticConstraint) in footprint:
            if explain:
                return (False,
                    "Continuous problems, except for nonconvex quadratic ones "
                    "(PICOS' choice).")
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
        return 1.0  # Stable free/open source solver.

    @classmethod
    def test_availability(cls):
        """Implement :meth:`~.solver.Solver.test_availability`."""
        cls.check_import("pyscipopt")

    @classmethod
    def names(cls):
        """Implement :meth:`~.solver.Solver.names`."""
        return "scip", "SCIP", "SCIP Optimization Suite", "PySCIPOpt"

    @classmethod
    def is_free(cls):
        """Implement :meth:`~.solver.Solver.is_free`."""
        return False

    def __init__(self, problem):
        """Initialize a SCIP solver interface.

        :param ~picos.Problem problem: The problem to be solved.
        """
        super(SCIPSolver, self).__init__(problem)

        self._scipVars = []
        """A list of all SCIP variables added."""

        self._scipVarOffset = dict()
        """Maps PICOS variables to SCIP variable offsets."""

        self._scipCons = dict()
        """
        Maps PICOS constraints to lists of SCIP constraints.

        For PICOS second order cone constraints, the first entry in the list is
        a SCIP quadratic constraint and the second entry is a SCIP linear
        auxiliary constraint.
        """

        self._scipQuadObjAuxVar = None
        """A SCIP auxiliary variable to support a PICOS quadratic objective."""

        self._scipQuadObjAuxCon = None
        """A SCIP auxiliary constraint to support a PICOS quadr. objective."""

    def reset_problem(self):
        """Implement :meth:`~.solver.Solver.reset_problem`."""
        self.int = None

        self._scipVars.clear()
        self._scipVarOffset.clear()
        self._scipCons.clear()

        self._scipQuadObjAuxVar = None
        self._scipQuadObjAuxCon = None

    def _import_variable(self, picosVar):
        from math import ceil, floor

        dim = picosVar.dim
        inf = self.int.infinity()

        # Retrieve type code.
        if isinstance(picosVar, (BinaryVariable, IntegerVariable)):
            scipVarType = "I"
        else:
            scipVarType = "C"

        # Retrieve bounds.
        lowerBounds = [-inf]*dim
        upperBounds = [inf]*dim
        lower, upper = picosVar.bound_dicts
        for i, b in lower.items():
            lowerBounds[i] = b
        for i, b in upper.items():
            upperBounds[i] = b

        # Refine bounds.
        if isinstance(picosVar, IntegerVariable):
            lowerBounds = [int(ceil(b)) for b in lowerBounds]
            upperBounds = [int(floor(b)) for b in upperBounds]

        # Import scalar variables.
        self._scipVarOffset[picosVar] = len(self._scipVars)
        for i in range(dim):
            self._scipVars.append(self.int.addVar(
                vtype=scipVarType, lb=lowerBounds[i], ub=upperBounds[i]))

    def _import_variable_values(self):
        # TODO: Import variable values with SCIP.
        raise NotImplementedError

    def _reset_variable_values(self):
        # TODO: Import variable values with SCIP.
        raise NotImplementedError

    def _affinexp_pic2scip(self, picosExpression):
        """Transform an affine expression from PICOS to SCIP.

        Multidimensional PICOS expressions are converted to a number of scalar
        SCIP expressions.

        :returns: A :class:`list` of :class:`SCIP expressions
            <pyscipopt.scip.Expr>`.
        """
        from pyscipopt.scip import Expr

        if picosExpression is None:
            yield Expr()
            return

        rows = picosExpression.sparse_rows(self._scipVarOffset)
        for scipVars, coefficients, constant in rows:
            scipExpression = Expr()
            for scipVarIndex, coefficient in zip(scipVars, coefficients):
                scipExpression += coefficient * self._scipVars[scipVarIndex]
            scipExpression += constant
            yield scipExpression

    def _scalar_affinexp_pic2scip(self, picosExpression):
        """Transform a PICOS scalar affine expression into a SCIP expression.

        :returns: A :class:`SCIP expression <pyscipopt.scip.Expr>`.
        """
        assert len(picosExpression) == 1
        return next(self._affinexp_pic2scip(picosExpression))

    def _quadexp_pic2scip(self, picosExpression):
        # Convert affine part.
        scipExpression = self._scalar_affinexp_pic2scip(picosExpression.aff)

        # Convert quadratic part.
        for (pVar1, pVar2), pCoefficients \
        in picosExpression._sparse_quads.items():
            for sparseIndex in range(len(pCoefficients)):
                localVar1Index   = pCoefficients.I[sparseIndex]
                localVar2Index   = pCoefficients.J[sparseIndex]
                localCoefficient = pCoefficients.V[sparseIndex]
                scipVar1 = self._scipVars[
                    self._scipVarOffset[pVar1] + localVar1Index]
                scipVar2 = self._scipVars[
                    self._scipVarOffset[pVar2] + localVar2Index]
                scipExpression += localCoefficient * scipVar1 * scipVar2

        return scipExpression

    def _import_linear_constraint(self, picosConstraint):
        assert isinstance(picosConstraint, AffineConstraint)

        scipCons = []
        picosLHS = picosConstraint.lmr
        for scipLHS in self._affinexp_pic2scip(picosLHS):
            if picosConstraint.is_increasing():
                scipCons.append(self.int.addCons(scipLHS <= 0.0))
            elif picosConstraint.is_decreasing():
                scipCons.append(self.int.addCons(scipLHS >= 0.0))
            elif picosConstraint.is_equality():
                scipCons.append(self.int.addCons(scipLHS == 0.0))
            else:
                assert False, "Unexpected constraint relation."

        return scipCons

    def _import_quad_constraint(self, picosConstraint):
        assert isinstance(picosConstraint,
            (ConvexQuadraticConstraint, NonconvexQuadraticConstraint))

        scipLHS  = self._quadexp_pic2scip(picosConstraint.le0)
        scipCons = [self.int.addCons(scipLHS <= 0.0)]

        return scipCons

    def _import_socone_constraint(self, picosConstraint):
        scipCons = []
        scipQuadLHS = self._quadexp_pic2scip(
            (picosConstraint.ne | picosConstraint.ne)
            - (picosConstraint.ub * picosConstraint.ub))
        scipCons.append(self.int.addCons(scipQuadLHS <= 0.0))

        scipAuxLHS = self._scalar_affinexp_pic2scip(picosConstraint.ub)
        if scipAuxLHS.degree() > 0:
            scipCons.append(self.int.addCons(scipAuxLHS >= 0.0))

        return scipCons

    def _import_rscone_constraint(self, picosConstraint):
        scipCons = []
        scipLHS = self._quadexp_pic2scip(
            (picosConstraint.ne | picosConstraint.ne)
            - (picosConstraint.ub1 * picosConstraint.ub2))
        scipCons.append(self.int.addCons(scipLHS <= 0.0))

        # Make sure that either the RHS is constant, or one of the two
        # expressions is non-negative.
        scipAuxLHS = self._scalar_affinexp_pic2scip(picosConstraint.ub1)
        if scipAuxLHS.degree() > 0:
            scipCons.append(self.int.addCons(scipAuxLHS >= 0.0))
        else:
            scipAuxLHS = self._scalar_affinexp_pic2scip(picosConstraint.ub2)
            if scipAuxLHS.degree() > 0:
                scipCons.append(self.int.addCons(scipAuxLHS >= 0.0))

        return scipCons

    def _import_constraint(self, picosConstraint):
        # Import constraint based on type.
        if isinstance(picosConstraint, AffineConstraint):
            scipCons = self._import_linear_constraint(picosConstraint)
        elif isinstance(picosConstraint,
                (ConvexQuadraticConstraint, NonconvexQuadraticConstraint)):
            scipCons = self._import_quad_constraint(picosConstraint)
        elif isinstance(picosConstraint, SOCConstraint):
            scipCons = self._import_socone_constraint(picosConstraint)
        elif isinstance(picosConstraint, RSOCConstraint):
            scipCons = self._import_rscone_constraint(picosConstraint)
        else:
            assert isinstance(picosConstraint, DummyConstraint), \
                "Unexpected constraint type: {}".format(
                picosConstraint.__class__.__name__)

        # Map PICOS constraints to lists of SCIP constraints.
        self._scipCons[picosConstraint] = scipCons

    def _import_objective(self):
        picosSense, picosObjective = self.ext.no

        # Retrieve objective sense.
        if picosSense == "min":
            scipSense = "minimize"
        else:
            assert picosSense == "max"
            scipSense = "maximize"

        # Import objective function.
        scipObjective = self._scalar_affinexp_pic2scip(picosObjective)

        self.int.setObjective(scipObjective, scipSense)

    def _import_problem(self):
        import pyscipopt as scip

        # Create a problem instance.
        self.int = scip.Model()

        # Import variables.
        for variable in self.ext.variables.values():
            self._import_variable(variable)

        # Import constraints.
        for constraint in self.ext.constraints.values():
            self._import_constraint(constraint)

        # Set objective.
        self._import_objective()

    def _update_problem(self):
        # TODO: Support all problem updates supported by SCIP.
        raise NotImplementedError

    def _solve(self):
        import pyscipopt as scip

        # TODO: Give Problem an interface for checks like this.
        isLP = isinstance(self.ext.no.function, AffineExpression) \
            and all(isinstance(constraint, AffineConstraint)
            for constraint in self.ext.constraints.values())

        # Reset options.
        self.int.resetParams()

        # verbosity
        picosVerbosity = self.verbosity()
        if picosVerbosity <= -1:
            scipVerbosity = 0
        elif picosVerbosity == 0:
            scipVerbosity = 2
        elif picosVerbosity == 1:
            scipVerbosity = 3
        elif picosVerbosity >= 2:
            scipVerbosity = 5
        self.int.setIntParam("display/verblevel", scipVerbosity)

        # TODO: "numerics/lpfeastol" has been replaced in SCIP 7.0.0 (PySCIPOpt
        #       3.0.0) by "numerics/lpfeastolfactor", which is multiplied with
        #       some unknown default feasibility (maybe "numerics/feastol"?).
        #       Looking at it now it is not clear to me which of SCIP's
        #       feasibility tolerances does what exactly and whether they are
        #       absolute or relative measures. The workaround is to maintain
        #       PICOS' old guess but ignore abs_prim_fsb_tol with the new SCIP.
        if int(scip.__version__.split(".")[0]) < 3:
            # abs_prim_fsb_tol
            if self.ext.options.abs_prim_fsb_tol is not None:
                self.int.setRealParam("numerics/lpfeastol",
                    self.ext.options.abs_prim_fsb_tol)

        # abs_dual_fsb_tol
        if self.ext.options.abs_dual_fsb_tol is not None:
            self.int.setRealParam("numerics/dualfeastol",
                self.ext.options.abs_dual_fsb_tol)

        # rel_prim_fsb_tol, rel_dual_fsb_tol
        relFsbTols = [tol for tol in (self.ext.options.rel_prim_fsb_tol,
            self.ext.options.rel_dual_fsb_tol) if tol is not None]
        if relFsbTols:
            self.int.setRealParam("numerics/feastol", min(relFsbTols))

        # abs_ipm_opt_tol, abs_bnb_opt_tol
        absOptTols = [tol for tol in (self.ext.options.abs_ipm_opt_tol,
            self.ext.options.abs_bnb_opt_tol) if tol is not None]
        if absOptTols:
            self.int.setRealParam("limits/absgap", min(absOptTols))

        # rel_ipm_opt_tol, rel_bnb_opt_tol
        relOptTols = [tol for tol in (self.ext.options.rel_ipm_opt_tol,
            self.ext.options.rel_bnb_opt_tol) if tol is not None]
        if relOptTols:
            self.int.setRealParam("limits/gap", min(relOptTols))

        # timelimit
        if self.ext.options.timelimit is not None:
            self.int.setRealParam("limits/time", self.ext.options.timelimit)

        # treememory
        if self.ext.options.treememory is not None:
            self.int.setRealParam("limits/memory", self.ext.options.treememory)

        # max_fsb_nodes
        if self.ext.options.max_fsb_nodes is not None:
            self.int.setRealParam(
                "limits/solutions", float(self.ext.options.max_fsb_nodes))

        # Handle SCIP-specific options.
        for key, value in self.ext.options.scip_params.items():
            try:
                if isinstance(value, bool):
                    self.int.setBoolParam(key, value)
                elif isinstance(value, str):
                    if len(value) == 1:
                        try:
                            self.int.setCharParam(key, ord(value))
                        except LookupError:
                            self.int.setStringParam(key, value)
                    else:
                        self.int.setStringParam(key, value)
                elif isinstance(value, float):
                    self.int.setRealParam(key, value)
                elif isinstance(value, int):
                    try:
                        self.int.setIntParam(key, value)
                    except LookupError:
                        try:
                            self.int.setLongintParam(key, value)
                        except LookupError:
                            self.int.setRealParam(key, float(value))
                else:
                    raise TypeError(
                        "The value '{}' has an uncommon type.".format(value))
            except KeyError as error:
                self._handle_bad_solver_specific_option_key(key, error)
            except ValueError as error:
                self._handle_bad_solver_specific_option_value(key, value, error)
            except (TypeError, LookupError) as error:
                picos_option = "{}_params".format(self.name)
                assert picos_option in self.ext.options, \
                    "The PICOS option '{}' does not exist.".format(picos_option)

                raise OptionValueError(
                    "Failed to guess type of {} option '{}' set via '{}'."
                    .format(self.short_name, key, picos_option)) from error

        # Handle unsupported options.
        self._handle_unsupported_options(
            "hotstart", "lp_node_method", "lp_root_method", "max_iterations")

        # In the case of a pure LP, disable presolve to get duals.
        if self.ext.options.duals is not False and isLP:
            self._debug("Disabling SCIP's presolve, heuristics, and propagation"
                " features to allow for LP duals.")

            self.int.setPresolve(scip.SCIP_PARAMSETTING.OFF)
            self.int.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

            # Note that this is a helper to set options, so they get reset at
            # the beginning of the function instead of in the else-scope below.
            self.int.disablePropagation()
        else:
            self.int.setPresolve(scip.SCIP_PARAMSETTING.DEFAULT)
            self.int.setHeuristics(scip.SCIP_PARAMSETTING.DEFAULT)

        # Attempt to solve the problem.
        with self._header(), self._stopwatch():
            self.int.optimize()

        # Retrieve primals.
        primals = {}
        if self.ext.options.primals is not False:
            for picosVar in self.ext.variables.values():
                scipVarOffset = self._scipVarOffset[picosVar]

                value = []
                for localIndex in range(picosVar.dim):
                    scipVar = self._scipVars[scipVarOffset + localIndex]
                    value.append(self.int.getVal(scipVar))

                primals[picosVar] = value

        # Retrieve duals for LPs.
        duals = {}
        if self.ext.options.duals is not False and isLP:
            for picosConstraint in self.ext.constraints.values():
                if isinstance(picosConstraint, DummyConstraint):
                    duals[picosConstraint] = cvxopt.spmatrix(
                        [], [], [], picosConstraint.size)
                    continue

                assert isinstance(picosConstraint, AffineConstraint)

                # Retrieve dual value for constraint.
                scipDuals = []
                for scipCon in self._scipCons[picosConstraint]:
                    scipDuals.append(self.int.getDualsolLinear(scipCon))
                picosDual = cvxopt.matrix(scipDuals, picosConstraint.size)

                # Flip sign based on constraint relation.
                if not picosConstraint.is_increasing():
                    picosDual = -picosDual

                # Flip sign based on objective sense.
                if picosDual and self.ext.no.direction == "min":
                    picosDual = -picosDual

                duals[picosConstraint] = picosDual

        # Retrieve objective value.
        value = self.int.getObjVal()

        # Retrieve solution status.
        status = self.int.getStatus()
        if   status == "optimal":
            primalStatus   = SS_OPTIMAL
            dualStatus     = SS_OPTIMAL
            problemStatus  = PS_FEASIBLE
        elif status == "timelimit":
            primalStatus   = SS_PREMATURE
            dualStatus     = SS_PREMATURE
            problemStatus  = PS_UNKNOWN
        elif status == "infeasible":
            primalStatus   = SS_INFEASIBLE
            dualStatus     = SS_UNKNOWN
            problemStatus  = PS_INFEASIBLE
        elif status == "unbounded":
            primalStatus   = SS_UNKNOWN
            dualStatus     = SS_INFEASIBLE
            problemStatus  = PS_UNBOUNDED
        else:
            primalStatus   = SS_UNKNOWN
            dualStatus     = SS_UNKNOWN
            problemStatus  = PS_UNKNOWN

        return self._make_solution(
            value, primals, duals, primalStatus, dualStatus, problemStatus)


# --------------------------------------
__all__ = api_end(_API_START, globals())
