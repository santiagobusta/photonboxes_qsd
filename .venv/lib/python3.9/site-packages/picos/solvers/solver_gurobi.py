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

"""Implementation of :class:`GurobiSolver`."""

from collections import namedtuple

import cvxopt
import numpy

from .. import settings
from ..apidoc import api_end, api_start
from ..constraints import (AffineConstraint, ConvexQuadraticConstraint,
                           DummyConstraint, NonconvexQuadraticConstraint,
                           RSOCConstraint, SOCConstraint)
from ..expressions import (CONTINUOUS_VARTYPES, AffineExpression,
                           BinaryVariable, IntegerVariable,
                           QuadraticExpression)
from ..expressions.data import cvx2csr, cvx2np
from ..modeling.footprint import Specification
from ..modeling.solution import (PS_FEASIBLE, PS_INF_OR_UNB, PS_INFEASIBLE,
                                 PS_UNBOUNDED, PS_UNKNOWN, PS_UNSTABLE,
                                 SS_EMPTY, SS_FEASIBLE, SS_INFEASIBLE,
                                 SS_OPTIMAL, SS_PREMATURE, SS_UNKNOWN)
from .solver import Solver

_API_START = api_start(globals())
# -------------------------------


class GurobiSolver(Solver):
    """Interface to the Gurobi solver via its official Python interface."""

    # TODO: Don't support (conic) quadratic constraints when duals are
    #       requested because their precision is bad and can't be controlled?
    SUPPORTED_8 = Specification(
        objectives=[
            AffineExpression,
            QuadraticExpression],
        constraints=[
            DummyConstraint,
            AffineConstraint,
            SOCConstraint,
            RSOCConstraint,
            ConvexQuadraticConstraint])

    SUPPORTED_9 = Specification(
        objectives=[
            AffineExpression,
            QuadraticExpression],
        constraints=[
            DummyConstraint,
            AffineConstraint,
            SOCConstraint,
            RSOCConstraint,
            NonconvexQuadraticConstraint])

    @classmethod
    def _gurobi9(cls):
        try:
            import gurobipy as gurobi
        except ImportError:
            # This method should be used only after test_availability confirmed
            # that gurobipy is available, however that method does not actually
            # perform an import and so it could still fail here due to a bad
            # installation. In this case an exception will be raised when Gurobi
            # is actually selected and we just return False as a dummy here.
            return False
        else:
            return hasattr(gurobi, "MVar")

    @classmethod
    def supports(cls, footprint, explain=False):
        """Implement :meth:`~.solver.Solver.supports`."""
        result = Solver.supports(footprint, explain)
        if not result or (explain and not result[0]):
            return result

        supported = cls.SUPPORTED_9 if cls._gurobi9() else cls.SUPPORTED_8

        if footprint not in supported:
            if explain:
                return False, supported.mismatch_reason(footprint)
            else:
                return False

        return (True, None) if explain else True

    @classmethod
    def default_penalty(cls):
        """Implement :meth:`~.solver.Solver.default_penalty`."""
        return 0.0  # Commercial solver.

    @classmethod
    def test_availability(cls):
        """Implement :meth:`~.solver.Solver.test_availability`."""
        cls.check_import("gurobipy")

    @classmethod
    def names(cls):
        """Implement :meth:`~.solver.Solver.names`."""
        return "gurobi", "Gurobi", "Gurobi Optimizer", None

    @classmethod
    def is_free(cls):
        """Implement :meth:`~.solver.Solver.is_free`."""
        return False

    GurobiMetaConstraint = namedtuple(
        "GurobiMetaConstraint", ("auxCons", "auxVars"))

    def __init__(self, problem):
        """Initialize a Gurobi solver interface.

        :param ~picos.Problem problem: The problem to be solved.
        """
        super(GurobiSolver, self).__init__(problem)

        self._matint_decision = None

        self._gurobiVar = dict()
        """Maps PICOS variable indices to Gurobi variables. (Matrix interf.)"""

        self._gurobiVars = []
        """A list of all Gurobi variables added. (Legacy interface.)"""

        self._gurobiVarOffset = dict()
        """Maps PICOS variables to Gurobi variable offsets. (Legacy interf.)"""

        self._gurobiLinCon = dict()
        """Maps a PICOS linear constraint to (a) Gurobi linear constraint(s)."""

        self._gurobiQuadCon = dict()
        """Maps a PICOS quadr. constraint to (a) Gurobi quadr. constraint(s)."""

        self._gurobiConicCon = dict()
        """Maps a PICOS quadr. constraint to its Gurobi representation."""

    def _make_matint_decision(self):
        default = settings.PREFER_GUROBI_MATRIX_INTERFACE
        choice = self.ext.options.gurobi_matint

        if not choice and (not default or choice is not None):
            return False

        if not self._gurobi9():
            if choice:
                raise RuntimeError(
                    "Gurobi's matrix interface should be used by user's choice "
                    "but Gurobi < 9 appears to be installed. More precisely, "
                    "gurobipy.Mvar is not available.")
            else:
                return False

        try:
            self.check_import("scipy.sparse")
        except ModuleNotFoundError as error:
            if choice:
                raise RuntimeError(
                    "Gurobi's matrix interface should be used by user's choice "
                    "but this requires SciPy, which was not found.") from error
            else:
                return False

        assert choice or (choice is None and default)
        return True

    @property
    def matint(self):
        """Whether Gurobi's matrix interface is in use."""
        if self._matint_decision is None:
            self._matint_decision = self._make_matint_decision()

        decision = self._matint_decision
        choice = self.ext.options.gurobi_matint

        if (choice and not decision) \
        or (decision and not choice and choice is not None):
            raise NotImplementedError(
                "The user's choice with respect to using Gurobi's matrix "
                "interface has changed between solution attempts. This is not "
                "supported. To re-load the problem with the other interface, "
                "you must manually reset your problem's solution strategy.")

        return decision

    def reset_problem(self):
        """Implement :meth:`~.solver.Solver.reset_problem`."""
        self.int = None

        self._gurobiVar.clear()
        self._gurobiVars.clear()
        self._gurobiVarOffset.clear()

        self._gurobiLinCon.clear()
        self._gurobiQuadCon.clear()
        self._gurobiConicCon.clear()

    def _import_variable(self, picosVar):
        import gurobipy as gurobi

        dim = picosVar.dim

        # Retrieve types.
        if isinstance(picosVar, CONTINUOUS_VARTYPES):
            gurobiVarType = gurobi.GRB.CONTINUOUS
        elif isinstance(picosVar, IntegerVariable):
            gurobiVarType = gurobi.GRB.INTEGER
        elif isinstance(picosVar, BinaryVariable):
            gurobiVarType = gurobi.GRB.BINARY
        else:
            assert False, "Unexpected variable type."

        # Retrieve bounds.
        lowerBounds = [-gurobi.GRB.INFINITY]*dim
        upperBounds = [gurobi.GRB.INFINITY]*dim
        lower, upper = picosVar.bound_dicts
        for i, b in lower.items():
            lowerBounds[i] = b
        for i, b in upper.items():
            upperBounds[i] = b

        # Import the variable.
        if self.matint:
            gurobiVar = self.int.addMVar(dim, lb=lowerBounds, ub=upperBounds,
                vtype=gurobiVarType, name=picosVar.name)

            self._gurobiVar[picosVar] = gurobiVar
        else:
            gurobiVarsDict = self.int.addVars(
                dim, lb=lowerBounds, ub=upperBounds, vtype=gurobiVarType)
            gurobiVars = [gurobiVarsDict[i] for i in range(dim)]

            self._gurobiVarOffset[picosVar] = len(self._gurobiVars)
            self._gurobiVars.extend(gurobiVars)

    def _remove_variable(self, picosVar):
        if self.matint:
            gurobiVar = self._gurobiVar.pop(picosVar)

            self.int.remove(gurobiVar)
        else:
            offset = self._gurobiVarOffset[picosVar]
            dim = picosVar.dim

            gurobiVars = self._gurobiVars[offset:offset + dim]

            self._gurobiVars = (
                self._gurobiVars[:offset] + self._gurobiVars[offset + dim:])

            for other in self._gurobiVarOffset:
                if self._gurobiVarOffset[other] > offset:
                    self._gurobiVarOffset[other] -= dim

            self.int.remove(gurobiVars)

    def _import_variable_values(self):
        for picosVar in self.ext.variables.values():
            if picosVar.valued:
                value = picosVar.internal_value

                if self.matint:
                    gurobiVar = self._gurobiVar[picosVar]
                    gurobiVar.Start = value
                else:
                    offset = self._gurobiVarOffset[picosVar]
                    dim = picosVar.dim

                    gurobiVars = self._gurobiVars[offset, offset + picosVar.dim]

                    for localIndex in range(dim):
                        gurobiVars[localIndex].Start = value[localIndex]

    def _reset_variable_values(self):
        import gurobipy as gurobi

        if self.matint:
            gurobiVars = self._gurobiVar.values()
        else:
            gurobiVars = self._gurobiVars

        for gurobiVar in gurobiVars:
            gurobiVar.Start = gurobi.GRB.UNDEFINED

    def _affexp_pic2grb_matint(self, picosExpression):
        assert self.matint

        # NOTE: Constant Gurobi matrix expressions don't exist; return thus
        #       constant PICOS expressions as NumPy arrays.
        gurobiExpression = numpy.ravel(cvx2np(
            picosExpression._constant_coef))

        for picosVar, coef in picosExpression._linear_coefs.items():
            A = cvx2csr(coef)
            x = self._gurobiVar[picosVar]

            # NOTE: Using __(r)matmul__ as PICOS supports Python 3.4 and the
            #       @-operator was implemented in Python 3.5.
            gurobiExpression += x.__rmatmul__(A)

        return gurobiExpression

    def _affexp_pic2grb_legacy(self, picosExpression):
        import gurobipy as gurobi

        assert not self.matint

        for J, V, c in picosExpression.sparse_rows(self._gurobiVarOffset):
            gurobiVars = [self._gurobiVars[j] for j in J]
            gurobiExpression = gurobi.LinExpr(V, gurobiVars)
            gurobiExpression.addConstant(c)

            yield gurobiExpression

    def _scalar_affexp_pic2grb(self, picosExpression):
        assert len(picosExpression) == 1

        if self.matint:
            gurobiExpression = self._affexp_pic2grb_matint(picosExpression)

            if picosExpression.constant:
                assert isinstance(gurobiExpression, numpy.ndarray)
                assert gurobiExpression.shape == (1,)

                return gurobiExpression[0]
            else:
                return gurobiExpression
        else:
            return next(self._affexp_pic2grb_legacy(picosExpression))

    def _quadexp_pic2grb(self, picosExpression):
        import gurobipy as gurobi

        assert isinstance(picosExpression, QuadraticExpression)

        if self.matint:
            # Import affine part.
            gurobiExpression = self._affexp_pic2grb_matint(picosExpression.aff)

            # Import quadratic part.
            for picosVars, coef in picosExpression._sparse_quads.items():
                Q = cvx2csr(coef)
                x = self._gurobiVar[picosVars[0]]
                y = self._gurobiVar[picosVars[1]]

                gurobiExpression += x.__matmul__(Q).__matmul__(y)
        else:
            # Import affine part.
            gurobiExpression = gurobi.QuadExpr(
                self._scalar_affexp_pic2grb(picosExpression.aff))

            # Import quadratic part.
            V, I, J  = [], [], []
            for (x, y), Q in picosExpression._sparse_quads.items():
                xOffset = self._gurobiVarOffset[x]
                yOffset = self._gurobiVarOffset[y]

                V.extend(Q.V)
                I.extend(self._gurobiVars[i] for i in Q.I + xOffset)
                J.extend(self._gurobiVars[j] for j in Q.J + yOffset)

            gurobiExpression.addTerms(V, I, J)

        return gurobiExpression

    def _import_linear_constraint(self, picosCon):
        import gurobipy as gurobi

        assert isinstance(picosCon, AffineConstraint)

        if self.matint:
            gurobiLHS = self._affexp_pic2grb_matint(picosCon.lhs)
            gurobiRHS = self._affexp_pic2grb_matint(picosCon.rhs)

            # HACK: Fallback to the legacy interface for constant constraints.
            # NOTE: This happens to work with remove_constraint since
            #       gurobipy.Model.remove accepts both lists and constraints.
            if isinstance(gurobiLHS, numpy.ndarray) \
            and isinstance(gurobiRHS, numpy.ndarray):
                if picosCon.is_increasing():
                    gurobiSense = gurobi.GRB.LESS_EQUAL
                elif picosCon.is_decreasing():
                    gurobiSense = gurobi.GRB.GREATER_EQUAL
                elif picosCon.is_equality():
                    gurobiSense = gurobi.GRB.EQUAL
                else:
                    assert False, "Unexpected constraint relation."

                return [self.int.addLConstr(a, gurobiSense, b)
                    for a, b in zip(gurobiLHS, gurobiRHS)]

            # Construct the constraint.
            if picosCon.is_increasing():
                gurobiCon = gurobiLHS <= gurobiRHS
            elif picosCon.is_decreasing():
                gurobiCon = gurobiLHS >= gurobiRHS
            elif picosCon.is_equality():
                gurobiCon = gurobiLHS == gurobiRHS
            else:
                assert False, "Unexpected constraint relation."

            # Add the constraint.
            gurobiCon = self.int.addConstr(gurobiCon)

            return gurobiCon
        else:
            # Retrieve sense.
            if picosCon.is_increasing():
                gurobiSense = gurobi.GRB.LESS_EQUAL
            elif picosCon.is_decreasing():
                gurobiSense = gurobi.GRB.GREATER_EQUAL
            elif picosCon.is_equality():
                gurobiSense = gurobi.GRB.EQUAL
            else:
                assert False, "Unexpected constraint relation."

            # Append scalar constraints.
            gurobiCons = [self.int.addLConstr(gurobiLHS, gurobiSense, 0.0)
                for gurobiLHS in self._affexp_pic2grb_legacy(picosCon.lmr)]

            return gurobiCons

    def _import_quad_constraint(self, picosCon):
        import gurobipy as gurobi

        # NOTE: NonconvexQuadraticConstraint includes ConvexQuadraticConstraint.
        assert isinstance(picosCon, NonconvexQuadraticConstraint)

        if self.matint:
            gurobiLE0 = self._quadexp_pic2grb(picosCon.le0)
            gurobiCon = self.int.addConstr(gurobiLE0 <= 0)
        else:
            gurobiLHS = self._quadexp_pic2grb(picosCon.le0)
            gurobiRHS = -gurobiLHS.getLinExpr().getConstant()

            if gurobiRHS:
                gurobiLHS.getLinExpr().addConstant(gurobiRHS)

            gurobiCon = self.int.addQConstr(
                gurobiLHS, gurobi.GRB.LESS_EQUAL, gurobiRHS)

        return gurobiCon

    def _import_socone_constraint(self, picosCon):
        import gurobipy as gurobi

        assert isinstance(picosCon, SOCConstraint)

        n = len(picosCon.ne)

        # Load defining expressions.
        gurobiRHS = self._scalar_affexp_pic2grb(picosCon.ub)

        if self.matint:
            # Load defining expressions.
            gurobiLHS = self._affexp_pic2grb_matint(picosCon.ne)

            # Add auxiliary variables for both sides.
            gurobiLHSVar = self.int.addMVar(n, lb=-gurobi.GRB.INFINITY)
            gurobiRHSVar = self.int.addMVar(1)

            # Add constraints to identify auxiliary variables with expressions.
            gurobiLHSCon = self.int.addConstr(gurobiLHSVar == gurobiLHS)
            gurobiRHSCon = self.int.addConstr(gurobiRHSVar == gurobiRHS)

            # Add a quadratic constraint over the auxiliary variables that
            # represents the PICOS second order cone constraint itself.
            gurobiQuadLHS = gurobiLHSVar.__matmul__(gurobiLHSVar)
            gurobiQuadRHS = gurobiRHSVar.__matmul__(gurobiRHSVar)
            gurobiQuadCon = self.int.addConstr(gurobiQuadLHS <= gurobiQuadRHS)

            # Collect auxiliary objects.
            auxCons = [gurobiLHSCon, gurobiRHSCon, gurobiQuadCon]
            auxVars = [gurobiLHSVar, gurobiRHSVar]
        else:
            # Load defining expressions.
            gurobiLHS = self._affexp_pic2grb_legacy(picosCon.ne)

            # Add auxiliary variables: One for every dimension of the left hand
            # side of the PICOS constraint and one for its right hand side.
            gurobiLHSVarsDict = self.int.addVars(
                n, lb=-gurobi.GRB.INFINITY, ub=gurobi.GRB.INFINITY)
            gurobiLHSVars = gurobiLHSVarsDict.values()
            gurobiRHSVar = self.int.addVar(lb=0.0, ub=gurobi.GRB.INFINITY)

            # Add constraints that identify the left hand side Gurobi auxiliary
            # variables with entries of the PICOS left hand side expression.
            gurobiLHSDict = dict(enumerate(gurobiLHS))
            gurobiLHSConsDict = self.int.addConstrs(
                gurobiLHSVarsDict[d] == gurobiLHSDict[d] for d in range(n))
            gurobiLHSCons = gurobiLHSConsDict.values()

            # Add a constraint that identifies the right hand side Gurobi
            # auxiliary variable with the PICOS right hand side expression.
            gurobiRHSCon = self.int.addLConstr(
                gurobiRHSVar, gurobi.GRB.EQUAL, gurobiRHS)

            # Add a quadratic constraint over the auxiliary variables that
            # represents the PICOS second order cone constraint itself.
            quadExpr = gurobi.QuadExpr()
            quadExpr.addTerms([1.0] * n, gurobiLHSVars, gurobiLHSVars)
            gurobiQuadCon = self.int.addQConstr(
                quadExpr, gurobi.GRB.LESS_EQUAL, gurobiRHSVar * gurobiRHSVar)

            # Collect auxiliary objects.
            auxCons = list(gurobiLHSCons) + [gurobiRHSCon, gurobiQuadCon]
            auxVars = list(gurobiLHSVars) + [gurobiRHSVar]

        return self.GurobiMetaConstraint(auxCons=auxCons, auxVars=auxVars)

    def _import_rscone_constraint(self, picosCon):
        import gurobipy as gurobi

        assert isinstance(picosCon, RSOCConstraint)

        n = len(picosCon.ne)

        # Load defining expressions.
        gurobiRHS = (
            self._scalar_affexp_pic2grb(picosCon.ub1),
            self._scalar_affexp_pic2grb(picosCon.ub2))

        if self.matint:
            # Load defining expressions.
            gurobiLHS = self._affexp_pic2grb_matint(picosCon.ne)

            # Add auxiliary variables for both sides.
            gurobiLHSVar = self.int.addMVar(n, lb=-gurobi.GRB.INFINITY)
            gurobiRHSVars = (self.int.addMVar(1), self.int.addMVar(1))

            # Add constraints to identify auxiliary variables with expressions.
            gurobiLHSCon = self.int.addConstr(gurobiLHSVar == gurobiLHS)
            gurobiRHSConsDict = self.int.addConstrs(
                gurobiRHSVars[i] == gurobiRHS[i] for i in range(2))
            gurobiRHSCons = gurobiRHSConsDict.values()

            # Add a quadratic constraint over the auxiliary variables that
            # represents the PICOS rotated second order cone constraint itself.
            gurobiQuadLHS = gurobiLHSVar.__matmul__(gurobiLHSVar)
            gurobiQuadRHS = gurobiRHSVars[0].__matmul__(gurobiRHSVars[1])
            gurobiQuadCon = self.int.addConstr(gurobiQuadLHS <= gurobiQuadRHS)

            # Collect auxiliary objects.
            auxCons = [gurobiLHSCon] + list(gurobiRHSCons) + [gurobiQuadCon]
            auxVars = [gurobiLHSVar] + list(gurobiRHSVars)
        else:
            # Load defining expressions.
            gurobiLHS = self._affexp_pic2grb_legacy(picosCon.ne)

            # Add auxiliary variables: One for every dimension of the left hand
            # side of the PICOS constraint and one for its right hand side.
            gurobiLHSVarsDict = self.int.addVars(
                n, lb=-gurobi.GRB.INFINITY, ub=gurobi.GRB.INFINITY)
            gurobiLHSVars = gurobiLHSVarsDict.values()
            gurobiRHSVars = self.int.addVars(
                2, lb=0.0, ub=gurobi.GRB.INFINITY).values()

            # Add constraints that identify the left hand side Gurobi auxiliary
            # variables with entries of the PICOS left hand side expression.
            gurobiLHSDict = dict(enumerate(gurobiLHS))
            gurobiLHSConsDict = self.int.addConstrs(
                gurobiLHSVarsDict[d] == gurobiLHSDict[d] for d in range(n))
            gurobiLHSCons = gurobiLHSConsDict.values()

            # Add two constraints that identify the right hand side Gurobi
            # auxiliary variables with the PICOS right hand side expressions.
            gurobiRHSConsDict = self.int.addConstrs(
                gurobiRHSVars[i] == gurobiRHS[i] for i in (0, 1))
            gurobiRHSCons = gurobiRHSConsDict.values()

            # Add a quadratic constraint over the auxiliary variables that
            # represents the PICOS second order cone constraint itself.
            quadExpr = gurobi.QuadExpr()
            quadExpr.addTerms([1.0] * n, gurobiLHSVars, gurobiLHSVars)
            gurobiQuadCon = self.int.addQConstr(quadExpr, gurobi.GRB.LESS_EQUAL,
                gurobiRHSVars[0] * gurobiRHSVars[1])

            # Collect auxiliary objects.
            auxCons = (
                list(gurobiLHSCons) + list(gurobiRHSCons) + [gurobiQuadCon])
            auxVars = list(gurobiLHSVars) + list(gurobiRHSVars)

        return self.GurobiMetaConstraint(auxCons=auxCons, auxVars=auxVars)

    def _import_constraint(self, picosCon):
        if isinstance(picosCon, AffineConstraint):
            self._gurobiLinCon[picosCon] = \
                self._import_linear_constraint(picosCon)
        elif isinstance(picosCon, NonconvexQuadraticConstraint):
            self._gurobiQuadCon[picosCon] = \
                self._import_quad_constraint(picosCon)
        elif isinstance(picosCon, SOCConstraint):
            self._gurobiConicCon[picosCon] = \
                self._import_socone_constraint(picosCon)
        elif isinstance(picosCon, RSOCConstraint):
            self._gurobiConicCon[picosCon] = \
                self._import_rscone_constraint(picosCon)
        else:
            assert isinstance(picosCon, DummyConstraint), \
                "Unexpected constraint type: {}".format(
                picosCon.__class__.__name__)

    def _remove_constraint(self, picosCon):
        if isinstance(picosCon, AffineConstraint):
            self.int.remove(self._gurobiLinCon.pop(picosCon))
        elif isinstance(picosCon, NonconvexQuadraticConstraint):
            self.int.remove(self._gurobiQuadCon.pop(picosCon))
        elif isinstance(picosCon, (SOCConstraint, RSOCConstraint)):
            metaCon = self._gurobiConicCon.pop(picosCon)

            self.int.remove(metaCon.auxCons)
            self.int.remove(metaCon.auxVars)
        else:
            assert isinstance(picosCon, DummyConstraint), \
                "Unexpected constraint type: {}".format(
                picosCon.__class__.__name__)

    def _import_objective(self):
        import gurobipy as gurobi

        picosSense, picosObjective = self.ext.no

        # Retrieve objective sense.
        if picosSense == "min":
            gurobiSense = gurobi.GRB.MINIMIZE
        else:
            assert picosSense == "max"
            gurobiSense = gurobi.GRB.MAXIMIZE

        # Retrieve objective function.
        if isinstance(picosObjective, AffineExpression):
            gurobiObjective = self._scalar_affexp_pic2grb(picosObjective)
        else:
            assert isinstance(picosObjective, QuadraticExpression)
            gurobiObjective = self._quadexp_pic2grb(picosObjective)

        self.int.setObjective(gurobiObjective, gurobiSense)

    def _import_problem(self):
        import gurobipy as gurobi

        # Create a problem instance.
        if self._license_warnings:
            self.int = gurobi.Model()
        else:
            with self._enforced_verbosity():
                self.int = gurobi.Model()

        # Import variables.
        for variable in self.ext.variables.values():
            self._import_variable(variable)

        # Import constraints.
        for constraint in self.ext.constraints.values():
            self._import_constraint(constraint)

        # Set objective.
        self._import_objective()

    def _update_problem(self):
        for oldConstraint in self._removed_constraints():
            self._remove_constraint(oldConstraint)

        for oldVariable in self._removed_variables():
            self._remove_variable(oldVariable)

        for newVariable in self._new_variables():
            self._import_variable(newVariable)

        for newConstraint in self._new_constraints():
            self._import_constraint(newConstraint)

        if self._objective_has_changed():
            self._import_objective()

    def _solve(self):
        import gurobipy as gurobi

        # Reset options.
        # NOTE: OutputFlag = 0 prevents resetParams from printing to console.
        self.int.Params.OutputFlag = 0
        self.int.resetParams()

        # verbosity
        self.int.Params.OutputFlag = 1 if self.verbosity() > 0 else 0

        # abs_prim_fsb_tol
        if self.ext.options.abs_prim_fsb_tol is not None:
            self.int.Params.FeasibilityTol = self.ext.options.abs_prim_fsb_tol

        # abs_dual_fsb_tol
        if self.ext.options.abs_dual_fsb_tol is not None:
            self.int.Params.OptimalityTol = self.ext.options.abs_dual_fsb_tol

        # rel_ipm_opt_tol
        if self.ext.options.rel_ipm_opt_tol is not None:
            self.int.Params.BarConvTol = self.ext.options.rel_ipm_opt_tol

            # HACK: Work around low precision (conic) quadratic duals.
            self.int.Params.BarQCPConvTol = \
                0.01 * self.ext.options.rel_ipm_opt_tol

        # abs_bnb_opt_tol
        if self.ext.options.abs_bnb_opt_tol is not None:
            self.int.Params.MIPGapAbs = self.ext.options.abs_bnb_opt_tol

        # rel_bnb_opt_tol
        if self.ext.options.rel_bnb_opt_tol is not None:
            self.int.Params.MIPGap = self.ext.options.rel_bnb_opt_tol

        # integrality_tol
        if self.ext.options.integrality_tol is not None:
            self.int.Params.IntFeasTol = self.ext.options.integrality_tol

        # markowitz_tol
        if self.ext.options.markowitz_tol is not None:
            self.int.Params.MarkowitzTol = self.ext.options.markowitz_tol

        # max_iterations
        if self.ext.options.max_iterations is not None:
            self.int.Params.BarIterLimit = self.ext.options.max_iterations
            self.int.Params.IterationLimit = self.ext.options.max_iterations

        _lpm = {"interior": 2, "psimplex": 0, "dsimplex": 1}

        # lp_node_method
        if self.ext.options.lp_node_method is not None:
            value = self.ext.options.lp_node_method
            assert value in _lpm, "Unexpected lp_node_method value."
            self.int.Params.SiftMethod = _lpm[value]

        # lp_root_method
        if self.ext.options.lp_root_method is not None:
            value = self.ext.options.lp_root_method
            assert value in _lpm, "Unexpected lp_root_method value."
            self.int.Params.Method = _lpm[value]

        # timelimit
        if self.ext.options.timelimit is not None:
            self.int.Params.TimeLimit = self.ext.options.timelimit

        # max_fsb_nodes
        if self.ext.options.max_fsb_nodes is not None:
            self.int.Params.SolutionLimit = self.ext.options.max_fsb_nodes

        # hotstart
        if self.ext.options.hotstart:
            self._import_variable_values()
        else:
            self._reset_variable_values()

        # Handle Gurobi-specific options.
        for key, value in self.ext.options.gurobi_params.items():
            if not self.int.getParamInfo(key):
                self._handle_bad_solver_specific_option_key(key)

            try:
                self.int.setParam(key, value)
            except TypeError as error:
                self._handle_bad_solver_specific_option_value(key, value, error)

        # Handle unsupported options.
        self._handle_unsupported_option("treememory")

        # Extend functionality for continuous problems.
        if self.ext.is_continuous():
            # Compute duals also for QPs and QC(Q)Ps.
            if self.ext.options.duals is not False:
                self.int.setParam(gurobi.GRB.Param.QCPDual, 1)

            # Allow nonconvex quadratic objectives.
            # TODO: Allow querying self.ext.objective directly.
            # TODO: Check if this should/must be set also for Gurobi >= 9.
            if self.ext.footprint.nonconvex_quadratic_objective:
                self.int.setParam(gurobi.GRB.Param.NonConvex, 2)

        # Attempt to solve the problem.
        with self._header(), self._stopwatch():
            try:
                self.int.optimize()
            except gurobi.GurobiError as error:
                if error.errno == gurobi.GRB.Error.Q_NOT_PSD:
                    self._handle_continuous_nonconvex_error(error)
                else:
                    raise

        # Retrieve primals.
        primals = {}
        if self.ext.options.primals is not False:
            for picosVar in self.ext.variables.values():
                try:
                    if self.matint:
                        value = cvxopt.matrix(self._gurobiVar[picosVar].X)
                    else:
                        o = self._gurobiVarOffset[picosVar]
                        d = picosVar.dim

                        value = [v.X for v in self._gurobiVars[o:o + d]]
                except (AttributeError, gurobi.GurobiError):
                    # NOTE: AttributeError is raised for gurobipy.Var,
                    #       gurobi.GurobiError for gurobipy.MVar.
                    primals[picosVar] = None
                else:
                    primals[picosVar] = value

        # Retrieve duals.
        duals = {}
        if self.ext.options.duals is not False and self.ext.is_continuous():
            for picosCon in self.ext.constraints.values():
                if isinstance(picosCon, DummyConstraint):
                    duals[picosCon] = cvxopt.spmatrix([], [], [], picosCon.size)
                    continue

                # HACK: Work around gurobiCon.getAttr(gurobi.GRB.Attr.Pi)
                #       printing a newline to console when it raises an
                #       AttributeError and OutputFlag is enabled. This is a
                #       WONTFIX on Gurobi's end (PICOS #264, Gurobi #14248).
                # TODO: Check if this also happens for urobiCon.Pi, which is now
                #       used for both interfaces.
                oldOutput = self.int.Params.OutputFlag
                self.int.Params.OutputFlag = 0

                try:
                    if isinstance(picosCon, AffineConstraint):
                        gurobiCon = self._gurobiLinCon[picosCon]

                        # HACK: Seee _import_linear_constraint.
                        if not self.matint or isinstance(gurobiCon, list):
                            gurobiDual = [c.Pi for c in gurobiCon]
                        else:
                            gurobiDual = gurobiCon.Pi

                        picosDual = cvxopt.matrix(gurobiDual, picosCon.size)

                        if not picosCon.is_increasing():
                            picosDual = -picosDual
                    elif isinstance(picosCon, SOCConstraint):
                        gurobiMetaCon = self._gurobiConicCon[picosCon]

                        if self.matint:
                            ne, ub, _ = gurobiMetaCon.auxCons
                            dual = numpy.hstack([ub.Pi, ne.Pi])
                            picosDual = cvxopt.matrix(dual)
                        else:
                            n = len(picosCon.ne)
                            assert len(gurobiMetaCon.auxCons) == n + 2

                            ne = gurobiMetaCon.auxCons[:n]
                            ub = gurobiMetaCon.auxCons[n]

                            z, lbd = [c.Pi for c in ne], ub.Pi
                            picosDual = cvxopt.matrix([lbd] + z)
                    elif isinstance(picosCon, RSOCConstraint):
                        gurobiMetaCon = self._gurobiConicCon[picosCon]

                        if self.matint:
                            ne, ub1, ub2, _ = gurobiMetaCon.auxCons
                            dual = numpy.hstack([ub1.Pi, ub2.Pi, ne.Pi])
                            picosDual = cvxopt.matrix(dual)
                        else:
                            n = len(picosCon.ne)
                            assert len(gurobiMetaCon.auxCons) == n + 3

                            ne = gurobiMetaCon.auxCons[:n]
                            ub1 = gurobiMetaCon.auxCons[n]
                            ub2 = gurobiMetaCon.auxCons[n + 1]

                            z, a, b = [c.Pi for c in ne], ub1.Pi, ub2.Pi
                            picosDual = cvxopt.matrix([a] + [b] + z)
                    elif isinstance(picosCon, NonconvexQuadraticConstraint):
                        picosDual = None
                    else:
                        assert isinstance(picosCon, DummyConstraint), \
                            "Unexpected constraint type: {}".format(
                            picosCon.__class__.__name__)

                    # Flip sign based on objective sense.
                    if picosDual and self.ext.no.direction == "min":
                        picosDual = -picosDual
                except (AttributeError, gurobi.GurobiError):
                    # NOTE: AttributeError is raised for gurobipy.Constr,
                    #       gurobi.GurobiError for gurobipy.MConstr.
                    duals[picosCon] = None
                else:
                    duals[picosCon] = picosDual

                # HACK: See above. Also: Silence Gurobi while enabling output.
                if oldOutput != 0:
                    with self._enforced_verbosity(noStdOutAt=float("inf")):
                        self.int.Params.OutputFlag = oldOutput

        # Retrieve objective value.
        try:
            value = self.int.ObjVal
        except AttributeError:
            value = None

        # Retrieve solution status.
        statusCode = self.int.Status
        if statusCode   == gurobi.GRB.Status.LOADED:
            raise RuntimeError("Gurobi claims to have just loaded the problem "
                "while PICOS expects the solution search to have terminated.")
        elif statusCode == gurobi.GRB.Status.OPTIMAL:
            primalStatus   = SS_OPTIMAL
            dualStatus     = SS_OPTIMAL
            problemStatus  = PS_FEASIBLE
        elif statusCode == gurobi.GRB.Status.INFEASIBLE:
            primalStatus   = SS_INFEASIBLE
            dualStatus     = SS_UNKNOWN
            problemStatus  = PS_INFEASIBLE
        elif statusCode == gurobi.GRB.Status.INF_OR_UNBD:
            primalStatus   = SS_UNKNOWN
            dualStatus     = SS_UNKNOWN
            problemStatus  = PS_INF_OR_UNB
        elif statusCode == gurobi.GRB.Status.UNBOUNDED:
            primalStatus   = SS_UNKNOWN
            dualStatus     = SS_INFEASIBLE
            problemStatus  = PS_UNBOUNDED
        elif statusCode == gurobi.GRB.Status.CUTOFF:
            # "Optimal objective for model was proven to be worse than the value
            # specified in the Cutoff parameter. No solution information is
            # available."
            primalStatus   = SS_PREMATURE
            dualStatus     = SS_PREMATURE
            problemStatus  = PS_UNKNOWN
        elif statusCode == gurobi.GRB.Status.ITERATION_LIMIT:
            primalStatus   = SS_PREMATURE
            dualStatus     = SS_PREMATURE
            problemStatus  = PS_UNKNOWN
        elif statusCode == gurobi.GRB.Status.NODE_LIMIT:
            primalStatus   = SS_PREMATURE
            dualStatus     = SS_EMPTY  # Applies only to mixed integer problems.
            problemStatus  = PS_UNKNOWN
        elif statusCode == gurobi.GRB.Status.TIME_LIMIT:
            primalStatus   = SS_PREMATURE
            dualStatus     = SS_PREMATURE
            problemStatus  = PS_UNKNOWN
        elif statusCode == gurobi.GRB.Status.SOLUTION_LIMIT:
            primalStatus   = SS_PREMATURE
            dualStatus     = SS_PREMATURE
            problemStatus  = PS_UNKNOWN
        elif statusCode == gurobi.GRB.Status.INTERRUPTED:
            primalStatus   = SS_PREMATURE
            dualStatus     = SS_PREMATURE
            problemStatus  = PS_UNKNOWN
        elif statusCode == gurobi.GRB.Status.NUMERIC:
            primalStatus   = SS_UNKNOWN
            dualStatus     = SS_UNKNOWN
            problemStatus  = PS_UNSTABLE
        elif statusCode == gurobi.GRB.Status.SUBOPTIMAL:
            # "Unable to satisfy optimality tolerances; a sub-optimal solution
            # is available."
            primalStatus   = SS_FEASIBLE
            dualStatus     = SS_FEASIBLE
            problemStatus  = PS_FEASIBLE
        elif statusCode == gurobi.GRB.Status.INPROGRESS:
            raise RuntimeError("Gurobi claims solution search to be 'in "
                "progress' while PICOS expects it to have terminated.")
        elif statusCode == gurobi.GRB.Status.USER_OBJ_LIMIT:
            # "User specified an objective limit (a bound on either the best
            # objective or the best bound), and that limit has been reached."
            primalStatus   = SS_FEASIBLE
            dualStatus     = SS_EMPTY  # Applies only to mixed integer problems.
            problemStatus  = PS_FEASIBLE
        else:
            primalStatus   = SS_UNKNOWN
            dualStatus     = SS_UNKNOWN
            problemStatus  = PS_UNKNOWN

        return self._make_solution(
            value, primals, duals, primalStatus, dualStatus, problemStatus)


# --------------------------------------
__all__ = api_end(_API_START, globals())
