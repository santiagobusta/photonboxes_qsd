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

"""Implementation of :class:`CPLEXSolver`."""

import time
from collections import namedtuple

import cvxopt

from ..apidoc import api_end, api_start
from ..constraints import (AffineConstraint, ConvexQuadraticConstraint,
                           DummyConstraint, RSOCConstraint, SOCConstraint)
from ..expressions import (CONTINUOUS_VARTYPES, AffineExpression,
                           BinaryVariable, IntegerVariable,
                           QuadraticExpression)
from ..modeling.footprint import Specification
from ..modeling.solution import (PS_FEASIBLE, PS_ILLPOSED, PS_INF_OR_UNB,
                                 PS_INFEASIBLE, PS_UNBOUNDED, PS_UNKNOWN,
                                 PS_UNSTABLE, SS_EMPTY, SS_FAILURE,
                                 SS_FEASIBLE, SS_INFEASIBLE, SS_OPTIMAL,
                                 SS_PREMATURE, SS_UNKNOWN)
from .solver import (ConflictingOptionsError, DependentOptionError, Solver,
                     UnsupportedOptionError)

_API_START = api_start(globals())
# -------------------------------


#: Maps CPLEX status code to PICOS status triples.
CPLEX_STATUS_CODES = {
    # primal status, dual status,   problem status
1:   (SS_OPTIMAL,    SS_OPTIMAL,    PS_FEASIBLE),    # CPX_STAT_OPTIMAL
2:   (SS_UNKNOWN,    SS_INFEASIBLE, PS_UNBOUNDED),   # CPX_STAT_UNBOUNDED
3:   (SS_INFEASIBLE, SS_UNKNOWN,    PS_INFEASIBLE),  # CPX_STAT_INFEASIBLE
4:   (SS_UNKNOWN,    SS_UNKNOWN,    PS_INF_OR_UNB),  # CPX_STAT_INForUNBD
5:   (SS_INFEASIBLE, SS_UNKNOWN,    PS_UNSTABLE),    # CPX_STAT_OPTIMAL_INFEAS
6:   (SS_UNKNOWN,    SS_UNKNOWN,    PS_UNSTABLE),    # CPX_STAT_NUM_BEST
    # 7—9 are not defined.
10:  (SS_PREMATURE,  SS_PREMATURE,  PS_UNKNOWN),     # CPX_STAT_ABORT_IT_LIM
11:  (SS_PREMATURE,  SS_PREMATURE,  PS_UNKNOWN),     # CPX_STAT_ABORT_TIME_LIM
12:  (SS_PREMATURE,  SS_PREMATURE,  PS_UNKNOWN),     # CPX_STAT_ABORT_OBJ_LIM
13:  (SS_PREMATURE,  SS_PREMATURE,  PS_UNKNOWN),     # CPX_STAT_ABORT_USER
    # 14—19 seem irrelevant (CPX_STAT_*_RELAXED_*).
20:  (SS_UNKNOWN,    SS_UNKNOWN,    PS_ILLPOSED),    # …_OPTIMAL_FACE_UNBOUNDED
21:  (SS_PREMATURE,  SS_PREMATURE,  PS_UNKNOWN),     # …_ABORT_PRIM_OBJ_LIM
22:  (SS_PREMATURE,  SS_PREMATURE,  PS_UNKNOWN),     # …_ABORT_DUAL_OBJ_LIM
23:  (SS_FEASIBLE,   SS_FEASIBLE,   PS_FEASIBLE),    # CPX_STAT_FEASIBLE
    # 24 irrelevant (CPX_STAT_FIRSTORDER).
25:  (SS_PREMATURE,  SS_PREMATURE,  PS_UNKNOWN),     # …_ABORT_DETTIME_LIM
    # 26—29 are not defined.
    # 30—39 seem irrelevant (CPX_STAT_CONFLICT_*).
    # 40—100 are not defined.
101: (SS_OPTIMAL,    SS_EMPTY,      PS_FEASIBLE),    # CPXMIP_OPTIMAL
102: (SS_OPTIMAL,    SS_EMPTY,      PS_FEASIBLE),    # CPXMIP_OPTIMAL_TOL
103: (SS_INFEASIBLE, SS_EMPTY,      PS_INFEASIBLE),  # CPXMIP_INFEASIBLE
104: (SS_PREMATURE,  SS_EMPTY,      PS_UNKNOWN),     # CPXMIP_SOL_LIM          ?
105: (SS_FEASIBLE,   SS_EMPTY,      PS_FEASIBLE),    # CPXMIP_NODE_LIM_FEAS
106: (SS_PREMATURE,  SS_EMPTY,      PS_UNKNOWN),     # CPXMIP_NODE_LIM_INFEAS
107: (SS_FEASIBLE,   SS_EMPTY,      PS_FEASIBLE),    # CPXMIP_TIME_LIM_FEAS
108: (SS_PREMATURE,  SS_EMPTY,      PS_UNKNOWN),     # CPXMIP_TIME_LIM_INFEAS
109: (SS_FEASIBLE,   SS_EMPTY,      PS_FEASIBLE),    # CPXMIP_FAIL_FEAS
110: (SS_FAILURE,    SS_EMPTY,      PS_UNKNOWN),     # CPXMIP_FAIL_INFEAS
111: (SS_FEASIBLE,   SS_EMPTY,      PS_FEASIBLE),    # CPXMIP_MEM_LIM_FEAS
112: (SS_PREMATURE,  SS_EMPTY,      PS_UNKNOWN),     # CPXMIP_MEM_LIM_INFEAS
113: (SS_FEASIBLE,   SS_EMPTY,      PS_FEASIBLE),    # CPXMIP_ABORT_FEAS
114: (SS_PREMATURE,  SS_EMPTY,      PS_UNKNOWN),     # CPXMIP_ABORT_INFEAS
115: (SS_INFEASIBLE, SS_EMPTY,      PS_UNSTABLE),    # CPXMIP_OPTIMAL_INFEAS
116: (SS_FEASIBLE,   SS_EMPTY,      PS_FEASIBLE),    # CPXMIP_FAIL_FEAS_NO_TREE
117: (SS_FAILURE,    SS_EMPTY,      PS_UNKNOWN),     # …_FAIL_INFEAS_NO_TREE
118: (SS_UNKNOWN,    SS_EMPTY,      PS_UNBOUNDED),   # CPXMIP_UNBOUNDED
119: (SS_UNKNOWN,    SS_EMPTY,      PS_INF_OR_UNB),  # CPXMIP_INForUNBD
    # 120—126 seem irrelevant (CPXMIP_*_RELAXED_*).
127: (SS_FEASIBLE,   SS_EMPTY,      PS_FEASIBLE),    # CPXMIP_FEASIBLE
128: (SS_OPTIMAL,    SS_EMPTY,      PS_FEASIBLE),    # …_POPULATESOL_LIM       ?
129: (SS_OPTIMAL,    SS_EMPTY,      PS_FEASIBLE),    # …_OPTIMAL_POPULATED     ?
130: (SS_OPTIMAL,    SS_EMPTY,      PS_FEASIBLE),    # …_OPTIMAL_POPULATED_TOL ?
131: (SS_FEASIBLE,   SS_EMPTY,      PS_FEASIBLE),    # CPXMIP_DETTIME_LIM_FEAS
132: (SS_PREMATURE,  SS_EMPTY,      PS_UNKNOWN),     # CPXMIP_DETTIME_LIM_INFEAS
}


class CPLEXSolver(Solver):
    """Interface to the CPLEX solver via its official Python interface.

    .. note ::
        Names are used instead of indices for identifying both variables and
        constraints since indices can change if the CPLEX instance is modified.
    """

    # NOTE: When making changes, also see the section in _solve that tells CPLEX
    #       the problem type.
    SUPPORTED = Specification(
        objectives=[
            AffineExpression,
            QuadraticExpression],
        constraints=[
            DummyConstraint,
            AffineConstraint,
            SOCConstraint,
            RSOCConstraint,
            ConvexQuadraticConstraint])

    NONCONVEX_QP = Specification(
        objectives=[QuadraticExpression],
        constraints=[DummyConstraint, AffineConstraint])

    MetaConstraint = namedtuple("MetaConstraint", ("con", "dim"))

    @classmethod
    def supports(cls, footprint, explain=False):
        """Implement :meth:`~.solver.Solver.supports`."""
        result = Solver.supports(footprint, explain)
        if not result or (explain and not result[0]):
            return result

        # Support QPs and MIQPs with a nonconvex objective.
        # NOTE: SUPPORTED fully excludes nonconvex quadratic constraints. This
        #       further excludes QCQPs and MIQCQPs with a nonconvex objective.
        # TODO: See which of the excluded cases can be supported as well.
        if footprint.nonconvex_quadratic_objective \
        and footprint not in cls.NONCONVEX_QP:
            if explain:
                return (False, "(MI)QCQPs with nonconvex objective.")
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
        return 0.0  # Commercial solver.

    @classmethod
    def test_availability(cls):
        """Implement :meth:`~.solver.Solver.test_availability`."""
        cls.check_import("cplex")

    @classmethod
    def names(cls):
        """Implement :meth:`~.solver.Solver.names`."""
        return "cplex", "CPLEX", "IBM ILOG CPLEX Optimization Studio", None

    @classmethod
    def is_free(cls):
        """Implement :meth:`~.solver.Solver.is_free`."""
        return False

    def __init__(self, problem):
        """Initialize a CPLEX solver interface.

        :param ~picos.Problem problem: The problem to be solved.
        """
        super(CPLEXSolver, self).__init__(problem)

        self._cplexVar = dict(start=dict(), length=dict())
        """Maps a PICOS variable to a CPLEX start index and length."""

        self._cplexLinCon = dict(start=dict(), length=dict())
        """Maps a PICOS linear constraint to a CPLEX start index and length."""

        self._cplexQuadCon = dict(start=dict(), length=dict())
        """Maps a PICOS quadr. constraint to a CPLEX start index and length."""

        self._cplexMetaCon = dict()
        """Maps PICOS (rotated) second order conic constraints to a named tuple.

        The tuple has a ``dim`` property containing the linear auxiliary
        variable dimension and can thus be used as a key of ``self._cplexVar``.
        """

    def __del__(self):
        if self.int is not None:
            self.int.end()

    def reset_problem(self):
        """Implement :meth:`~.solver.Solver.reset_problem`."""
        if self.int is not None:
            self.int.end()
            self.int = None

        self._cplexVar["start"].clear()
        self._cplexVar["length"].clear()

        self._cplexLinCon["start"].clear()
        self._cplexLinCon["length"].clear()

        self._cplexQuadCon["start"].clear()
        self._cplexQuadCon["length"].clear()

        self._cplexMetaCon.clear()

    @classmethod
    def _register(cls, registry, key, indices):
        if isinstance(indices, int):
            start, length = indices, 1
        else:
            start, length = indices[0], len(indices)

            # Expect that indices are consecutive.
            assert isinstance(indices, range) \
                or tuple(indices) == tuple(range(start, start + length)), \
                "Not consecutive: {}".format(indices)

        registry["start"][key] = start
        registry["length"][key] = length

    @classmethod
    def _lookup(cls, registry, key):
        start = registry["start"][key]
        length = registry["length"][key]

        return list(range(start, start + length))

    @classmethod
    def _unregister(cls, registry, key):
        starts = registry["start"]

        start = starts.pop(key)
        length = registry["length"].pop(key)
        indices = list(range(start, start + length))

        for other in starts:
            if starts[other] > start:
                starts[other] -= length

        return indices

    def _import_variable(self, picosVar):
        import cplex

        dim = picosVar.dim

        # Retrieve types.
        if isinstance(picosVar, CONTINUOUS_VARTYPES):
            types = dim * self.int.variables.type.continuous
        elif isinstance(picosVar, IntegerVariable):
            types = dim * self.int.variables.type.integer
        elif isinstance(picosVar, BinaryVariable):
            types = dim * self.int.variables.type.binary
        else:
            assert False, "Unexpected variable type."

        # Retrieve bounds.
        lowerBounds = [-cplex.infinity]*dim
        upperBounds = [cplex.infinity]*dim
        lower, upper = picosVar.bound_dicts
        for i, b in lower.items():
            lowerBounds[i] = b
        for i, b in upper.items():
            upperBounds[i] = b

        # Import the variable.
        cplexIndices = self.int.variables.add(
            lb=lowerBounds, ub=upperBounds, types=types)

        # Register the variable.
        self._register(self._cplexVar, picosVar, cplexIndices)

    def _remove_variable(self, picosVar):
        # Unregister the variable.
        cplexIndices = self._unregister(self._cplexVar, picosVar)

        # Remove the variable.
        self.int.variables.delete(cplexIndices)

    def _affinexp_pic2cpl(self, picosExpression):
        import cplex

        for I, V, c in picosExpression.sparse_rows(self._cplexVar["start"]):
            yield cplex.SparsePair(ind=I, val=V), c

    def _scalar_affinexp_pic2cpl(self, picosExpression):
        assert len(picosExpression) == 1

        return next(self._affinexp_pic2cpl(picosExpression))

    def _quadexp_pic2cpl(self, picosExpression):
        import cplex

        assert isinstance(picosExpression, QuadraticExpression)

        start = self._cplexVar["start"]

        cplexI, cplexJ, cplexV = [], [], []
        for (x, y), Q in picosExpression._sparse_quads.items():
            cplexI.extend(Q.I + start[x])
            cplexJ.extend(Q.J + start[y])
            cplexV.extend(Q.V)

        return cplex.SparseTriple(ind1=cplexI, ind2=cplexJ, val=cplexV)

    def _import_linear_constraint(self, picosCon):
        assert isinstance(picosCon, AffineConstraint)

        length = len(picosCon)

        # Retrieve left hand side and right hand side expressions.
        cplexLHS, cplexRHS = [], []
        for linear, constant in self._affinexp_pic2cpl(picosCon.lmr):
            cplexLHS.append(linear)
            cplexRHS.append(-constant)

        # Retrieve senses.
        if picosCon.is_increasing():
            senses = length * "L"
        elif picosCon.is_decreasing():
            senses = length * "G"
        elif picosCon.is_equality():
            senses = length * "E"
        else:
            assert False, "Unexpected constraint relation."

        # Import the constraint.
        cplexIndices = self.int.linear_constraints.add(
            lin_expr=cplexLHS, senses=senses, rhs=cplexRHS)

        # Register the constraint.
        self._register(self._cplexLinCon, picosCon, cplexIndices)

    def _import_quad_constraint(self, picosCon):
        assert isinstance(picosCon, ConvexQuadraticConstraint)

        # Retrieve the affine term.
        cplexLinear, cplexRHS = self._scalar_affinexp_pic2cpl(picosCon.le0.aff)
        cplexRHS = -cplexRHS

        # Retrieve the quadratic term.
        cplexQuad = self._quadexp_pic2cpl(picosCon.le0)

        # Import the constraint.
        cplexIndices = self.int.quadratic_constraints.add(
            lin_expr=cplexLinear, quad_expr=cplexQuad, sense="L", rhs=cplexRHS)

        # Register the constraint.
        self._register(self._cplexQuadCon, picosCon, cplexIndices)

    # TODO: Handle SOC → Quadratic via a reformulation.
    def _import_socone_constraint(self, picosCon):
        import cplex

        assert isinstance(picosCon, SOCConstraint)

        picosLHS = picosCon.ne
        picosRHS = picosCon.ub
        picosLHSLen = len(picosLHS)

        # Add auxiliary variables: One for every dimension of the left hand side
        # of the PICOS constraint and one for its right hand side.
        cplexRHSVar = self.int.variables.add(
            lb=[0.0], ub=[+cplex.infinity],
            types=self.int.variables.type.continuous)[0]
        cplexLHSVars = self.int.variables.add(
            lb=[-cplex.infinity] * picosLHSLen,
            ub=[+cplex.infinity] * picosLHSLen,
            types=self.int.variables.type.continuous * picosLHSLen)

        # Add a constraint that identifies the right hand side CPLEX auxiliary
        # variable with the PICOS right hand side scalar expression.
        # NOTE: Order (RHS first) matters for dual retrieval.
        cplexRHSConLHS, cplexRHSConRHS = \
            self._scalar_affinexp_pic2cpl(-picosRHS)
        cplexRHSConRHS = -cplexRHSConRHS
        cplexRHSConLHS.ind.append(cplexRHSVar)
        cplexRHSConLHS.val.append(1.0)
        cplexRHSCon = self.int.linear_constraints.add(
            lin_expr=[cplexRHSConLHS], senses="E", rhs=[cplexRHSConRHS])[0]

        # Add constraints that identify the left hand side CPLEX auxiliary
        # variables with their slice of the PICOS left hand side expression.
        # TODO: Possible to get rid of the loop?
        cplexLHSConsLHSs, cplexLHSConsRHSs = [], []
        for localConIndex, (localLinExp, localConstant) in \
                enumerate(self._affinexp_pic2cpl(picosLHS)):
            localConstant = -localConstant
            localLinExp.ind.append(cplexLHSVars[localConIndex])
            localLinExp.val.append(-1.0)
            cplexLHSConsLHSs.append(localLinExp)
            cplexLHSConsRHSs.append(localConstant)
        cplexLHSCons = self.int.linear_constraints.add(
            lin_expr=cplexLHSConsLHSs, senses="E" * picosLHSLen,
            rhs=cplexLHSConsRHSs)

        # Add a quadratic constraint over the auxiliary variables that
        # represents the PICOS second order cone constraint itself.
        quadIndices = [cplexRHSVar] + list(cplexLHSVars)
        quadExpr = cplex.SparseTriple(
            ind1=quadIndices, ind2=quadIndices, val=[-1.0] + [1.0]*picosLHSLen)
        cplexQuadCon = self.int.quadratic_constraints.add(
            quad_expr=quadExpr, sense="L", rhs=0.0)

        # Register all auxiliary variables and constraints.
        cplexVars = [cplexRHSVar] + list(cplexLHSVars)
        cplexLinCons = [cplexRHSCon] + list(cplexLHSCons)
        metaCon = self.MetaConstraint(con=picosCon, dim=len(cplexVars))

        self._cplexMetaCon[picosCon] = metaCon
        self._register(self._cplexVar, metaCon, cplexVars)
        self._register(self._cplexLinCon, metaCon, cplexLinCons)
        self._register(self._cplexQuadCon, metaCon, cplexQuadCon)

    # TODO: Handle RSOC → Quadratic via a reformulation.
    def _import_rscone_constraint(self, picosCon):
        import cplex

        assert isinstance(picosCon, RSOCConstraint)

        picosLHS = picosCon.ne
        picosRHS1 = picosCon.ub1
        picosRHS2 = picosCon.ub2
        picosLHSLen = len(picosLHS)

        # Add auxiliary variables: One for every dimension of the left hand side
        # of the PICOS constraint and two for its right hand side.
        cplexRHSVars = self.int.variables.add(
            lb=[0.0, 0.0], ub=[+cplex.infinity] * 2,
            types=self.int.variables.type.continuous * 2)
        cplexLHSVars = self.int.variables.add(
            lb=[-cplex.infinity] * picosLHSLen,
            ub=[+cplex.infinity] * picosLHSLen,
            types=self.int.variables.type.continuous * picosLHSLen)

        # Add two constraints that identify the right hand side CPLEX auxiliary
        # variables with the PICOS right hand side scalar expressions.
        # NOTE: Order (RHS first) matters for dual retrieval.
        cplexRHSConsLHSs, cplexRHSConsRHSs = [], []
        for picosRHS, cplexRHSVar in zip((picosRHS1, picosRHS2), cplexRHSVars):
            linExp, constant = self._scalar_affinexp_pic2cpl(-picosRHS)
            linExp.ind.append(cplexRHSVar)
            linExp.val.append(1.0)
            constant = -constant
            cplexRHSConsLHSs.append(linExp)
            cplexRHSConsRHSs.append(constant)
        cplexRHSCons = self.int.linear_constraints.add(
            lin_expr=cplexRHSConsLHSs, senses="E" * 2, rhs=cplexRHSConsRHSs)

        # Add constraints that identify the left hand side CPLEX auxiliary
        # variables with their slice of the PICOS left hand side expression.
        # TODO: Possible to get rid of the loop?
        cplexLHSConsLHSs, cplexLHSConsRHSs = [], []
        for localConIndex, (localLinExp, localConstant) in \
                enumerate(self._affinexp_pic2cpl(picosLHS)):
            localLinExp.ind.append(cplexLHSVars[localConIndex])
            localLinExp.val.append(-1.0)
            localConstant = -localConstant
            cplexLHSConsLHSs.append(localLinExp)
            cplexLHSConsRHSs.append(localConstant)
        cplexLHSCons = self.int.linear_constraints.add(
            lin_expr=cplexLHSConsLHSs, senses="E" * picosLHSLen,
            rhs=cplexLHSConsRHSs)

        # Add a quadratic constraint over the auxiliary variables that
        # represents the PICOS rotated second order cone constraint itself.
        quadExpr = cplex.SparseTriple(
            ind1=[cplexRHSVars[0]] + list(cplexLHSVars),
            ind2=[cplexRHSVars[1]] + list(cplexLHSVars),
            val=[-1.0] + [1.0] * picosLHSLen)
        cplexQuadCon = self.int.quadratic_constraints.add(
            quad_expr=quadExpr, sense="L", rhs=0.0)

        # Register all auxiliary variables and constraints.
        cplexVars = list(cplexRHSVars) + list(cplexLHSVars)
        cplexLinCons = list(cplexRHSCons) + list(cplexLHSCons)

        metaCon = self.MetaConstraint(con=picosCon, dim=len(cplexVars))

        self._cplexMetaCon[picosCon] = metaCon
        self._register(self._cplexVar, metaCon, cplexVars)
        self._register(self._cplexLinCon, metaCon, cplexLinCons)
        self._register(self._cplexQuadCon, metaCon, cplexQuadCon)

    def _import_constraint(self, picosCon):
        if isinstance(picosCon, AffineConstraint):
            self._import_linear_constraint(picosCon)
        elif isinstance(picosCon, ConvexQuadraticConstraint):
            self._import_quad_constraint(picosCon)
        elif isinstance(picosCon, SOCConstraint):
            self._import_socone_constraint(picosCon)
        elif isinstance(picosCon, RSOCConstraint):
            self._import_rscone_constraint(picosCon)
        else:
            assert isinstance(picosCon, DummyConstraint), \
                "Unexpected constraint type: {}".format(
                picosCon.__class__.__name__)

    def _remove_constraint(self, picosCon):
        if isinstance(picosCon, AffineConstraint):
            cplexIndices = self._unregister(self._cplexLinCon, picosCon)

            self.int.linear_constraints.delete(cplexIndices)
        elif isinstance(picosCon, ConvexQuadraticConstraint):
            cplexIndices = self._unregister(self._cplexQuadCon, picosCon)

            self.int.quadratic_constraints.delete(cplexIndices)
        elif isinstance(picosCon, (SOCConstraint, RSOCConstraint)):
            metaCon = self._cplexMetaCon.pop(picosCon)

            cplexLinConIndices = self._unregister(self._cplexLinCon, metaCon)
            cplexQuadConIndices = self._unregister(self._cplexQuadCon, metaCon)
            cplexVarIndices = self._unregister(self._cplexVar, metaCon)

            self.int.linear_constraints.delete(cplexLinConIndices)
            self.int.quadratic_constraints.delete(cplexQuadConIndices)
            self.int.variables.delete(cplexVarIndices)
        else:
            assert isinstance(picosCon, DummyConstraint), \
                "Unexpected constraint type: {}".format(
                picosCon.__class__.__name__)

    def _import_affine_objective(self, picosExpression):
        assert isinstance(picosExpression, AffineExpression)
        assert picosExpression.scalar

        # Import constant part.
        self.int.objective.set_offset(picosExpression._constant_coef[0])

        # Import linear part.
        cplexLinear = []
        for picosVar, coefs in picosExpression._sparse_linear_coefs.items():
            cplexIndices = coefs.J + self._cplexVar["start"][picosVar]
            cplexCoefs = list(coefs)

            cplexLinear.extend(zip(cplexIndices, cplexCoefs))

        if cplexLinear:
            self.int.objective.set_linear(cplexLinear)

    def _reset_affine_objective(self):
        # Clear constant part.
        self.int.objective.set_offset(0.0)

        # Clear linear part.
        linear = self.int.objective.get_linear()
        if any(linear):
            self.int.objective.set_linear([(cplexVarIndex, 0.0)
                for cplexVarIndex, coef in enumerate(linear) if coef])

    def _import_quadratic_objective(self, picosExpression):
        assert isinstance(picosExpression, QuadraticExpression)

        # Import affine part of objective function.
        self._import_affine_objective(picosExpression.aff)

        # Import quadratic part of objective function.
        cplexQuadExpression = self._quadexp_pic2cpl(picosExpression)
        cplexQuadCoefs = zip(
            cplexQuadExpression.ind1, cplexQuadExpression.ind2,
            [2.0 * coef for coef in cplexQuadExpression.val])
        self.int.objective.set_quadratic_coefficients(cplexQuadCoefs)

    def _reset_quadratic_objective(self):
        quadratics = self.int.objective.get_quadratic()
        if quadratics:
            self.int.objective.set_quadratic(
                [(sparsePair.ind, [0]*len(sparsePair.ind))
                for sparsePair in quadratics])

    def _import_objective(self):
        picosSense, picosObjective = self.ext.no

        # Import objective sense.
        if picosSense == "min":
            cplexSense = self.int.objective.sense.minimize
        else:
            assert picosSense == "max"
            cplexSense = self.int.objective.sense.maximize
        self.int.objective.set_sense(cplexSense)

        # Import objective function.
        if isinstance(picosObjective, AffineExpression):
            self._import_affine_objective(picosObjective)
        else:
            assert isinstance(picosObjective, QuadraticExpression)
            self._import_quadratic_objective(picosObjective)

    def _reset_objective(self):
        self._reset_affine_objective()
        self._reset_quadratic_objective()

    def _import_problem(self):
        import cplex

        # Create a problem instance.
        self.int = cplex.Cplex()

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
            self._reset_objective()
            self._import_objective()

    def _solve(self):
        import cplex

        # Reset options.
        self.int.parameters.reset()

        o = self.ext.options
        p = self.int.parameters

        continuous = self.ext.is_continuous()

        # TODO: Allow querying self.ext.objective directly.
        nonconvex_quad_obj = self.ext.footprint.nonconvex_quadratic_objective

        # verbosity
        verbosity = self.verbosity()
        if verbosity <= 0:
            # Note that this behaviour disables warning even with a verbosity of
            # zero but this is still better than having verbose output for every
            # option that is set.
            self.int.set_results_stream(None)
        else:
            p.barrier.display.set(min(2, verbosity))
            p.conflict.display.set(min(2, verbosity))
            p.mip.display.set(min(5, verbosity))
            p.sifting.display.set(min(2, verbosity))
            p.simplex.display.set(min(2, verbosity))
            p.tune.display.set(min(3, verbosity))

        self.int.set_error_stream(None)  # Already handled as exceptions.

        # abs_prim_fsb_tol
        if o.abs_prim_fsb_tol is not None:
            p.simplex.tolerances.feasibility.set(o.abs_prim_fsb_tol)

        # abs_dual_fsb_tol
        if o.abs_dual_fsb_tol is not None:
            p.simplex.tolerances.optimality.set(o.abs_dual_fsb_tol)

        # rel_prim_fsb_tol, rel_dual_fsb_tol, rel_ipm_opt_tol
        convergenceTols = [tol for tol in (o.rel_prim_fsb_tol,
            o.rel_dual_fsb_tol, o.rel_ipm_opt_tol) if tol is not None]
        if convergenceTols:
            convergenceTol = min(convergenceTols)
            p.barrier.convergetol.set(convergenceTol)
            p.barrier.qcpconvergetol.set(convergenceTol)

        # abs_bnb_opt_tol
        if o.abs_bnb_opt_tol is not None:
            p.mip.tolerances.absmipgap.set(o.abs_bnb_opt_tol)

        # rel_bnb_opt_tol
        if o.rel_bnb_opt_tol is not None:
            p.mip.tolerances.mipgap.set(o.rel_bnb_opt_tol)

        # integrality_tol
        if o.integrality_tol is not None:
            p.mip.tolerances.integrality.set(o.integrality_tol)

        # markowitz_tol
        if o.markowitz_tol is not None:
            p.simplex.tolerances.markowitz.set(o.markowitz_tol)

        # max_iterations
        if o.max_iterations is not None:
            maxit = o.max_iterations
            p.barrier.limits.iteration.set(maxit)
            p.simplex.limits.iterations.set(maxit)

        _lpm = {"interior": 4, "psimplex": 1, "dsimplex": 2}

        # lp_node_method
        if o.lp_node_method is not None:
            assert o.lp_node_method in _lpm, "Unexpected lp_node_method value."
            p.mip.strategy.subalgorithm.set(_lpm[o.lp_node_method])

        # lp_root_method
        if o.lp_root_method is not None:
            assert o.lp_root_method in _lpm, "Unexpected lp_root_method value."
            p.lpmethod.set(_lpm[o.lp_root_method])

        # timelimit
        if o.timelimit is not None:
            p.timelimit.set(o.timelimit)

        # treememory
        if o.treememory is not None:
            p.mip.limits.treememory.set(o.treememory)

        # Handle option conflict between "max_fsb_nodes" and "pool_size".
        if o.max_fsb_nodes is not None \
        and o.pool_size is not None:
            raise ConflictingOptionsError("The options 'max_fsb_nodes' and "
                "'pool_size' cannot be used in conjunction.")

        # max_fsb_nodes
        if o.max_fsb_nodes is not None:
            p.mip.limits.solutions.set(o.max_fsb_nodes)

        # pool_size
        if o.pool_size is not None:
            if continuous:
                raise UnsupportedOptionError("The option 'pool_size' can only "
                    "be used with mixed integer problems.")
            maxNumSolutions = max(1, int(o.pool_size))
            p.mip.limits.populate.set(maxNumSolutions)
        else:
            maxNumSolutions = 1

        # pool_relgap
        if o.pool_rel_gap is not None:
            if o.pool_size is None:
                raise DependentOptionError("The option 'pool_rel_gap' requires "
                    "the option 'pool_size'.")
            p.mip.pool.relgap.set(o.pool_rel_gap)

        # pool_abs_gap
        if o.pool_abs_gap is not None:
            if o.pool_size is None:
                raise DependentOptionError("The option 'pool_abs_gap' requires "
                    "the option 'pool_size'.")
            p.mip.pool.absgap.set(o.pool_abs_gap)

        # hotstart
        if o.hotstart:
            indices, values = [], []
            for picosVar in self.ext.variables.values():
                if picosVar.valued:
                    indices.extend(self._lookup(self._cplexVar, picosVar))
                    values.extend(cvxopt.matrix(picosVar.internal_value))

            if indices:
                self.int.MIP_starts.add(
                    cplex.SparsePair(ind=indices, val=values),
                    self.int.MIP_starts.effort_level.repair)

        # Set the optimality target now so that cplex_params may overwrite it.
        # This allows solving QPs and MIQPs with a nonconvex objective.
        if nonconvex_quad_obj:
            p.optimalitytarget.set(3)

        # Load a virtual machine config.
        if self.ext.options.cplex_vmconfig:
            self.int.copy_vmconfig(self.ext.options.cplex_vmconfig)

        # Handle CPLEX-specific options.
        for key, value in o.cplex_params.items():
            try:
                parameter = getattr(self.int.parameters, key)
            except AttributeError as error:
                self._handle_bad_solver_specific_option_key(key, error)

            try:
                parameter.set(value)
            except cplex.exceptions.errors.CplexError as error:
                self._handle_bad_solver_specific_option_value(key, value, error)

        # Handle options "cplex_upr_bnd_limit", "cplex_lwr_bnd_limit" and
        # "cplex_bnd_monitor" via a CPLEX callback handler.
        callback = None
        if o.cplex_upr_bnd_limit or o.cplex_lwr_bnd_limit \
        or o.cplex_bnd_monitor:
            from cplex.callbacks import MIPInfoCallback

            class PicosInfoCallback(MIPInfoCallback):
                def __call__(self):
                    v1 = self.get_incumbent_objective_value()
                    v2 = self.get_best_objective_value()
                    ub = max(v1, v2)
                    lb = min(v1, v2)
                    if self.bounds is not None:
                        elapsedTime = time.time() - self.startTime
                        self.bounds.append((elapsedTime, lb, ub))
                    if self.lbound is not None and lb >= self.lbound:
                        self.printer("The specified lower bound was reached, "
                            "so PICOS will ask CPLEX to stop the search.")
                        self.abort()
                    if self.ubound is not None and ub <= self.ubound:
                        self.printer("The specified upper bound was reached, "
                            "so PICOS will ask CPLEX to stop the search.")
                        self.abort()

            # Register the callback handler with CPLEX.
            callback = self.int.register_callback(PicosInfoCallback)

            # Pass parameters to the callback handler. Note that
            # callback.startTime will be set just before optimization begins.
            callback.printer = self._verbose
            callback.ubound = o.cplex_upr_bnd_limit
            callback.lbound = o.cplex_lwr_bnd_limit
            callback.bounds = [] if o.cplex_bnd_monitor else None

        # Inform CPLEX about the problem type.
        # This seems necessary, as otherwise LP can get solved as MIP, producing
        # misleading status output (e.g. "not integer feasible").
        conTypes = set(c.__class__ for c in self.ext.constraints.values())
        quadObj = isinstance(self.ext.no.function, QuadraticExpression)
        cplexTypes = self.int.problem_type

        if quadObj:
            if conTypes.issubset(set([DummyConstraint, AffineConstraint])):
                cplexType = cplexTypes.QP if continuous else cplexTypes.MIQP
            else:
                # Assume quadratic constraint types.
                cplexType = cplexTypes.QCP if continuous else cplexTypes.MIQCP
        else:
            if conTypes.issubset(set([DummyConstraint, AffineConstraint])):
                cplexType = cplexTypes.LP if continuous else cplexTypes.MILP
            else:
                # Assume quadratic constraint types.
                cplexType = cplexTypes.QCP if continuous else cplexTypes.MIQCP

        # Silence a warning explaining that optimality target 3 changes the
        # problem type from QP to MIQP by doing so manually.
        if nonconvex_quad_obj:
            # Enforce consistency with CPLEXSolver.supports.
            assert cplexType in (cplexTypes.QP, cplexTypes.MIQP)

            if p.optimalitytarget.get() == 3:  # User might have changed it.
                cplexType = cplexTypes.MIQP

        if cplexType is not None:
            self.int.set_problem_type(cplexType)

        # Attempt to solve the problem.
        if callback:
            callback.startTime = time.time()
        with self._header(), self._stopwatch():
            try:
                if maxNumSolutions > 1:
                    self.int.populate_solution_pool()
                    numSolutions = self.int.solution.pool.get_num()
                else:
                    self.int.solve()
                    numSolutions = 1
            except cplex.exceptions.errors.CplexSolverError as error:
                if error.args[2] == 5002:
                    self._handle_continuous_nonconvex_error(error)
                else:
                    raise

        solutions = []
        for solutionNum in range(numSolutions):
            # Retrieve primals.
            primals = {}
            if o.primals is not False:
                for picosVar in self.ext.variables.values():
                    try:
                        indices = self._lookup(self._cplexVar, picosVar)

                        if maxNumSolutions > 1:
                            value = self.int.solution.pool.get_values(
                                solutionNum, indices)
                        else:
                            value = self.int.solution.get_values(indices)

                        primals[picosVar] = value
                    except cplex.exceptions.errors.CplexSolverError:
                        primals[picosVar] = None

            # Retrieve duals.
            duals = {}
            if o.duals is not False and continuous:
                assert maxNumSolutions == 1

                for picosCon in self.ext.constraints.values():
                    if isinstance(picosCon, DummyConstraint):
                        duals[picosCon] = cvxopt.spmatrix(
                            [], [], [], picosCon.size)
                        continue

                    try:
                        if isinstance(picosCon, AffineConstraint):
                            indices = self._lookup(self._cplexLinCon, picosCon)
                            values = self.int.solution.get_dual_values(indices)
                            picosDual = cvxopt.matrix(values, picosCon.size)

                            if not picosCon.is_increasing():
                                picosDual = -picosDual
                        elif isinstance(picosCon, SOCConstraint):
                            metaCon = self._cplexMetaCon[picosCon]
                            indices = self._lookup(self._cplexLinCon, metaCon)
                            values = self.int.solution.get_dual_values(indices)
                            picosDual = -cvxopt.matrix(values)
                            picosDual[0] = -picosDual[0]
                        elif isinstance(picosCon, RSOCConstraint):
                            metaCon = self._cplexMetaCon[picosCon]
                            indices = self._lookup(self._cplexLinCon, metaCon)
                            values = self.int.solution.get_dual_values(indices)
                            picosDual = -cvxopt.matrix(values)
                            picosDual[0] = -picosDual[0]
                            picosDual[1] = -picosDual[1]
                        elif isinstance(picosCon, ConvexQuadraticConstraint):
                            picosDual = None
                        else:
                            assert False, "Unexpected constraint type."

                        if picosDual and self.ext.no.direction == "min":
                            picosDual = -picosDual
                    except cplex.exceptions.errors.CplexSolverError:
                        duals[picosCon] = None
                    else:
                        duals[picosCon] = picosDual

            # Retrieve objective value.
            try:
                if quadObj:
                    # FIXME: Retrieval of QP and MIQP objective value appears to
                    #        miss the quadratic part.
                    value = None
                elif maxNumSolutions > 1:
                    value = self.int.solution.pool.get_objective_value(
                        solutionNum)
                else:
                    value = self.int.solution.get_objective_value()
            except cplex.exceptions.errors.CplexSolverError:
                value = None

            # Retrieve solution status.
            code = self.int.solution.get_status()
            if code in CPLEX_STATUS_CODES:
                prmlStatus, dualStatus, probStatus = CPLEX_STATUS_CODES[code]
            else:
                prmlStatus = SS_UNKNOWN
                dualStatus = SS_UNKNOWN
                probStatus = PS_UNKNOWN

            info = {}
            if o.cplex_bnd_monitor:
                info["bounds_monitor"] = callback.bounds

            solutions.append(self._make_solution(value, primals, duals,
                prmlStatus, dualStatus, probStatus, info))

        if maxNumSolutions > 1:
            return solutions
        else:
            assert len(solutions) == 1
            return solutions[0]


# --------------------------------------
__all__ = api_end(_API_START, globals())
