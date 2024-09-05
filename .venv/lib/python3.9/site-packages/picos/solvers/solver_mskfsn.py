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

"""Implementation of :class:`MOSEKFusionSolver`."""

import sys

import cvxopt

from ..apidoc import api_end, api_start
from ..constraints import (AffineConstraint, DummyConstraint, LMIConstraint,
                           RSOCConstraint, SOCConstraint)
from ..expressions import AffineExpression, BinaryVariable, IntegerVariable
from ..modeling.footprint import Specification
from ..modeling.solution import (PS_FEASIBLE, PS_ILLPOSED, PS_INF_OR_UNB,
                                 PS_INFEASIBLE, PS_UNBOUNDED, PS_UNKNOWN,
                                 SS_EMPTY, SS_FEASIBLE, SS_INFEASIBLE,
                                 SS_OPTIMAL, SS_UNKNOWN)
from .solver import ProblemUpdateError, Solver

_API_START = api_start(globals())
# -------------------------------


class MOSEKFusionSolver(Solver):
    """Interface to the MOSEK solver via its high level Fusion API.

    Supports both MOSEK 8 and 9.

    The Fusion API is currently much slower than MOSEK's low level Python API.
    If this changes in the future, the Fusion API would be the prefered
    interface.
    """

    SUPPORTED = Specification(
        objectives=[
            AffineExpression],
        constraints=[
            DummyConstraint,
            AffineConstraint,
            SOCConstraint,
            RSOCConstraint,
            LMIConstraint])

    @classmethod
    def supports(cls, footprint, explain=False):
        """Implement :meth:`~.solver.Solver.supports`."""
        result = Solver.supports(footprint, explain)
        if not result or (explain and not result[0]):
            return result

        # No integer SDPs.
        if footprint.integer and ("con", LMIConstraint) in footprint:
            if explain:
                return False, "Integer Semidefinite Programs."
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
        return 0.5  # Commercial solver with slower interface.

    @classmethod
    def test_availability(cls):
        """Implement :meth:`~.solver.Solver.test_availability`."""
        cls.check_import("mosek.fusion")

    @classmethod
    def names(cls):
        """Implement :meth:`~.solver.Solver.names`."""
        return "mskfsn", "MOSEK", "MOSEK", "Fusion API"

    @classmethod
    def is_free(cls):
        """Implement :meth:`~.solver.Solver.is_free`."""
        return False

    def __init__(self, problem):
        """Initialize a MOSEK (Fusion) solver interface.

        :param ~picos.Problem problem: The problem to be solved.
        """
        super(MOSEKFusionSolver, self).__init__(problem)

        # Maps PICOS variables to MOSEK variables and vice versa.
        self.knownVariables = {}

        # Maps PICOS constraints to MOSEK constraints and vice versa.
        self.knownConstraints = {}

    def __del__(self):
        if self.int is not None:
            self.int.dispose()

    def reset_problem(self):
        """Implement :meth:`~.solver.Solver.reset_problem`."""
        if self.int is not None:
            self.int.dispose()
        self.int = None
        self.knownVariables.clear()
        self.knownConstraints.clear()

    @classmethod
    def _get_major_version(cls):
        if not hasattr(cls, "mosekVersion"):
            import mosek
            cls.mosekVersion = mosek.Env.getversion()

        return cls.mosekVersion[0]

    ver = property(lambda self: self.__class__._get_major_version())
    """The major version of the available MOSEK library."""

    @classmethod
    def _mosek_sparse_triple(cls, I, J, V):
        """Transform a sparse triple (e.g. from CVXOPT) for use with MOSEK."""
        if cls._get_major_version() >= 9:
            IJV = list(IJV for IJV in zip(I, J, V) if IJV[2] != 0)
            I, J, V = (list(X) for X in zip(*IJV)) if IJV else ([], [], [])
        else:
            I = list(I) if not isinstance(I, list) else I
            J = list(J) if not isinstance(J, list) else J
            V = list(V) if not isinstance(V, list) else V

        return I, J, V

    @classmethod
    def _matrix_cvx2msk(cls, cvxoptMatrix):
        """Transform a CVXOPT (sparse) matrix into a MOSEK (sparse) matrix."""
        import mosek.fusion as msk

        M = cvxoptMatrix
        n, m = M.size

        if type(M) is cvxopt.spmatrix:
            return msk.Matrix.sparse(
                n, m, *cls._mosek_sparse_triple(M.I, M.J, M.V))
        elif type(M) is cvxopt.matrix:
            return msk.Matrix.dense(n, m, list(M.T))
        else:
            raise ValueError("Argument must be a CVXOPT matrix.")

    @classmethod
    def _mosek_vstack(cls, *expressions):
        """Vertically stack MOSEK expressions.

        This is a wrapper around MOSEK's :func:`vstack
        <mosek.fusion.Expr.vstack>` function that silences a FutureWarning.
        """
        import mosek.fusion as msk

        if cls._get_major_version() >= 9:
            return msk.Expr.vstack(*expressions)
        else:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                return msk.Expr.vstack(*expressions)

    def _affinexp_pic2msk(self, picosExpression):
        """Transform an affine expression from PICOS to MOSEK.

        Requries all contained variables to be known to MOSEK.
        """
        import mosek.fusion as msk

        assert isinstance(picosExpression, AffineExpression)

        vectorShape = [len(picosExpression), 1]
        targetShape = list(picosExpression.size)

        if self.ver < 9:
            vectorShape = msk.Set.make(vectorShape)
            targetShape = msk.Set.make(targetShape)

        # Convert linear part of expression.
        firstSummand = True
        for picosVar, factor in picosExpression._linear_coefs.items():
            mosekVar = self.knownVariables[picosVar]

            summand = msk.Expr.mul(self._matrix_cvx2msk(factor), mosekVar)
            if firstSummand:
                mosekExpression = summand
                firstSummand = False
            else:
                mosekExpression = msk.Expr.add(mosekExpression, summand)

        # Convert constant term of expression.
        if picosExpression.constant is not None:
            mosekConstant = msk.Expr.constTerm(
                self._matrix_cvx2msk(picosExpression._constant_coef))

            if firstSummand:
                mosekExpression = mosekConstant
            else:
                mosekExpression = msk.Expr.add(mosekExpression, mosekConstant)
        elif firstSummand:
            mosekExpression = msk.Expr.zeros(vectorShape)

        # Restore the expression's original shape.
        # NOTE: Transposition due to differing major orders.
        mosekExpression = msk.Expr.reshape(
            msk.Expr.transpose(mosekExpression), targetShape)

        if self._debug():
            self._debug(
                "Affine expression converted: {} → {}".format(
                repr(picosExpression), mosekExpression.toString()))

        return mosekExpression

    @classmethod
    def _bounds_pic2msk(cls, picosVar, fixMOSEK9=False):
        """Transform PICOS variable bounds to MOSEK matrices or scalars.

        Scalars are returned in the case of homogenous bounds.
        """
        import mosek.fusion as msk

        dim = picosVar.dim
        lower, upper = picosVar.bound_dicts

        if fixMOSEK9:
            LV, LI, LJ = [-1e20]*dim, list(range(dim)), [0]*dim
            UV, UI, UJ = [+1e20]*dim, list(range(dim)), [0]*dim

            for i, b in lower.items():
                LV[i] = b

            for i, b in upper.items():
                UV[i] = b
        else:
            LV, LI = [], []
            UV, UI = [], []

            for i, b in lower.items():
                LI.append(i)
                LV.append(b)

            for i, b in upper.items():
                UI.append(i)
                UV.append(b)

            LJ = [0]*len(LV)
            UJ = [0]*len(UV)

        mosekBounds = [None, None]
        for side, I, J, V in ((0, LI, LJ, LV), (1, UI, UJ, UV)):
            if len(V) == dim and len(set(V)) == 1:
                mosekBounds[side] = V[0]
            elif V:
                mosekBounds[side] = msk.Matrix.sparse(dim, 1, I, J, V)

        return mosekBounds

    def _import_variable(self, picosVar):
        import mosek.fusion as msk

        shape = [picosVar.dim, 1]

        # Import variable bounds.
        if not isinstance(picosVar, BinaryVariable):
            # Retrieve lower and upper bounds.
            lower, upper = self._bounds_pic2msk(picosVar)

            # Convert bounds to a domain.
            if lower is None and upper is None:
                domain = msk.Domain.unbounded()
            elif lower is not None and upper is None:
                domain = msk.Domain.greaterThan(lower)
            elif lower is None and upper is not None:
                domain = msk.Domain.lessThan(upper)
            elif lower is not None and upper is not None:
                if lower == upper:
                    domain = msk.Domain.equalsTo(lower)
                elif self.ver >= 9:
                    # HACK: MOSEK 9 does not accept sparse (partial) range
                    #       domains anymore. The workaround triggers a MOSEK
                    #       warning, but there is no other way to pass such
                    #       variable bounds directly.
                    if isinstance(lower, msk.Matrix) \
                    or isinstance(upper, msk.Matrix):
                        lower, upper = self._bounds_pic2msk(picosVar, True)
                        if isinstance(lower, msk.Matrix):
                            lower = lower.getDataAsArray()
                        if isinstance(upper, msk.Matrix):
                            upper = upper.getDataAsArray()

                    domain = msk.Domain.inRange(lower, upper, shape)
                else:
                    domain = msk.Domain.inRange(lower, upper)

        # Refine the domain with the variable's type.
        if isinstance(picosVar, BinaryVariable):
            domain = msk.Domain.binary()
        elif isinstance(picosVar, IntegerVariable):
            domain = msk.Domain.integral(domain)

        # Create the MOSEK variable.
        mosekVar = self.int.variable(picosVar.name, shape, domain)

        # Map the PICOS variable to the MOSEK variable and vice versa.
        self.knownVariables[picosVar] = mosekVar
        self.knownVariables[mosekVar] = picosVar

        if self._debug():
            self._debug("Variable imported: {} → {}"
                .format(picosVar, " ".join(mosekVar.toString().split())))

    # TODO: This needs a test.
    def _import_variable_values(self, integralOnly=False):
        for picosVar in self.ext.variables.values():
            if integralOnly and not isinstance(
                    picosVar, (BinaryVariable, IntegerVariable)):
                continue

            if picosVar.valued:
                value = picosVar.internal_value

                if isinstance(value, cvxopt.spmatrix):
                    value = cvxopt.matrix(value)

                self.knownVariables[picosVar].setLevel(list(value))

    def _import_linear_constraint(self, picosCon):
        import mosek.fusion as msk

        assert isinstance(picosCon, AffineConstraint)

        # Separate constraint into a linear function and a constant.
        linear, bound = picosCon.bounded_linear_form()

        # Rewrite constraint in MOSEK types: The linear function is represented
        # as a MOSEK expression while the constant term becomes a MOSEK domain.
        linear = self._affinexp_pic2msk(linear[:])
        bound  = self._matrix_cvx2msk(bound._constant_coef)

        if picosCon.is_increasing():
            domain = msk.Domain.lessThan(bound)
        elif picosCon.is_decreasing():
            domain = msk.Domain.greaterThan(bound)
        elif picosCon.is_equality():
            domain = msk.Domain.equalsTo(bound)
        else:
            assert False, "Unexpected constraint relation."

        # Import the constraint.
        if picosCon.name is None:
            return self.int.constraint(linear, domain)
        else:
            return self.int.constraint(picosCon.name, linear, domain)

    def _import_socone_constraint(self, picosCon):
        import mosek.fusion as msk

        assert isinstance(picosCon, SOCConstraint)

        coneElement = self._mosek_vstack(
            msk.Expr.flatten(self._affinexp_pic2msk(picosCon.ub)),
            msk.Expr.flatten(self._affinexp_pic2msk(picosCon.ne)))

        # TODO: Remove zeros from coneElement[1:].

        return self.int.constraint(coneElement, msk.Domain.inQCone())

    def _import_rscone_constraint(self, picosCon):
        import mosek.fusion as msk

        assert isinstance(picosCon, RSOCConstraint)

        # MOSEK handles the vector [x₁; x₂; x₃] as input for a constraint of the
        # form ‖x₃‖² ≤ 2x₁x₂ whereas PICOS handles the expressions e₁, e₂ and e₃
        # for a constraint of the form ‖e₁‖² ≤ e₂e₃.
        # Neutralize MOSEK's additional factor of two by scaling e₂ and e₃ by
        # sqrt(0.5) each to obtain x₁ and x₂ respectively.
        scale = 0.5**0.5
        coneElement = self._mosek_vstack(
            msk.Expr.flatten(self._affinexp_pic2msk(scale * picosCon.ub1)),
            msk.Expr.flatten(self._affinexp_pic2msk(scale * picosCon.ub2)),
            msk.Expr.flatten(self._affinexp_pic2msk(picosCon.ne)))

        # TODO: Remove zeros from coneElement[2:].

        return self.int.constraint(coneElement, msk.Domain.inRotatedQCone())

    def _import_sdp_constraint(self, picosCon):
        import mosek.fusion as msk
        assert isinstance(picosCon, LMIConstraint)

        semiDefMatrix = self._affinexp_pic2msk(picosCon.psd)

        return self.int.constraint(semiDefMatrix, msk.Domain.inPSDCone())

    def _import_constraint(self, picosCon):
        import mosek.fusion as msk

        # HACK: Work around faulty MOSEK warnings (warning 705).
        import os
        with open(os.devnull, "w") as devnull:
            self.int.setLogHandler(devnull)

            if isinstance(picosCon, AffineConstraint):
                mosekCon = self._import_linear_constraint(picosCon)
            elif isinstance(picosCon, SOCConstraint):
                mosekCon = self._import_socone_constraint(picosCon)
            elif isinstance(picosCon, RSOCConstraint):
                mosekCon = self._import_rscone_constraint(picosCon)
            elif isinstance(picosCon, LMIConstraint):
                mosekCon = self._import_sdp_constraint(picosCon)
            else:
                assert False, "Unexpected constraint type: {}".format(
                    picosCon.__class__.__name__)

            self.int.setLogHandler(sys.stdout)

        # Map the PICOS constraint to the MOSEK constraint and vice versa.
        self.knownConstraints[picosCon] = mosekCon
        self.knownConstraints[mosekCon] = picosCon

        if self._debug():
            self._debug("Constraint imported: {} → {}".format(picosCon,
                " ".join(mosekCon.toString().split()) if not isinstance(
                mosekCon, msk.PSDConstraint) else mosekCon))

    def _import_objective(self):
        import mosek.fusion as msk

        picosSense, picosObjective = self.ext.no

        if picosSense == "min":
            mosekSense = msk.ObjectiveSense.Minimize
        else:
            assert picosSense == "max"
            mosekSense = msk.ObjectiveSense.Maximize

        mosekObjective = self._affinexp_pic2msk(picosObjective)

        self.int.objective(mosekSense, mosekObjective)

        if self._debug():
            self._debug(
                "Objective imported: {} {} → {} {}".format(
                picosSense, picosObjective, mosekSense,
                " ".join(mosekObjective.toString().split())))

    def _import_problem(self):
        import mosek.fusion as msk

        # Create a problem instance.
        self.int = msk.Model()
        self.int.setLogHandler(sys.stdout)

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
        for oldConstraint in self._removed_constraints():
            raise ProblemUpdateError(
                "MOSEK does not support removal of constraints.")

        for oldVariable in self._removed_variables():
            raise ProblemUpdateError(
                "MOSEK does not support removal of variables.")

        for newVariable in self._new_variables():
            self._import_variable(newVariable)

        for newConstraint in self._new_constraints():
            self._import_constraint(newConstraint)

        if self._objective_has_changed():
            self._import_objective()

    def _solve(self):
        import mosek.fusion as msk
        from mosek import objsense

        # MOSEK 8 has additional parameters and status codes.
        mosek8 = self.ver < 9

        # Reset options.
        # HACK: This is a direct access to MOSEK's internal Task object, which
        #       is necessary as the Fusion API has no call to reset options.
        # TODO: As soon as the Fusion API offers option reset, use it instead.
        self.int.getTask().setdefaults()
        self.int.optserverHost("")

        # verbosity
        self.int.setSolverParam("log", max(0, self.verbosity()))

        # abs_prim_fsb_tol
        if self.ext.options.abs_prim_fsb_tol is not None:
            value = self.ext.options.abs_prim_fsb_tol

            # Interior-point primal feasibility tolerances.
            for ptype in ("", "Co") + (("Qo",) if mosek8 else ()):
                self.int.setSolverParam("intpnt{}TolPfeas".format(ptype), value)

            # Simplex primal feasibility tolerance.
            self.int.setSolverParam("basisTolX", value)

            # Mixed-integer (primal) feasibility tolerance.
            self.int.setSolverParam("mioTolFeas", value)

        # abs_dual_fsb_tol
        if self.ext.options.abs_dual_fsb_tol is not None:
            value = self.ext.options.abs_dual_fsb_tol

            # Interior-point dual feasibility tolerances.
            for ptype in ("", "Co") + (("Qo",) if mosek8 else ()):
                self.int.setSolverParam("intpnt{}TolDfeas".format(ptype), value)

            # Simplex dual feasibility (optimality) tolerance.
            self.int.setSolverParam("basisTolS", value)

        # rel_dual_fsb_tol
        if self.ext.options.rel_dual_fsb_tol is not None:
            # Simplex relative dual feasibility (optimality) tolerance.
            self.int.setSolverParam("basisRelTolS",
                self.ext.options.rel_dual_fsb_tol)

        # rel_ipm_opt_tol
        if self.ext.options.rel_ipm_opt_tol is not None:
            value = self.ext.options.rel_ipm_opt_tol

            # Interior-point primal feasibility tolerances.
            for ptype in ("", "Co") + (("Qo",) if mosek8 else ()):
                self.int.setSolverParam(
                    "intpnt{}TolRelGap".format(ptype), value)

        # abs_bnb_opt_tol
        if self.ext.options.abs_bnb_opt_tol is not None:
            self.int.setSolverParam("mioTolAbsGap",
                self.ext.options.abs_bnb_opt_tol)

        # rel_bnb_opt_tol
        if self.ext.options.rel_bnb_opt_tol is not None:
            self.int.setSolverParam("mioTolRelGap",
                self.ext.options.rel_bnb_opt_tol)

        # integrality_tol
        if self.ext.options.integrality_tol is not None:
            self.int.setSolverParam("mioTolAbsRelaxInt",
                self.ext.options.integrality_tol)

        # max_iterations
        if self.ext.options.max_iterations is not None:
            value = self.ext.options.max_iterations
            self.int.setSolverParam("biMaxIterations",     value)
            self.int.setSolverParam("intpntMaxIterations", value)
            self.int.setSolverParam("simMaxIterations",    value)

        if self.ext.options.lp_node_method is not None \
        or self.ext.options.lp_root_method is not None:
            # TODO: Give Problem an interface for checks like this.
            _islp = isinstance(
                self.ext.no.function, AffineExpression) \
                and all([isinstance(constraint, AffineConstraint)
                for constraint in self.ext.constraints.values()])

            _lpm = {
                "interior": "intpnt" if _islp else "conic",
                "psimplex": "primalSimplex",
                "dsimplex": "dualSimplex"}

        # lp_node_method
        if self.ext.options.lp_node_method is not None:
            value = self.ext.options.lp_node_method
            assert value in _lpm, "Unexpected lp_node_method value."
            self.int.setSolverParam("mioNodeOptimizer", _lpm[value])

        # lp_root_method
        if self.ext.options.lp_root_method is not None:
            value = self.ext.options.lp_root_method
            assert value in _lpm, "Unexpected lp_root_method value."
            self.int.setSolverParam("mioRootOptimizer", _lpm[value])

        # timelimit
        if self.ext.options.timelimit is not None:
            value = float(self.ext.options.timelimit)
            self.int.setSolverParam("optimizerMaxTime", value)
            self.int.setSolverParam("mioMaxTime",       value)

        # max_fsb_nodes
        if self.ext.options.max_fsb_nodes is not None:
            self.int.setSolverParam("mioMaxNumSolutions",
                self.ext.options.max_fsb_nodes)

        # hotstart
        if self.ext.options.hotstart:
            # TODO: Check if valued variables (i.e. a hotstart) are utilized by
            #       MOSEK beyond mioConstructSol, and whether it makes sense to
            #       (1) also value continuous variables and (2) reset variable
            #       values when hotstart gets disabled again (see Gurobi).
            self.int.setSolverParam("mioConstructSol", "on")
            self._import_variable_values(integralOnly=True)

        # Handle MOSEK-specific options.
        for key, value in self.ext.options.mskfsn_params.items():
            try:
                self.int.setSolverParam(key, value)
            except msk.ParameterError as error:
                self._handle_bad_solver_specific_option_key(key, error)
            except ValueError as error:
                self._handle_bad_solver_specific_option_value(key, value, error)

        # Handle 'mosek_server' option.
        # FIXME: This produces unsolicited console output with MOSEK 9.2.
        if self.ext.options.mosek_server:
            self.int.optserverHost(self.ext.options.mosek_server)

        # Handle unsupported options.
        self._handle_unsupported_option("treememory")

        # Attempt to solve the problem.
        with self._header(), self._stopwatch():
            self.int.solve()

        # Retrieve primals.
        primals = {}
        if self.ext.options.primals is not False:
            for picosVar in self.ext.variables.values():
                mosekVar = self.knownVariables[picosVar]
                try:
                    primals[picosVar] = list(mosekVar.level())
                except msk.SolutionError:
                    primals[picosVar] = None

        # Retrieve duals.
        duals = {}
        if self.ext.options.duals is not False:
            for picosCon in self.ext.constraints.values():
                if isinstance(picosCon, DummyConstraint):
                    duals[picosCon] = cvxopt.spmatrix([], [], [], picosCon.size)
                    continue

                # Retrieve corresponding MOSEK constraint.
                mosekCon = self.knownConstraints[picosCon]

                # Retrieve its dual.
                try:
                    mosekDual = mosekCon.dual()
                except msk.SolutionError:
                    duals[picosCon] = None
                    continue

                # Devectorize the dual.
                # NOTE: Change from row-major to column-major order.
                size = picosCon.size
                picosDual = cvxopt.matrix(mosekDual, (size[1], size[0])).T

                # Adjust the dual based on constraint type.
                if isinstance(picosCon, (AffineConstraint, LMIConstraint)):
                    if not picosCon.is_increasing():
                        picosDual = -picosDual
                elif isinstance(picosCon, SOCConstraint):
                    picosDual = -picosDual
                elif isinstance(picosCon, RSOCConstraint):
                    # MOSEK handles the vector [x₁; x₂; x₃] as input for a
                    # constraint of the form ‖x₃‖² ≤ 2x₁x₂ whereas PICOS handles
                    # the expressions e₁, e₂ and e₃ for a constraint of the form
                    # ‖e₁‖² ≤ e₂e₃. MOSEK's additional factor of two was
                    # neutralized on import by scaling e₂ and e₃ by sqrt(0.5)
                    # each to obtain x₁ and x₂ respectively. Scale now also the
                    # dual returned by MOSEK to make up for this.
                    scale = 0.5**0.5
                    alpha = scale * picosDual[0]
                    beta  = scale * picosDual[1]
                    z     = list(-picosDual[2:])

                    # HACK: Work around a potential documentation bug in MOSEK:
                    #       The first two vector elements of the rotated
                    #       quadratic cone dual are non-positive (allowing for a
                    #       shorter notation in the linear part of their dual
                    #       representation) even though their definition of the
                    #       (self-dual) rotated quadratic cone explicitly states
                    #       that they are non-negative (as in PICOS).
                    alpha = -alpha
                    beta  = -beta

                    picosDual = cvxopt.matrix([alpha, beta] + z)
                else:
                    assert False, \
                        "Constraint type belongs to unsupported problem type."

                # Flip sign based on objective sense.
                if (self.int.getTask().getobjsense() == objsense.minimize):
                    picosDual = -picosDual

                duals[picosCon] = picosDual

        # Retrieve objective value.
        try:
            value = float(self.int.primalObjValue())
        except msk.SolutionError:
            value = None

        # Retrieve solution status.
        primalStatus  = self._solution_status_pic2msk(
            self.int.getPrimalSolutionStatus())
        dualStatus    = self._solution_status_pic2msk(
            self.int.getDualSolutionStatus())
        problemStatus = self._problem_status_pic2msk(self.int.getProblemStatus(
            msk.SolutionType.Default), not self.ext.is_continuous())

        # Correct two known bad solution states:
        if problemStatus == PS_INFEASIBLE and primalStatus == SS_UNKNOWN:
            primalStatus = SS_INFEASIBLE
        if problemStatus == PS_UNBOUNDED and dualStatus == SS_UNKNOWN:
            dualStatus = SS_INFEASIBLE

        return self._make_solution(
            value, primals, duals, primalStatus, dualStatus, problemStatus)

    def _solution_status_pic2msk(self, statusCode):
        from mosek.fusion import SolutionStatus as ss

        map = {
            ss.Undefined:       SS_EMPTY,
            ss.Unknown:         SS_UNKNOWN,
            ss.Optimal:         SS_OPTIMAL,
            ss.Feasible:        SS_FEASIBLE,
            ss.Certificate:     SS_INFEASIBLE,
            ss.IllposedCert:    SS_UNKNOWN
        }

        if self.ver <= 8:
            map.update({
                ss.NearOptimal:     SS_FEASIBLE,
                ss.NearFeasible:    SS_UNKNOWN,
                ss.NearCertificate: SS_UNKNOWN,
            })

        try:
            return map[statusCode]
        except KeyError:
            self._warn("The MOSEK Fusion solution status code {} is not known "
                "to PICOS.".format(statusCode))
            return SS_UNKNOWN

    def _problem_status_pic2msk(self, statusCode, integerProblem):
        from mosek.fusion import ProblemStatus as ps

        try:
            return {
                ps.Unknown:                     PS_UNKNOWN,
                ps.PrimalAndDualFeasible:       PS_FEASIBLE,
                ps.PrimalFeasible:
                    PS_FEASIBLE if integerProblem else PS_UNKNOWN,
                ps.DualFeasible:                PS_UNKNOWN,
                ps.PrimalInfeasible:            PS_INFEASIBLE,
                ps.DualInfeasible:              PS_UNBOUNDED,
                ps.PrimalAndDualInfeasible:     PS_INFEASIBLE,
                ps.IllPosed:                    PS_ILLPOSED,
                ps.PrimalInfeasibleOrUnbounded: PS_INF_OR_UNB
            }[statusCode]
        except KeyError:
            self._warn("The MOSEK Fusion problem status code {} is not known to"
                " PICOS.".format(statusCode))
            return PS_UNKNOWN


# --------------------------------------
__all__ = api_end(_API_START, globals())
