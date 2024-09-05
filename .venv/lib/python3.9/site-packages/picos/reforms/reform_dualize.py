# ------------------------------------------------------------------------------
# Copyright (C) 2020 Maximilian Stahlberg
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

"""Implementation of :class:`Dualization`."""

from itertools import chain

import cvxopt

from ..apidoc import api_end, api_start
from ..constraints import (AffineConstraint, DummyConstraint, LMIConstraint,
                           RSOCConstraint, SOCConstraint)
from ..expressions import AffineExpression, Constant
from ..expressions.algebra import rsoc, soc
from ..expressions.variables import (CONTINUOUS_VARTYPES, RealVariable,
                                     SymmetricVariable)
from ..modeling import Problem
from ..modeling.footprint import Footprint, Specification
from ..modeling.solution import PS_INFEASIBLE, PS_UNBOUNDED, Solution
from .reformulation import Reformulation

_API_START = api_start(globals())
# -------------------------------


class Dualization(Reformulation):
    """Lagrange dual problem reformulation."""

    SUPPORTED = Specification(
        objectives=[
            AffineExpression],
        variables=CONTINUOUS_VARTYPES,
        constraints=[
            DummyConstraint,
            AffineConstraint,
            SOCConstraint,
            RSOCConstraint,
            LMIConstraint])

    @classmethod
    def supports(cls, footprint):
        """Implement :meth:`~.reformulation.Reformulation.supports`."""
        return footprint.options.dualize and footprint in cls.SUPPORTED

    @classmethod
    def predict(cls, footprint):
        """Implement :meth:`~.reformulation.Reformulation.predict`."""
        vars = []
        cons = []

        # Objective direction.
        dir = "min" if footprint.direction == "max" else "max"

        # Objective function.
        # HACK: Conversion adds a zero-fixed variable to the objective so that
        #       it is never constant. This makes prediction succeed in any case.
        obj = AffineExpression.make_type(
            shape=(1, 1), constant=False, nonneg=False)

        # Variables and their bounds.
        bound_cons = []
        for vartype, k in footprint.variables.items():
            cons.append((AffineConstraint.make_type(
                dim=vartype.subtype.dim, eq=True), k))

            if vartype.subtype.bnd:
                bound_cons.append((AffineConstraint.make_type(
                    dim=vartype.subtype.bnd, eq=False), k))

        # Constraints by type.
        for contype, k in chain(footprint.constraints.items(), bound_cons):
            if contype.clstype is DummyConstraint:
                pass
            elif contype.clstype is AffineConstraint:
                d = contype.subtype.dim

                # Dual variable with conic membership as bounds.
                vars.append((RealVariable.make_var_type(
                    dim=d, bnd=0 if contype.subtype.eq else d), k))
            elif contype.clstype is SOCConstraint:
                n = contype.subtype.argdim
                d = n + 1

                # Dual variable.
                vars.append((RealVariable.make_var_type(dim=d, bnd=0), k))

                # Conic membership constraint.
                cons.append((SOCConstraint.make_type(argdim=n), k))
            elif contype.clstype is RSOCConstraint:
                n = contype.subtype.argdim
                d = n + 2

                # Dual variable.
                vars.append((RealVariable.make_var_type(dim=d, bnd=0), k))

                # Conic membership constraint.
                cons.append((RSOCConstraint.make_type(argdim=n), k))
            elif contype.clstype is LMIConstraint:
                n = contype.subtype.diag
                d = n*(n + 1)//2

                # Dual variable.
                vars.append((SymmetricVariable.make_var_type(dim=d, bnd=0), k))

                # Conic membership constraint.
                cons.append((LMIConstraint.make_type(diag=n), k))
            else:
                assert False, "Unexpected constraint type."

        # HACK: See above.
        vars.append((RealVariable.make_var_type(dim=1, bnd=2), 1))

        # Option changes.
        nd_opts = footprint.options.updated(
            dualize=False,
            primals=footprint.options.duals,
            duals=footprint.options.primals
        ).nondefaults

        return Footprint.from_types(dir, obj, vars, cons, nd_opts)

    def forward(self):
        """Implement :meth:`~.reformulation.Reformulation.forward`."""
        self._pc2dv = {}  # Maps primal constraints to dual variables.
        self._pv2dc = {}  # Maps primal variables to dual constraints.

        P = self.input

        # Create the dual problem from scratch.
        D = self.output = Problem(copyOptions=P.options)

        # Reset the 'dualize' option so that solvers can load the dual.
        D.options.dualize = False

        # Swap 'primals' and 'duals' option.
        D.options.primals = P.options.duals
        D.options.duals = P.options.primals

        # Retrieve the primal objective in standard form.
        original_primal_direction, primal_objective = P.objective.normalized
        if original_primal_direction == "max":
            primal_objective = -primal_objective

        # Start building the dual objective function.
        dual_objective = primal_objective.cst

        # Start building the dual linear equalities for each primal variable.
        obj_coefs = primal_objective._linear_coefs
        dual_equality_terms = {var: -Constant(obj_coefs[var].T)
            if var in obj_coefs else AffineExpression.zero(var.dim)
            for var in P.variables.values()}

        # Turn variable bounds into affine inequality constraints.
        # NOTE: If bound-free variables are required by another reformulation or
        #       solver in the future, there should be a reformulation for this.
        bound_cons = []
        for primal_var in P.variables.values():
            bound_constraint = primal_var.bound_constraint
            if bound_constraint:
                bound_cons.append(bound_constraint)

        # Handle the primal constraints.
        # NOTE: Constraint conversion works as follows:
        #       1. Transform constraint into conic standard form:
        #          fx = Ax-b >_K 0.
        #       2. Create a dual variable for the constraint.
        #       3. Constrain the dual variable to be a member of the dual
        #          cone K*.
        #       4. Extend the dual's linear equalities for each primal variable.
        #       5. Extend the dual's objective function.
        # TODO: Consider adding a ConicConstraint abstract subclass of
        #       Constraint with a method conic_form that returns a pair of cone
        #       member (affine expression) and cone. The conic standard form
        #       function f(x) and the dual cone can then be obtained by negation
        #       and through a new Cone.dual method, respectively.
        for primal_con in chain(P.constraints.values(), bound_cons):
            if isinstance(primal_con, DummyConstraint):
                pass
            elif isinstance(primal_con, AffineConstraint):
                # Transform constraint into conic standard form.
                fx = primal_con.ge0.vec

                # Create a dual variable for the constraint.
                # NOTE: Conic membership is expressed through variable bounds.
                self._pc2dv[primal_con] = dual_var = RealVariable(
                    "aff_{}".format(primal_con.id), len(fx),
                    lower=0 if primal_con.is_inequality() else None)

                # Extend the dual's linear equalities for each primal variable.
                for var, coef in fx._linear_coefs.items():
                    dual_equality_terms[var] += coef.T*dual_var

                # Extend the dual's objective function.
                dual_objective -= fx._constant_coef.T*dual_var
            elif isinstance(primal_con, SOCConstraint):
                # Transform constraint into conic standard form.
                fx = (primal_con.ub // primal_con.ne.vec)

                # Create a dual variable for the constraint.
                self._pc2dv[primal_con] = dual_var = RealVariable(
                    "soc_{}".format(primal_con.id), len(fx))

                # Constrain the dual variable to be a member of the dual cone.
                D.add_constraint(dual_var << soc())

                # Extend the dual's linear equalities for each primal variable.
                for var, coef in fx._linear_coefs.items():
                    dual_equality_terms[var] += coef.T*dual_var

                # Extend the dual's objective function.
                dual_objective -= fx._constant_coef.T*dual_var
            elif isinstance(primal_con, RSOCConstraint):
                # Transform constraint into conic standard form.
                fx = (primal_con.ub1 // primal_con.ub2 // primal_con.ne.vec)

                # Create a dual variable for the constraint.
                self._pc2dv[primal_con] = dual_var = RealVariable(
                    "rso_{}".format(primal_con.id), len(fx))

                # Constrain the dual variable to be a member of the dual cone.
                D.add_constraint(dual_var << rsoc(4))

                # Extend the dual's linear equalities for each primal variable.
                for var, coef in fx._linear_coefs.items():
                    dual_equality_terms[var] += coef.T*dual_var

                # Extend the dual's objective function.
                dual_objective -= fx._constant_coef.T*dual_var
            elif isinstance(primal_con, LMIConstraint):
                # Transform constraint into conic standard form.
                fx = primal_con.psd

                # Create a dual variable for the constraint.
                self._pc2dv[primal_con] = dual_var = SymmetricVariable(
                    "lmi_{}".format(primal_con.id), fx.shape)

                # Constrain the dual variable to be a member of the dual cone.
                D.add_constraint(dual_var >> 0)

                # Extend the dual's linear equalities for each primal variable.
                for var, coef in fx._linear_coefs.items():
                    dual_equality_terms[var] += coef.T*dual_var.vec

                # Extend the dual's objective function.
                dual_objective -= fx._constant_coef.T*dual_var.vec
            else:
                assert False, "Unexpected constraint type."

        # Add the finished linear equalities for each variable.
        for var, term in dual_equality_terms.items():
            self._pv2dc[var] = D.add_constraint(term == 0)

        # Adjust dual optimization direction to obtain the duality property.
        if original_primal_direction == "max":
            dual_direction = "min"
            dual_objective = -dual_objective
        else:
            dual_direction = "max"

        # HACK: Add a zero-fixed variable to the objective so that it is never
        #       constant. This makes prediction succeed in any case.
        dual_objective += RealVariable("zero", 1, lower=0, upper=0)

        # Set the finished dual objective.
        D.objective = dual_direction, dual_objective

    def update(self):
        """Implement :meth:`~.reformulation.Reformulation.update`."""
        # TODO: Implement updates to an existing dualization. This is optional.
        raise NotImplementedError

    def backward(self, solution):
        """Implement :meth:`~.reformulation.Reformulation.backward`."""
        P = self.input

        # Require primal values to be properly vectorized.
        if not solution.vectorizedPrimals:
            raise NotImplementedError(
                "Solving with dualize=True is not supported with solvers that "
                "produce unvectorized primal solutions.")

        # Retrieve primals for the primal problem.
        primals = {}
        for var in P.variables.values():
            con = self._pv2dc[var]

            if con in solution.duals and solution.duals[con] is not None:
                primal = -solution.duals[con]
            else:
                primal = None

            if primal is not None:
                primals[var] = cvxopt.matrix(primal)
            else:
                primals[var] = None

        # Retrieve duals for the primal problem.
        duals = {}
        for con in P.constraints.values():
            if isinstance(con, DummyConstraint):
                continue

            var = self._pc2dv[con]
            dual = solution.primals[var] if var in solution.primals else None

            if dual is not None:
                duals[con] = var._vec.devectorize(cvxopt.matrix(dual))
            else:
                duals[con] = None

        # Swap infeasible/unbounded for a problem status.
        if solution.problemStatus == PS_UNBOUNDED:
            problemStatus = PS_INFEASIBLE
        elif solution.problemStatus == PS_INFEASIBLE:
            problemStatus = PS_UNBOUNDED
        else:
            problemStatus = solution.problemStatus

        return Solution(
            primals=primals,
            duals=duals,
            problem=solution.problem,
            solver=solution.solver,
            primalStatus=solution.dualStatus,  # Intentional swap.
            dualStatus=solution.primalStatus,  # Intentional swap.
            problemStatus=problemStatus,
            searchTime=solution.searchTime,
            info=solution.info,
            vectorizedPrimals=True,
            reportedValue=solution.reportedValue)


# --------------------------------------
__all__ = api_end(_API_START, globals())
