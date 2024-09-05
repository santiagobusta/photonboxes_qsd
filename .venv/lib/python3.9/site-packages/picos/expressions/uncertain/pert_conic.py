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

"""Implements :class:`ConicPerturbationSet`."""

from collections import namedtuple

import cvxopt

from ... import glyphs, settings
from ...apidoc import api_end, api_start
from ...caching import cached_property
from ...constraints import SOCConstraint
from ..cone_product import ProductCone
from ..cone_soc import SecondOrderCone
from ..data import cvxopt_hcat, cvxopt_vcat
from ..exp_affine import Constant
from ..expression import NotValued
from ..set_ball import Ball
from ..variables import RealVariable
from ..vectorizations import FullVectorization
from .perturbation import Perturbation, PerturbationUniverse
from .uexp_affine import UncertainAffineExpression
from .uexpression import IntractableWorstCase

_API_START = api_start(globals())
# -------------------------------


class ConicPerturbationSet(PerturbationUniverse):
    r"""A conic description of a :class:`~.perturbation.Perturbation`.

    :Definition:

    An instance :math:`\Theta` of this class defines a perturbation parameter

    .. math::

        \theta \in \Theta = \{t \in \mathbb{R}^{m \times n}
            \mid \exists u \in \mathbb{R}^l
            : A\operatorname{vec}(t) + Bu + c \in K\}

    where :math:`m, n \in \mathbb{Z}_{\geq 1}`,
    :math:`l \in \mathbb{Z}_{\geq 0}`, :math:`A \in \mathbb{R}^{k \times mn}`,
    :math:`B \in \mathbb{R}^{k \times l}`, :math:`c \in \mathbb{R}^k` and
    :math:`K \subseteq \mathbb{R}^k` is a (product) cone for some
    :math:`k \in \mathbb{Z}_{\geq 1}`.

    :Usage:

    Obtaining :math:`\theta` is done in a number of steps:

    1. Create an instance of this class (see :meth:`__init__`).
    2. Access :attr:`element` to obtain a regular, fresh
       :class:`~.variables.RealVariable` representing :math:`t`.
    3. Define :math:`\Theta` through any number of regular PICOS constraints
       that depend only on :math:`t` and that have a conic representation by
       passing the constraints to :meth:`bound`.
    4. Call :meth:`compile` to obtain a handle to the
       :class:`~.perturbation.Perturbation` :math:`\theta`.
    5. You can now use :math:`\theta` to build instances of
       :class:`~.uexp_affine.UncertainAffineExpression` and derived constraint
       types.

    It is best practice to assign :math:`t` to a Python variable and overwrite
    that variable with :math:`\theta` on compilation.

    Alternatively, you can obtain a compiled :math:`\Theta` from the factory
    method :meth:`from_constraints` and access :math:`\theta` via
    :attr:`parameter`.

    :Example:

    >>> from picos import Constant, Norm, RealVariable
    >>> from picos.uncertain import ConicPerturbationSet
    >>> S = ConicPerturbationSet("P", (4, 4))
    >>> P = S.element; P  # Obtain a temporary parameter to describe S with.
    <4×4 Real Variable: P>
    >>> S.bound(Norm(P, float("inf")) <= 1)  # Confine each element to [-1,1].
    >>> S.bound(Norm(P, 1) <= 4); S  # Allow a perturbation budget of 4.
    <4×4 Conic Perturbation Set: {P : ‖P‖_max ≤ 1 ∧ ‖P‖_sum ≤ 4}>
    >>> P = S.compile(); P  # Compile the set and obtain the actual parameter.
    <4×4 Perturbation: P>
    >>> A = Constant("A", range(16), (4, 4))
    >>> U = A + P; U  # Define an uncertain data matrix.
    <4×4 Uncertain Affine Expression: A + P>
    >>> x = RealVariable("x", 4)
    >>> U*x  # Define an uncertain affine expression.
    <4×1 Uncertain Affine Expression: (A + P)·x>
    """

    def __init__(self, parameter_name, shape=(1, 1)):
        """Create a :class:`ConicPerturbationSet`.

        :param str parameter_name:
            Name of the parameter that lives in the set.

        :param shape:
            The shape of a vector or matrix perturbation.
        :type shape:
            int or tuple or list
        """
        from ...modeling import Problem

        self._compiled = False
        self._element = RealVariable(parameter_name, shape)
        self._parameter = Perturbation(self, parameter_name, shape)
        self._bounds = Problem()

    Subtype = namedtuple("Subtype", (
        "param_dim", "cone_type", "dual_cone_type", "has_B"))

    def _subtype(self):
        return self.Subtype(
            self._parameter.dim,
            self.K.type,
            self.K.dual_cone.type,
            self.B is not None)

    @classmethod
    def from_constraints(cls, parameter_name, *constraints):
        """Create a :class:`ConicPerturbationSet` from constraints.

        The constraints must concern a single regular decision variable that
        plays the role of the :attr:`element` :math:`t`. This variable is not
        stored or modified and can be reused in a different context.

        :param str parameter_name:
            Name of the parameter that lives in the set.

        :param constraints:
            A parameter sequence of constraints that concern a single regular
            decision variable whose internal vectorization is trivial (its
            dimension must match the product over its shape) and that have a
            conic representation.

        :raises ValueError:
            If the constraints do not all concern the same single variable.

        :raises TypeError:
            If the variable uses a nontrivial vectorization format or if the
            constraints do not all have a conic representation.

        :Example:

        >>> from picos.expressions.uncertain import ConicPerturbationSet
        >>> from picos import RealVariable
        >>> x = RealVariable("x", 4)
        >>> T = ConicPerturbationSet.from_constraints("t", abs(x) <= 2, x >= 0)
        >>> print(T)
        {t : ‖t‖ ≤ 2 ∧ t ≥ 0}
        >>> print(repr(T.parameter))
        <4×1 Perturbation: t>
        """
        T, t, seen_variable = None, None, None

        for constraint in constraints:
            if len(constraint.variables) != 1:
                raise ValueError("The constraint {} does not concern exactly "
                    "one variable.".format(constraint))

            variable = next(iter(constraint.variables))

            if not isinstance(variable._vec, FullVectorization):
                raise TypeError("The variable {} cannot be used to construct a "
                    "{} from constraints because it uses a nontrivial "
                    "vectorization format.".format(variable, cls.name))

            if not seen_variable:
                seen_variable = variable
                T = cls(parameter_name, variable.shape)
                t = T.element
            elif variable is not seen_variable:
                raise ValueError("The constraints do not concern the same "
                    "single variable (found {} and {})."
                    .format(seen_variable.name, variable.name))

            T.bound(constraint.replace_mutables({variable: t}))

        T.compile()
        return T

    def __str__(self):
        return glyphs.set(glyphs.sep(self._parameter.name,
            glyphs.and_("", "").join(str(con)
            for con in self._bounds.constraints.values())))

    @classmethod
    def _get_type_string_base(cls):
        return "Conic Perturbation Set"

    def __repr__(self):
        return glyphs.repr2("{} {}".format(glyphs.shape(self._element.shape),
            self._get_type_string_base()), self.__str__())

    def _forbid_compiled(self):
        if self._compiled:
            raise RuntimeError(
                "The perturbation set has already been compiled.")

    def _require_compiled(self):
        if not self._compiled:
            raise RuntimeError(
                "The perturbation set has not yet been compiled.")

    @property
    def element(self):
        r"""The perturbation element :math:`t` describing the set.

        This is a regular :class:`~.variables.RealVariable` that you can use to
        create constraints to pass to :meth:`bound`. You can then obtain the
        "actual" perturbation parameter :math:`\theta` to use in expressions
        alongside your decision variaiables using :meth:`compile`.

        .. warning::

            If you use this object instead of :attr:`parameter` to define a
            decision problem then it will act as a regular decision variable,
            which is probably not what you want.

        :raises RuntimeError:
            If the set was already compiled.
        """
        self._forbid_compiled()
        return self._element

    def bound(self, constraint):
        r"""Add a constraint that bounds :math:`t`.

        The constraints do not need to be conic but they need to have a *conic
        representation*, which may involve any number of auxiliary variables.
        For instance, given a constant *uncertainty budget* :math:`b`, you may
        add the bound :math:`\lVert t \rVert_1 \leq b` (via
        ``picos.Norm(t, 1) <= b``) which can be represented in conic form as

        .. math::

            &\exists v \in \mathbb{R}^{\operatorname{dim}(t)}
                : -t \leq v \land t \leq v \land \mathbf{1}^Tv \leq b \\
            \Longleftrightarrow~
            &\exists v \in \mathbb{R}^{\operatorname{dim}(t)} :
            \begin{pmatrix}
                v + t \\
                v - t \\
                b - \mathbf{1}^Tv
            \end{pmatrix} \in \mathbb{R}_{\geq 0}^{2\operatorname{dim}(t) + 1}.

        The auxiliary variable :math:`v` then plays the role of (a slice of)
        :math:`u` in the formal definition of :math:`\Theta`.

        When you are done adding bounds, you can obtain :math:`\theta` using
        :meth:`compile`.

        :raises RuntimeError:
            If the set was already compiled.
        """
        self._forbid_compiled()

        vars = constraint.variables

        if len(vars) != 1 or next(iter(vars)) != self._element:
            raise ValueError(
                "The constraint {} does not bound (only) the perturbation "
                "element {}.".format(constraint, self._element.string))

        self._bounds.add_constraint(constraint)

    def compile(self, validate_feasibility=False):
        r"""Compile the set and return :math:`\theta`.

        Internally, this computes the matrices :math:`A` and :math:`B`, the
        vector :math:`c` and the (product) cone :math:`K`.

        :param bool validate_feasibility:
            Whether to solve the feasibility problem associated with the bounds
            on :math:`t` to verify that :math:`\Theta` is nonempty.

        :returns:
            An instance of :class:`~.perturbation.Perturbation`.

        :raises RuntimeError:
            If the set was already compiled.

        :raises TypeError:
            If the bound constraints could not be put into conic form.

        :raises ValueError:
            If :math:`\Theta` could not be verified to be nonempty (needs
            ``validate_feasibility=True``).
        """
        from ...constraints import DummyConstraint
        from ...modeling import NoStrategyFound

        self._forbid_compiled()

        # If no bounds are given, add a DummyConstraint to mark t free.
        if not self._bounds.constraints:
            self._bounds.add_constraint(DummyConstraint(self._element))

        # Transform all bounds into conic form.
        try:
            conic_bounds = self._bounds.conic_form
        except NoStrategyFound as error:
            raise TypeError("Could not find a conic representation for all of "
                "the bounds on {}.".format(self._element.string)) from error

        # Validate bound feasibility if requested.
        if validate_feasibility:
            from ...modeling.solution import SS_OPTIMAL, SS_FEASIBLE

            solution = conic_bounds.solve(
                primals=None, apply_solution=False, **settings.INTERNAL_OPTIONS)

            status = solution.primalStatus

            if status not in (SS_OPTIMAL, SS_FEASIBLE):
                raise ValueError("Could not verify that the bounds on {} are "
                    "feasible: The solver {} reports a primal solution state of"
                    " {} for the associated feasibility problem.".format(
                    self._element.string, solution.solver, status))

        # Form a virtual variable u from the auxiliary variables.
        u = [var for var in conic_bounds.variables.values()
             if var is not self._element]

        # Convert auxiliary variable bounds to additional affine constraints.
        conic_bounds.add_list_of_constraints([
            var.bound_constraint for var in u if var.bound_constraint])

        # Reformulate bounds with respect to a single product cone K.
        A, B, c, K = [], [], [], []
        for constraint in conic_bounds.constraints.values():
            member, cone = constraint.conic_membership_form

            if self._element in member._linear_coefs:
                A.append(member._linear_coefs[self._element])
            else:
                A.append(cvxopt.spmatrix(
                    [], [], [], size=(cone.dim, self._element.dim)))

            if u:
                B.append(cvxopt_hcat([
                    member._linear_coefs[var] if var in member._linear_coefs
                    else cvxopt.spmatrix([], [], [], size=(cone.dim, var.dim))
                    for var in u]))

            c.append(member._constant_coef)
            K.append(cone)

        self._A = Constant("A", cvxopt_vcat(A))
        self._B = Constant("B", cvxopt_vcat(B)) if u else None
        self._c = Constant("c", cvxopt_vcat(c))
        self._K = ProductCone(*K)

        # Store the bounds in conic form to speed up worst_case.
        self._conic_bounds = conic_bounds

        self._compiled = True
        return self._parameter

    @property
    def distributional(self):
        """Implement for :class:`~.perturbation.PerturbationUniverse`."""
        return False

    @property
    def parameter(self):
        r"""The perturbation parameter :math:`\theta` living in the set.

        This is the object returned by :meth:`compile`.

        :raises RuntimeError:
            If the set has not been compiled.
        """
        self._require_compiled()
        return self._parameter

    @property
    def A(self):
        r"""The compiled matrix :math:`A`.

        :raises RuntimeError:
            If the set has not been compiled.
        """
        self._require_compiled()
        return self._A

    @property
    def B(self):
        r"""The compiled matrix :math:`B` or :obj:`None` if :math:`l = 0`.

        :raises RuntimeError:
            If the set has not been compiled.
        """
        self._require_compiled()
        return self._B

    @property
    def c(self):
        r"""The compiled vector :math:`c`.

        :raises RuntimeError:
            If the set has not been compiled.
        """
        self._require_compiled()
        return self._c

    @property
    def K(self):
        r"""The compiled (product) cone :math:`K`.

        :raises RuntimeError:
            If the set has not been compiled.
        """
        self._require_compiled()
        return self._K

    def worst_case(self, scalar, direction):
        """Implement for :class:`~.perturbation.PerturbationUniverse`."""
        from ...modeling import SolutionFailure

        self._require_compiled()
        self._check_worst_case_argument_scalar(scalar)
        self._check_worst_case_argument_direction(direction)

        p = self._parameter
        P = self._conic_bounds.copy()
        x = P.variables[p.name]
        f = scalar.replace_mutables({p: x})

        self._check_worst_case_f_and_x(f, x)

        if (direction == "min" and not f.convex) \
        or (direction == "max" and not f.concave):
            raise IntractableWorstCase("PICOS refuses to compute {}({}) for {} "
                "as this is a nonconvex problem.".format(direction, f.string,
                glyphs.element(p.name, self)))

        P.set_objective(direction, f if direction != "find" else None)

        try:
            P.solve(**settings.INTERNAL_OPTIONS)
            return f.safe_value, x.safe_value
        except (SolutionFailure, NotValued) as error:
            raise RuntimeError("Failed to compute {}({}) for {}.".format(
                direction, f.string, glyphs.element(x.string, self))) from error

    @cached_property
    def unit_ball_form(self):
        """A recipe to repose from ellipsoidal to unit norm ball uncertainty.

        If the set is :attr:`ellipsoidal`, then this is a pair ``(U, M)`` where
        ``U`` is a :class:`~.pert_conic.UnitBallPerturbationSet` and ``M`` is a
        dictionary mapping the old :attr:`parameter` to an affine expression of
        the new parameter that can represent the old parameter in an expression
        (see :meth:`~.expression.Expression.replace_mutables`).
        The mapping ``M`` is empty if and only if the perturbation set is
        already an instance of :class:`~.pert_conic.UnitBallPerturbationSet`.

        If the uncertainty set is not ellipsoidal, then this is :obj:`None`.

        See also :attr:`SOCConstraint.unit_ball_form
        <.con_soc.SOCConstraint.unit_ball_form>`.

        :Example:

        >>> from picos import Problem, RealVariable, sum
        >>> from picos.uncertain import ConicPerturbationSet
        >>> # Create a conic perturbation set and a refinement recipe.
        >>> T = ConicPerturbationSet("t", (2, 2))
        >>> T.bound(abs(([[1, 2], [3, 4]] ^ T.element) + 1) <= 10)
        >>> t = T.compile()
        >>> U, mapping = T.unit_ball_form
        >>> print(U)
        {t' : ‖t'‖ ≤ 1}
        >>> print(mapping)
        {<2×2 Perturbation: t>: <2×2 Uncertain Affine Expression: t(t')>}
        >>> # Define and solve a conically uncertain LP.
        >>> X = RealVariable("X", (2, 2))
        >>> P = Problem()
        >>> P.set_objective("max", sum(X))
        >>> _ = P.add_constraint(X + 2*t <= 10)
        >>> print(repr(P.parameters["t"].universe))
        <2×2 Conic Perturbation Set: {t : ‖[2×2]⊙t + [1]‖ ≤ 10}>
        >>> _ = P.solve(solver="cvxopt")
        >>> print(X)
        [-8.00e+00  1.00e+00]
        [ 4.00e+00  5.50e+00]
        >>> # Refine the problem to a unit ball uncertain LP.
        >>> Q = Problem()
        >>> Q.set_objective("max", sum(X))
        >>> _ = Q.add_constraint(X + 2*mapping[t] <= 10)
        >>> print(repr(Q.parameters["t'"].universe))
        <2×2 Unit Ball Perturbation Set: {t' : ‖t'‖ ≤ 1}>
        >>> _ = Q.solve(solver="cvxopt")
        >>> print(X)
        [-8.00e+00  1.00e+00]
        [ 4.00e+00  5.50e+00]
        """
        self._require_compiled()

        if self._B is not None:
            return None

        K = self._K.refined

        if not isinstance(K, SecondOrderCone):
            return None

        C = self._A*self._element.vec + self._c << K

        assert isinstance(C, SOCConstraint)

        try:
            X, aff_y, y, _ = C.unit_ball_form
        except ValueError:
            return None

        assert X is self._element

        U = UnitBallPerturbationSet("{}'".format(X.name), X.shape)
        u = U.parameter
        replacement = UncertainAffineExpression("{}({})".format(X.name, u.name),
            X.shape, {(u,): aff_y._linear_coefs[y], (): aff_y._constant_coef})

        return U, {self._parameter: replacement}

    @property
    def ellipsoidal(self):
        """Whether the perturbation set is an ellipsoid.

        If this is true, then a :attr:`unit_ball_form` is available.
        """
        return bool(self.unit_ball_form)


class UnitBallPerturbationSet(ConicPerturbationSet):
    r"""Represents perturbation in an Euclidean or Frobenius unit norm ball.

    This is a :class:`~.pert_conic.ConicPerturbationSet` with fixed form

    .. math::

        \{t \in \mathbb{R}^{m \times n} \mid \lVert t \rVert_F \leq 1\}.

    After initialization, you can obtain the parameter using
    :attr:`~.pert_conic.ConicPerturbationSet.parameter`.
    """

    def __init__(self, parameter_name, shape=(1, 1)):
        """See :meth:`ConicPerturbationSet.__init__`."""
        ConicPerturbationSet.__init__(self, parameter_name, shape)
        self.bound(self._element << Ball())
        self.compile()

    @classmethod
    def _get_type_string_base(cls):
        return "Unit Ball Perturbation Set"

    @property
    def unit_ball_form(self):
        """Overwrite :attr:`ConicPerturbationSet.unit_ball_form`."""
        return self, {}


# --------------------------------------
__all__ = api_end(_API_START, globals())
