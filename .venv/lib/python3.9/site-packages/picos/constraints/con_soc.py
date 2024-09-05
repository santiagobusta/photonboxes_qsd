# ------------------------------------------------------------------------------
# Copyright (C) 2018-2020 Maximilian Stahlberg
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

"""Second order conce constraints."""

from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from .constraint import ConicConstraint

_API_START = api_start(globals())
# -------------------------------


class SOCConstraint(ConicConstraint):
    """Second order (:math:`2`-norm, Lorentz) cone membership constraint."""

    def __init__(self, normedExpression, upperBound, customString=None):
        """Construct a :class:`SOCConstraint`.

        :param ~picos.expressions.AffineExpression normedExpression:
            Expression under the norm.
        :param ~picos.expressions.AffineExpression upperBound:
            Upper bound on the normed expression.
        :param str customString:
            Optional string description.
        """
        from ..expressions import AffineExpression

        assert isinstance(normedExpression, AffineExpression)
        assert isinstance(upperBound, AffineExpression)
        assert len(upperBound) == 1

        # NOTE: len(normedExpression) == 1 is allowed even though this should
        #       rather be represented as an AbsoluteValueConstraint.

        self.ne = normedExpression
        self.ub = upperBound

        super(SOCConstraint, self).__init__(
            self._get_type_term(), customString, printSize=True)

    def _get_type_term(self):
        return "SOC"

    @cached_property
    def conic_membership_form(self):
        """Implement for :class:`~.constraint.ConicConstraint`."""
        from ..expressions import SecondOrderCone
        return (self.ub // self.ne.vec), SecondOrderCone(dim=(len(self.ne) + 1))

    @cached_property
    def unit_ball_form(self):
        r"""The constraint in Euclidean norm unit ball membership form.

        If this constraint has the form :math:`\lVert E(X) \rVert_F \leq c` with
        :math:`c > 0` constant and :math:`E(X)` an affine expression of a single
        (matrix) variable :math:`X` with :math:`y := \operatorname{vec}(E(X)) =
        A\operatorname{vec}(X) + b` for some invertible matrix :math:`A` and a
        vector :math:`b`, then we have :math:`\operatorname{vec}(X) = A^{-1}(y -
        b)` and we can write the elementwise vectorization of the constraint's
        feasible region as

        .. math::

            &\left\{\operatorname{vec}(X) \mid
                \lVert E(X) \rVert_F \leq c \right\} \\
            =~&\left\{\operatorname{vec}(X) \mid
                \lVert A\operatorname{vec}(X) + b \rVert_2 \leq c \right\} \\
            =~&\left\{A^{-1}(y - b) \mid \lVert y \rVert_2 \leq c \right\} \\
            =~&\left\{
                A^{-1}(y - b) \mid \lVert c^{-1}y \rVert_2 \leq 1
            \right\} \\
            =~&\left\{A^{-1}(cy - b) \mid \lVert y \rVert_2 \leq 1 \right\}.

        Therefor we can repose the constraint as two constraints:

        .. math::

            \lVert E(X) \rVert_F \leq c
                \Longleftrightarrow
            \exists y :
            \begin{cases}
                \operatorname{vec}(X) = A^{-1}(cy - b) \\
                \lVert y \rVert_2 \leq 1.
            \end{cases}

        This method returns the quadruple :math:`(X, A^{-1}(cy - b), y, B)`
        where :math:`y` is a fresh real variable vector (the same for subsequent
        calls) and :math:`B` is the Euclidean norm unit ball.

        :returns:
            A quadruple ``(X, aff_y, y, B)`` of type
            (:class:`~.variables.BaseVariable`,
            :class:`~.exp_affine.AffineExpression`,
            :class:`~.variables.RealVariable`, :class:`~picos.Ball`) such that
            the two constraints ``X.vec == aff_y`` and ``y << B`` combined are
            equivalent to this one.

        :raises NotImplementedError:
            If the expression under the norm does not reference exactly one
            variable or if that variable does not use a trivial vectorization
            format internally.

        :raises ValueError:
            If the upper bound is not constant, not positive, or if the matrix
            :math:`A` is not invertible.

        :Example:

        >>> import picos
        >>> A = picos.Constant("A", [[2, 0],
        ...                          [0, 1]])
        >>> x = picos.RealVariable("x", 2)
        >>> P = picos.Problem()
        >>> P.set_objective("max", picos.sum(x))
        >>> C = P.add_constraint(abs(A*x + 1) <= 10)
        >>> _ = P.solve(solver="cvxopt")
        >>> print(x)
        [ 1.74e+00]
        [ 7.94e+00]
        >>> Q = picos.Problem()
        >>> Q.set_objective("max", picos.sum(x))
        >>> x, aff_y, y, B, = C.unit_ball_form
        >>> _ = Q.add_constraint(x == aff_y)
        >>> _ = Q.add_constraint(y << B)
        >>> _ = Q.solve(solver="cvxopt")
        >>> print(x)
        [ 1.74e+00]
        [ 7.94e+00]
        >>> round(abs(P.value - Q.value), 4)
        0.0
        >>> round(y[0]**2 + y[1]**2, 4)
        1.0
        """
        from ..expressions import Ball, RealVariable
        from ..expressions.data import cvxopt_inverse
        from ..expressions.vectorizations import FullVectorization

        if len(self.ne.mutables) != 1:
            raise NotImplementedError("Unit ball membership form is only "
                "supported for second order conic constraints whose normed "
                "expression depends on exactly one mutable; found {} for {}."
                .format(len(self.ne.mutables), self))

        if not self.ub.constant:
            raise ValueError("The upper bound is not constant, so no unit ball "
                "form exists for {}.".format(self))

        c = self.ub
        if c.value <= 0:
            raise ValueError("The upper bound is not positive, so no unit ball "
                "form exists for {}.".format(self))

        X = tuple(self.ne.mutables)[0]

        if not isinstance(X._vec, FullVectorization):
            raise NotImplementedError(
                "The variable {} does not use a trivial vectorization format, "
                "so no unit ball form exists for {}.".format(X.name, self))

        A = self.ne._linear_coefs[X]
        b = self.ne.vec.cst

        if not A.size[0] == A.size[1]:
            raise ValueError("The dimensions dim({}) = {} and dim({}) = {} do "
                "not match, so no unit ball form exists for {}.".format(
                    X.name, X.dim, self.ne.string, len(self.ne), self))

        try:
            A_inverse = cvxopt_inverse(A)
        except ValueError as error:
            raise ValueError(
                "The linear operator applied to {} to form the linear part of "
                "{} is not bijective, so no unit ball form exists for {}."
                .format(X.name, self.ne.string, self)) from error

        y = RealVariable("__{}".format(X.name), X.dim)

        return X, A_inverse*(c*y - b), y, Ball()

    Subtype = namedtuple("Subtype", ("argdim",))

    def _subtype(self):
        return self.Subtype(len(self.ne))

    @classmethod
    def _cost(cls, subtype):
        return subtype.argdim + 1

    def _expression_names(self):
        yield "ne"
        yield "ub"

    def _str(self):
        return glyphs.le(glyphs.norm(self.ne.string), self.ub.string)

    def _get_size(self):
        return (len(self.ne) + 1, 1)

    def _get_slack(self):
        return self.ub.safe_value - abs(self.ne).safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
