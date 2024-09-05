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

"""Implements :class:`Ellipsoid`."""

import operator
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from ..constraints import SOCConstraint
from .data import convert_operands, cvxopt_inverse
from .exp_affine import AffineExpression, Constant
from .expression import Expression
from .set import Set

_API_START = api_start(globals())
# -------------------------------


class Ellipsoid(Set):
    r"""An affine transformation of the Euclidean unit ball.

    :Definition:

    For :math:`n \in \mathbb{Z}_{\geq 1}`, :math:`A \in \mathbb{R}^{n \times n}`
    invertible and :math:`c \in \mathbb{R}^n`, an instance of this class
    represents the set

    .. math::

        & \{Ax + c \mid \lVert x \rVert_2 \leq 1\} \\
        =~& \{x \mid \lVert A^{-1}(x - c) \rVert_2 \leq 1\} \\
        =~& \{x \mid \lVert x - c \rVert_{(A^{-1})^T A^{-1}} \leq 1\}.

    Unlike most sets, instances of this class offer a limited set of algebraic
    operations that are generalized from expressions to sets in the natural way.
    In particular, you can add or substract constant vectors of matching
    dimension and apply matrix multiplication from the left hand side, both of
    which will act on the term :math:`Ax + c` in the definition above.

    :Example:

    >>> from picos import Ellipsoid, RealVariable
    >>> Ellipsoid(3)  # Three-dimensional Euclidean unit ball.
    <Centered Unit Ball: {I·x : ‖x‖ ≤ 1}>
    >>> Ellipsoid(3, range(9))  # Linear transformation of the unit ball.
    <Centered Ellipsoid: {[3×3]·x : ‖x‖ ≤ 1}>
    >>> Ellipsoid(3, "2I", 1)  # Offset ball of radius two.
    <Offset Ellipsoid: {2·I·x + [1] : ‖x‖ ≤ 1}>
    >>> 2*Ellipsoid(3) + 1  # The same using algebraic operations.
    <Offset Ellipsoid: {2·I·x + [1] : ‖x‖ ≤ 1}>
    >>> x = RealVariable("x", 3)
    >>> (2*x + range(3)) << (4*Ellipsoid(3) + 5)  # Constraint creation.
    <4×1 SOC Constraint: ‖(4·I)^(-1)·(2·x + [3×1] - [5])‖ ≤ 1>

    .. note::

        Due to significant differences in scope, :class:`Ellipsoid` is not a
        superclass of :class:`~.set_ball.Ball` even though both classes can
        represent Euclidean balls around the origin.
    """

    def __init__(self, n, A="I", c=0):
        """Construct an ellipsoid.

        :param int n:
            Dimensionality :math:`n` of the ellipsoid.

        :param A:
            Invertible linear transformation matrix :math:`A`.
        :type A:
            :class:`~.exp_affine.AffineExpression` or recognized by
            :func:`~.data.load_data`

        :param c:
            The center :math:`c` of the ellispoid.
        :type c:
            :class:`~.exp_affine.AffineExpression` or recognized by
            :func:`~.data.load_data`

        .. warning::

            Invertibility of :math:`A` is not checked on instanciation.
            If :math:`A` is singular, a :exc:`RuntimeError` is raised once the
            inverse is needed.
        """
        if not isinstance(n, int):
            raise TypeError("Dimensionality must be an integer.")

        if n < 1:
            raise ValueError("Dimensionality must be positive.")

        # Load A.
        if not isinstance(A, Expression):
            A = AffineExpression.from_constant(A, shape=(n, n))
        else:
            A = A.refined

            if not isinstance(A, AffineExpression) or not A.constant:
                raise TypeError("A must be a constant real matrix.")

            if A.shape != (n, n):
                raise TypeError("A must be a {} matrix, got {} instead."
                    .format(glyphs.shape((n, n)), glyphs.shape(A.shape)))

        # Load c.
        if not isinstance(c, Expression):
            c = AffineExpression.from_constant(c, shape=n)
        else:
            c = c.refined

            if not isinstance(c, AffineExpression) or not c.constant:
                raise TypeError("c must be a constant real vector.")

            if c.shape != (n, 1):
                raise TypeError(
                    "c must be a {}-dimensional column vector.".format(n))

        self._n = n
        self._A = A
        self._c = c

        typeStr = (
            ("Centered " if c.is0 else "Offset ")
            + ("Unit Ball" if A.isI else "Ellipsoid"))

        varName = glyphs.free_var_name(A.string + c.string)
        symbStr = glyphs.set(glyphs.sep(
            glyphs.clever_add(glyphs.clever_mul(A.string, varName), c.string),
            glyphs.le(glyphs.norm(varName), 1)))

        Set.__init__(self, typeStr, symbStr)

    # --------------------------------------------------------------------------
    # Properties.
    # --------------------------------------------------------------------------

    @property
    def dim(self):
        """The dimensionality :math:`n`."""
        return self._n

    @property
    def A(self):
        """The linear operator matrix :math:`A`."""
        return self._A

    @cached_property
    def Ainv(self):
        """The inverse linear operator matrix :math:`A^{-1}`."""
        try:
            inverse = cvxopt_inverse(self._A.value_as_matrix)
        except ValueError as error:
            raise RuntimeError("The matrix A is not invertible.") from error

        return Constant(glyphs.inverse(self._A.string), inverse, self._A.shape)

    @property
    def c(self):
        """The center point :math:`c`."""
        return self._c

    # --------------------------------------------------------------------------
    # Abstract method implementations.
    # --------------------------------------------------------------------------

    def _get_mutables(self):
        return frozenset()

    def _replace_mutables(self, mapping):
        return self

    Subtype = namedtuple("Subtype", ("dim",))

    def _get_subtype(self):
        return self.Subtype(self._n)

    # --------------------------------------------------------------------------
    # Algebraic operations.
    # --------------------------------------------------------------------------

    @convert_operands()
    def __add__(self, other):
        if not isinstance(other, AffineExpression):
            return NotImplemented

        return Ellipsoid(self._n, self._A, self._c + other)

    @convert_operands()
    def __radd__(self, other):
        if not isinstance(other, AffineExpression):
            return NotImplemented

        return Ellipsoid(self._n, self._A, other + self._c)

    @convert_operands()
    def __mul__(self, other):
        if not isinstance(other, AffineExpression):
            return NotImplemented

        if not other.scalar:
            raise TypeError(
                "Can only multiply an Ellipsoid from the right with a scalar.")

        return Ellipsoid(self._n, self._A*other, self._c*other)

    @convert_operands()
    def __rmul__(self, other):
        if not isinstance(other, AffineExpression):
            return NotImplemented

        if other.shape not in (self._A.shape, (1, 1)):
            raise TypeError("Can only multiply a {}-dimensional Ellipsoid from "
                "the left with a scalar or a {} matrix.".format(
                self._n, glyphs.shape(self._A.shape)))

        return Ellipsoid(self._n, other*self._A, other*self._c)

    @convert_operands()
    def __truediv__(self, other):
        if not isinstance(other, AffineExpression):
            return NotImplemented

        if not other.scalar:
            raise TypeError("You may only divide an Ellipsoid by a scalar.")

        if not other.constant:
            raise TypeError("You may only divide an Ellipsoid by a constant.")

        if other.is0:
            raise ZeroDivisionError("Tried to divide an Ellipsoid by zero.")

        return Ellipsoid(self._n, self._A/other, self._c/other)

    # --------------------------------------------------------------------------
    # Constraint-creating operations.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        if relation == operator.__rshift__:
            if issubclass(other.clstype, AffineExpression):
                if subtype.dim == other.subtype.dim:
                    return SOCConstraint.make_type(subtype.dim)

        return NotImplemented

    def _rshift_implementation(self, element):
        if isinstance(element, AffineExpression):
            if len(element) != self._n:
                raise TypeError("Cannot constrain the {}-dimensional "
                    "expression {} into a {}-dimensional ellipsoid."
                    .format(len(element), element.string, self._n))

            one = AffineExpression.from_constant(1)
            return SOCConstraint(self.Ainv*(element.vec - self.c), one)
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
