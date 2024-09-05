# ------------------------------------------------------------------------------
# Copyright (C) 2019 Maximilian Stahlberg
# Based on the original picos.expressions module by Guillaume Sagnol.
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

"""Implements :class:`Norm`."""

import operator
from collections import namedtuple

import cvxopt
import numpy

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_unary_operator
from ..constraints import (AbsoluteValueConstraint, Constraint,
                           MatrixNormConstraint, SOCConstraint,
                           VectorNormConstraint)
from ..legacy import throw_deprecation_warning
from .data import (convert_and_refine_arguments, convert_operands, cvx2np,
                   make_fraction)
from .exp_affine import AffineExpression, ComplexAffineExpression
from .exp_sqnorm import SquaredNorm
from .expression import Expression, refine_operands, validate_prediction

_API_START = api_start(globals())
# -------------------------------


class Norm(Expression):
    r"""Entrywise :math:`p`-norm or :math:`L_{p,q}`-norm of an expression.

    This class can represent the absolute value, the modulus, a vector
    :math:`p`-norm and an entrywise matrix :math:`L_{p,q}`-norm of a (complex)
    affine expression. In addition to these convex norms, it can represent the
    concave *generalized vector* :math:`p`-*norm* with :math:`0 < p \leq 1`.

    Not all of these norms are available on the complex field; see the
    definitions below to learn more.

    :Definition:

    If :math:`q` is not given (:obj:`None`), then it is set equal to :math:`p`.

    1.  If the normed expression is a real scalar :math:`x`, then this is the
        absolute value

        .. math::

            |x| = \begin{cases}
               x, &\text{if }x \geq 0,\\
               -x, &\text{otherwise.}
            \end{cases}

        The parameters :math:`p` and :math:`q` are ignored in this case.

    2.  If the normed expression is a complex scalar :math:`z`, then this is the
        modulus

        .. math::

            |z| = \sqrt{\operatorname{Re}(z)^2 + \operatorname{Im}(z)^2}.

        The parameters :math:`p` and :math:`q` are ignored in this case.

    3.  If the normed expression is a real vector :math:`x` and :math:`p = q`,
        then this is the (generalized) vector :math:`p`-norm

        .. math::

            \lVert x \rVert_p = \left(\sum_{i=1}^n |x_i|^p\right)^{\frac{1}{p}}

        for :math:`p \in \mathbb{Q} \cup \{\infty\}` with :math:`p > 0` and
        :math:`|x_i|` the absolute value of the :math:`i`-th entry of :math:`x`.

        Note that for :math:`p < 1` the expression is not convex and thus not a
        proper norm. However, it is concave over the nonnegative orthant and
        posing a lower bound on such a generalized norm yields a convex
        constraint for :math:`x \geq 0`.

        .. warning::

            When you pose a lower bound on a concave generalized norm
            (:math:`p < 1`), then PICOS enforces :math:`x \geq 0` through an
            auxiliary constraint during solution search.

        Special cases:

        -   For :math:`p = 1`, this is the *Manhattan* or *Taxicab* norm
            :math:`\lVert x \rVert_{\text{sum}}`.
        -   For :math:`p = 2`, this is the Euclidean norm
            :math:`\lVert x \rVert = \lVert x \rVert_2`.
        -   For :math:`p = \infty`, this is the *Maximum*, *Chebyshev*, or
            *Infinity* norm :math:`\lVert x \rVert_{\text{max}}`.

    4.  If the normed expression is a real vector :math:`x` and
        :math:`p \neq q`, then it is treated as a matrix with a single row or a
        single column, depending on the shape associated with :math:`x`. See
        case (5).

    5.  If the normed expression is a complex vector :math:`z` and
        :math:`p = q`, then the definition is the same as in case (3) but with
        :math:`x = z`, :math:`|x_i|` the modulus of the :math:`i`-th entry of
        :math:`x`, and :math:`p \geq 1`.

    6.  If the normed expression is a real :math:`m \times n` matrix :math:`X`,
        then this is the :math:`L_{p,q}`-norm

        .. math::

            \lVert X \rVert_{p,q} =
            \left(\sum_{j = 1}^n \left(\sum_{i = 1}^m
                |X_{ij}|^p
            \right)^{\frac{q}{p}} \right)^{\frac{1}{q}}

        for :math:`p, q \in \mathbb{Q} \cup \{\infty\}` with :math:`p,q \geq 1`.

        If :math:`p = q`, then this is equal to the (generalized) vector
        :math:`p`-norm of the the vectorized matrix, that is
        :math:`\lVert X \rVert_{p,p} = \lVert \operatorname{vec}(X) \rVert_p`.
        In this case, the requirement :math:`p \geq 1` is relaxed to
        :math:`p > 0` and :math:`X` may be a complex matrix. See case (3).

        Special cases:

        -   For :math:`p = q = 2`, this is the Frobenius norm
            :math:`\lVert X \rVert = \lVert X \rVert_F`.
        -   For :math:`p = 1`, :math:`q = \infty`, this is the maximum absolute
            column sum

            .. math::

                \lVert X \rVert_{1,\infty} =
                \max_{j=1 \ldots n} \sum_{i=1}^m |X_{ij}|.

            This equals the operator norm induced by the vector :math:`1`-norm.
            You can obtain the maximum absolute row sum (the operator norm
            induced by the vector :math:`\infty`-norm) by first transposing
            :math:`X`.

    7.  Complex matrix norms are not supported.

    .. note::

        You can write :math:`\infty` in Python as ``float("inf")``.
    """

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    @convert_and_refine_arguments("x")
    def __init__(self, x, p=2, q=None, denominator_limit=1000):
        """Construct a :class:`Norm`.

        :param x: The affine expression to take the norm of.
        :type x: ~picos.expressions.ComplexAffineExpression
        :param float p: The value for :math:`p`, which is cast to a limited
            precision fraction.
        :param float q: The value for :math:`q`, which is cast to a limited
            precision fraction. The default of :obj:`None` means *equal to*
            :math:`p`.
        :param int denominator_limit: The largest allowed denominator when
            casting :math:`p` and :math:`q` to a fraction. Higher values can
            yield a greater precision at reduced performance.
        """
        # Validate x.
        if not isinstance(x, ComplexAffineExpression):
            raise TypeError("Can only form the norm of an affine expression, "
                "not of {}.".format(type(x).__name__))

        complex = not isinstance(x, AffineExpression)

        if isinstance(p, tuple) and len(p) == 2:
            throw_deprecation_warning("Arguments 'p' and 'q' to Norm must be "
                "given separately in the future.", decoratorLevel=1)
            p, q = p

        if q is None:
            q = p

        # Load p as a limtied precision fraction.
        if p == float("inf"):
            pNum = p
            pDen = 1
            pStr = glyphs.infty()
        else:
            pNum, pDen, p, pStr = make_fraction(p, denominator_limit)

        # Load q as a limtied precision fraction.
        if q == float("inf"):
            qNum = q
            qDen = 1
            qStr = glyphs.infty()
        else:
            qNum, qDen, q, qStr = make_fraction(q, denominator_limit)

        # Validate that p and q are in the allowed range.
        if p == q:
            if complex and p < 1:
                raise NotImplementedError(
                    "Complex p-norm requires {}.".format(glyphs.ge("p", "1")))
            elif p <= 0:
                raise ValueError(
                    "p-norm requires {}.".format(glyphs.gt("p", "0")))
        elif p != q:
            if complex:
                raise NotImplementedError(
                    "(p,q)-norm is not supported for complex expressions.")
            elif p < 1 or q < 1:
                raise ValueError("(p,q)-norm requires {} and {}."
                    .format(glyphs.ge("p", "1"), glyphs.ge("q", "1")))

        # Build the string representations.
        if len(x) == 1:
            typeStr = "Modulus" if complex else "Absolute Value"
            symbStr = glyphs.abs(x.string)
        elif p == q:
            vec = 1 in x.shape
            if p == 1:
                typeStr = "Manhattan Norm"
                symbStr = glyphs.pnorm(x.string, "sum")
            elif p == 2:
                typeStr = "Euclidean Norm" if vec else "Frobenius Norm"
                symbStr = glyphs.norm(x.string)
            elif p == float("inf"):
                typeStr = "Maximum Norm"
                symbStr = glyphs.pnorm(x.string, "max")
            else:
                if p < 1:
                    typeStr = "Generalized p-Norm" if vec else \
                        "Entrywise Generalized p-Norm"
                else:
                    typeStr = "Vector p-Norm" if vec else "Entrywise p-Norm"
                symbStr = glyphs.pnorm(x.string, pStr)
        else:
            if p == 1 and q == float("inf"):
                typeStr = "Maximum Absolute Column Sum"
            else:
                typeStr = "Matrix (p,q)-Norm"
            symbStr = glyphs.pqnorm(x.string, pStr, qStr)

        if complex:
            typeStr = "Complex " + typeStr

        # Reduce the complex to the real case.
        if complex:
            if len(x) == 1:
                # From modulus to real Euclidean norm.
                x = x.real // x.imag

                p, pNum, pDen = 2.0, 2, 1
                q, qNum, qDen = 2.0, 2, 1
            else:
                # From complex vector/entrywise p-norm to real matrix p,q-norm.
                if x.shape[0] == 1:
                    x = x.real // x.imag
                elif x.shape[1] == 1:
                    x = x.T.real // x.T.imag
                else:
                    x = x.vec.T.real // x.vec.T.imag

                assert p == q
                p, pNum, pDen = 2.0, 2, 1

        # Reduce an entrywise p-norm to a vector p-norm and make all vector
        # norms refer to column vectors.
        if p == q and x.shape[1] != 1:
            x = x.vec

        assert isinstance(x, AffineExpression)
        assert float(pNum) / float(pDen) == p and float(qNum) / float(qDen) == q

        self._x     = x
        self._pNum  = pNum
        self._pDen  = pDen
        self._qNum  = qNum
        self._qDen  = qDen
        self._limit = denominator_limit

        Expression.__init__(self, typeStr, symbStr)

    # --------------------------------------------------------------------------
    # Abstract method implementations and method overridings, except _predict.
    # --------------------------------------------------------------------------

    @cached_unary_operator
    def _get_refined(self):
        if self._x.constant:
            return AffineExpression.from_constant(self.value, 1, self.string)
        else:
            return self

    Subtype = namedtuple("Subtype", ("xShape", "pNum", "pDen", "qNum", "qDen"))

    def _get_subtype(self):
        # NOTE: The xShape field refers to the internal (real and potentially
        #       vectorized) representation of the normed expression.
        return self.Subtype(
            self._x.shape, self._pNum, self._pDen, self._qNum, self._qDen)

    def _get_value(self):
        value = self._x._get_value()

        if value.size == (1, 1):
            return abs(value)

        value = cvx2np(value)

        p, q = self.p, self.q

        if p == q:
            value = numpy.linalg.norm(numpy.ravel(value), p)
        else:
            columns = value.shape[1]
            value = numpy.linalg.norm([
                numpy.linalg.norm(numpy.ravel(value[:, j]), p)
                for j in range(columns)], q)

        return cvxopt.matrix(value)

    def _get_mutables(self):
        return self._x._get_mutables()

    def _is_convex(self):
        return self.p >= 1 or len(self._x) == 1

    def _is_concave(self):
        return self.p <= 1 and len(self._x) != 1

    def _replace_mutables(self, mapping):
        return self.__class__(
            self._x._replace_mutables(mapping), self.p, self.q, self._limit)

    def _freeze_mutables(self, freeze):
        return self.__class__(
            self._x._freeze_mutables(freeze), self.p, self.q, self._limit)

    # --------------------------------------------------------------------------
    # Python special method implementations, except constraint-creating ones.
    # --------------------------------------------------------------------------

    @classmethod
    def _mul(cls, self, other, forward):
        if isinstance(other, AffineExpression) and other.constant:
            factor = other.safe_value

            if not factor:
                return AffineExpression.zero()
            elif factor == 1:
                return self
            elif factor > 0:
                if forward:
                    string = glyphs.clever_mul(self.string, other.string)
                else:
                    string = glyphs.clever_mul(other.string, self.string)

                norm = cls(other*self._x, self.p, self.q, self._limit)
                norm._typeStr = "Scaled " + norm._typeStr
                norm._symbStr = string

                return norm

        if forward:
            return Expression.__mul__(self, other)
        else:
            return Expression.__rmul__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __mul__(self, other):
        return Norm._mul(self, other, True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __rmul__(self, other):
        return Norm._mul(self, other, False)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __pow__(self, other):
        if isinstance(other, AffineExpression):
            if not other.constant or other.value != 2:
                raise NotImplementedError(
                    "You may only take a norm to the power of two.")

            p, q = self.p, self.q

            if p == q and p == 2:
                result = SquaredNorm(self._x)
                result._symbStr = glyphs.squared(self.string)
                return result
            else:
                raise NotImplementedError(
                    "You may only square an Euclidean or Frobenius norm.")

        return Expression.__pow__(self, other)

    # --------------------------------------------------------------------------
    # Properties and functions that describe the expression.
    # --------------------------------------------------------------------------

    @property
    def p(self):
        """The parameter :math:`p`.

        This is a limited precision version of the parameter used when the norm
        was constructed.
        """
        return float(self._pNum) / float(self._pDen)

    @property
    def pnum(self):
        """The limited precision fraction numerator of :math:`p`."""
        return self._pNum

    @property
    def pden(self):
        """The limited precision fraction denominator of :math:`p`."""
        return self._pDen

    @property
    def q(self):
        """The parameter :math:`q`.

        This is a limited precision version of the parameter used when the norm
        was constructed.
        """
        return float(self._qNum) / float(self._qDen)

    @property
    def qnum(self):
        """The limited precision fraction numerator of :math:`q`."""
        return self._qNum

    @property
    def qden(self):
        """The limited precision fraction denominator of :math:`q`."""
        return self._qDen

    # --------------------------------------------------------------------------
    # Methods and properties that return modified copies.
    # --------------------------------------------------------------------------

    @property
    def x(self):
        """Real expression whose norm equals that of the original expression."""
        return self._x

    # --------------------------------------------------------------------------
    # Constraint-creating operators, and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        xShape, pNum, pDen, qNum, qDen = subtype
        xLen = xShape[0] * xShape[1]
        p = float(pNum) / float(pDen)
        q = float(qNum) / float(qDen)

        if relation == operator.__le__:
            if not (xLen == 1 or p >= 1):
                return NotImplemented  # Not convex.

            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                if xLen == 1:
                    return AbsoluteValueConstraint.make_type()
                elif p == q == 2:
                    return SOCConstraint.make_type(argdim=xLen)
                elif p == q:
                    return VectorNormConstraint.make_type(
                        xLen, pNum, pDen, Constraint.LE)
                else:
                    return MatrixNormConstraint.make_type(
                        xShape, pNum, pDen, qNum, qDen)
        elif relation == operator.__ge__:
            if not (xLen != 1 and p <= 1):
                return NotImplemented  # Not concave.

            assert p == q

            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                return VectorNormConstraint.make_type(
                    xLen, pNum, pDen, Constraint.GE)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        if not self.convex:
            raise TypeError("Cannot upper-bound a nonconvex generalized norm.")

        if isinstance(other, AffineExpression):
            if len(self._x) == 1:
                return AbsoluteValueConstraint(self._x, other)
            elif self.p == self.q == 2:
                # NOTE: The custom string is necessary in case the original x
                #       was complex; otherwise the string would be something
                #       like "‖vec([Re(xᵀ); Im(xᵀ)])‖ ≤ b".
                return SOCConstraint(self._x, other,
                    customString=glyphs.le(self.string, other.string))
            elif self.p == self.q:
                return VectorNormConstraint(self, Constraint.LE, other)
            else:
                return MatrixNormConstraint(self, other)
        else:
            return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __ge__(self, other):
        if not self.concave:
            raise TypeError("Cannot lower-bound a nonconcave norm.")

        assert self.p == self.q

        if isinstance(other, AffineExpression):
            return VectorNormConstraint(self, Constraint.GE, other)
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
