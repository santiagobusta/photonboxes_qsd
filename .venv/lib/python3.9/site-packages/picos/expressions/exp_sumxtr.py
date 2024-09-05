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

"""Implements :class:`SumExtremes`."""

import operator
from collections import namedtuple

import cvxopt
import numpy

from .. import glyphs
from ..apidoc import api_end, api_start
from ..constraints import Constraint, SumExtremesConstraint
from .data import convert_and_refine_arguments, convert_operands, cvx2np
from .exp_affine import AffineExpression, ComplexAffineExpression
from .expression import Expression, refine_operands, validate_prediction

_API_START = api_start(globals())
# -------------------------------


class SumExtremes(Expression):
    r"""Sum of the :math:`k` largest or smallest elements or eigenvalues.

    :Definition:

    Let :math:`k \in \mathbb{Z}_{\geq 1}`.

    1.  If :math:`x` is an :math:`n`-dimensional real vector or matrix and
        ``eigenvalues == False``, then this is the sum of the :math:`k \leq n`
        largest or smallest scalar elements of :math:`x`, depending on the truth
        value of ``largest``.

        Special cases:

        -   If :math:`k = 1`, this is either the largest element
            :math:`\max_{i = 1}^n \operatorname{vec}(x)_i` or the smallest
            element :math:`\min_{i = 1}^n \operatorname{vec}(x)_i` of :math:`x`.
        -   If :math:`k = n`, this is the sum of all elements
            :math:`\langle x, 1 \rangle` of :math:`x`.

    2.  If :math:`X` is an :math:`n \times n` hermitian matrix and
        ``eigenvalues == True``, then this is the sum of the :math:`k \leq n`
        largest or smallest eigenvalues of :math:`X`, depending on the truth
        value of ``largest``. Recall that the eigenvalues of a hermitian matrix
        are real.

        Special cases:

        -   If :math:`k = 1`, this is either the largest eigenvalue
            :math:`\lambda_{\max}(X)` or the smallest eigenvalue
            :math:`\lambda_{\min}(X)` of :math:`X`.
        -   If :math:`k = n`, this equals the trace
            :math:`\operatorname{tr}(X)`.

    If the given :math:`k` exceeds the :math:`n` of either case, then :math:`k`
    is silently clipped to :math:`n`.
    """

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    @convert_and_refine_arguments("x")
    def __init__(self, x, k, largest, eigenvalues=False):
        """Construct a :class:`SumExtremes`.

        :param x: The affine expression to take a sum over.
        :type x: ~picos.expressions.ComplexAffineExpression
        :param int k: Number of summands.
        :param bool largest: Whether to sum over the largest (eigen)values as
            opposed to the smallest.
        :param bool eigenvalues: Whether to sum eigenvalues instead of elements.
        """
        largest     = bool(largest)
        eigenvalues = bool(eigenvalues)

        lStr = "largest" if largest else "smallest"
        eStr = "eigenvalues" if eigenvalues else "scalar elements"
        what = "{} {}".format(lStr, eStr)

        # Validate x.
        if not isinstance(x, ComplexAffineExpression):
            raise TypeError("Can only sum {} of an affine expression, not of "
                "{}.".format(what, type(x).__name__))

        # Further validate x.
        if eigenvalues:
            if not x.square:
                raise TypeError("Cannot sum {} of {} as its shape of {} is not "
                    "square.".format(what, x.string, glyphs.shape(x.shape)))
            elif not x.hermitian:
                raise NotImplementedError(
                    "Summing the {0} of {1} is not supported as {1} is not "
                    "necessarily hermitian.".format(what, x.string))
        else:
            if not isinstance(x, AffineExpression):
                raise TypeError("Can only sum {} of a real-valued expression "
                    "but {} is properly complex.".format(what, x.string))

        # Validate k.
        if int(k) != k:
            raise ValueError(
                "Conversion of k = {} to an integer is ambiguous.".format(k))
        k = int(k)
        if k < 1:
            raise ValueError(
                "Number of {} to sum must be positive.".format(what))

        # Clip k to be at most n.
        k = min(k, x.shape[0]) if eigenvalues else min(k, len(x))

        # Find out if all (eigen)values are summed.
        full = k == x.shape[0] if eigenvalues else k == len(x)
        assert len(x) != 1 or full

        self._x           = x
        self._k           = k
        self._largest     = largest
        self._eigenvalues = eigenvalues
        self._full        = full

        s, lbd = x.string, glyphs.lambda_()
        if full:
            if eigenvalues:
                typeStr = "Sum of Eigenvalues"
                symbStr = symbStr = glyphs.trace(s)
            else:
                typeStr = "Sum of Elements"
                symbStr = glyphs.sum(s)
        elif k > 1:
            if eigenvalues and largest:
                typeStr = "Sum of Largest Eigenvalues"
                symbStr = glyphs.make_function(
                    "sum_{}_largest_{}".format(k, lbd))(s)
            elif eigenvalues and not largest:
                typeStr = "Sum of Smallest Eigenvalues"
                symbStr = glyphs.make_function(
                    "sum_{}_smallest_{}".format(k, lbd))(s)
            elif not eigenvalues and largest:
                typeStr = "Sum of Largest Elements"
                symbStr = glyphs.make_function("sum_{}_largest".format(k))(s)
            else:
                typeStr = "Sum of Smallest Elements"
                symbStr = glyphs.make_function("sum_{}_smallest".format(k))(s)
        else:
            if eigenvalues and largest:
                typeStr = "Largest Eigenvalue"
                symbStr = glyphs.make_function("{}_max".format(lbd))(s)
            elif eigenvalues and not largest:
                typeStr = "Smallest Eigenvalue"
                symbStr = glyphs.make_function("{}_min".format(lbd))(s)
            elif not eigenvalues and largest:
                typeStr = "Largest Element"
                symbStr = glyphs.max(s)
            else:
                typeStr = "Smallest Element"
                symbStr = glyphs.min(s)

        Expression.__init__(self, typeStr, symbStr)

    # --------------------------------------------------------------------------
    # Abstract method implementations and method overridings, except _predict.
    # --------------------------------------------------------------------------

    def _get_refined(self):
        if self._x.constant:
            return AffineExpression.from_constant(self.value, 1, self._symbStr)
        elif self._full:
            if len(self._x) == 1:
                return self._x  # Don't carry the string for an identity.
            if self._eigenvalues:
                return self._x.tr  # Symbolic strings already match.
            else:
                return (1 | self._x).renamed(self._symbStr)
        else:
            return self

    Subtype = namedtuple("Subtype",
        ("argdim", "k", "largest", "eigenvalues", "complex"))

    def _get_subtype(self):
        return self.Subtype(len(self._x), self._k, self._largest,
            self._eigenvalues, self._x.complex)

    def _get_value(self):
        value = self._x._get_value()

        if self._eigenvalues:
            value = sorted(numpy.linalg.eigvalsh(cvx2np(value)))
        else:
            value = sorted(value)

        value = sum(value[-self._k:] if self._largest else value[:self._k])
        value = cvxopt.matrix(value)

        return value

    def _get_mutables(self):
        return self._x._get_mutables()

    def _is_convex(self):
        return self._largest or self._full

    def _is_concave(self):
        return not self._largest or self._full

    def _replace_mutables(self, mapping):
        return self.__class__(self._x._replace_mutables(mapping),
            self._k, self._largest, self._eigenvalues)

    def _freeze_mutables(self, freeze):
        return self.__class__(self._x._freeze_mutables(freeze),
            self._k, self._largest, self._eigenvalues)

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

                product = cls(
                    factor*self._x, self._k, self._largest, self._eigenvalues)
                product._typeStr = "Scaled " + product._typeStr
                product._symbStr = string

                return product

        if forward:
            return Expression.__mul__(self, other)
        else:
            return Expression.__rmul__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __mul__(self, other):
        return SumExtremes._mul(self, other, True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __rmul__(self, other):
        return SumExtremes._mul(self, other, False)

    # --------------------------------------------------------------------------
    # Methods and properties that return expressions.
    # --------------------------------------------------------------------------

    @property
    def x(self):
        """The expression under the sum."""
        return self._x

    # --------------------------------------------------------------------------
    # Methods and properties that describe the expression.
    # --------------------------------------------------------------------------

    @property
    def k(self):
        """Number of (eigen)values to sum."""
        return self._k

    @property
    def largest(self):
        """Whether the sum concerns largest values as opposed to smallest."""
        return self._largest

    @property
    def eigenvalues(self):
        """Whether the sum concerns eigenvalues as opposed to elements."""
        return self._eigenvalues

    @property
    def full(self):
        """Whether the sum concerns *all* (eigen)values of the expression."""
        return self._full

    # --------------------------------------------------------------------------
    # Constraint-creating operators, and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        n = subtype.argdim
        k = subtype.k
        e = subtype.eigenvalues
        c = subtype.complex

        kmax = int(n**0.5) if e else n
        full = k == kmax

        convex  = subtype.largest or full
        concave = not subtype.largest or full

        if relation == operator.__le__:
            if not convex:
                return NotImplemented

            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                return SumExtremesConstraint.make_type(n, k, e, c)
        elif relation == operator.__ge__:
            if not concave:
                return NotImplemented

            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                return SumExtremesConstraint.make_type(n, k, e, c)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        if not self.convex:
            raise TypeError("Cannot upper-bound the nonconvex expression {}."
                .format(self._symbStr))

        if isinstance(other, AffineExpression):
            return SumExtremesConstraint(self, Constraint.LE, other)
        else:
            return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __ge__(self, other):
        if not self.concave:
            raise TypeError("Cannot upper-bound the nonconcave expression {}."
                .format(self._symbStr))

        if isinstance(other, AffineExpression):
            return SumExtremesConstraint(self, Constraint.GE, other)
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
