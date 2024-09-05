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

"""Implements :class:`SumExponentials`."""

import math
import operator
from collections import namedtuple

import cvxopt
import numpy

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property, cached_unary_operator
from ..constraints import LogSumExpConstraint, SumExponentialsConstraint
from .data import convert_and_refine_arguments, convert_operands, cvx2np
from .exp_affine import AffineExpression
from .expression import Expression, refine_operands, validate_prediction

_API_START = api_start(globals())
# -------------------------------


class SumExponentials(Expression):
    r"""Sum of elementwise exponentials of an affine expression.

    :Definition:

    Let :math:`x` be an :math:`n`-dimensional real affine expression.

    1.  If no additional expression :math:`y` is given, this is the sum of
        elementwise exponentials

        .. math::

            \sum_{i = 1}^n \exp(\operatorname{vec}(x)_i).

    2.  If an additional affine expression :math:`y` of same shape as :math:`x`
        is given, this is the sum of elementwise perspectives of exponentials

        .. math::

            \sum_{i = 1}^n \operatorname{vec}(y)_i \exp\left(
            \frac{\operatorname{vec}(x)_i}{\operatorname{vec}(y)_i}\right).

    .. warning::

        When you pose an upper bound :math:`t` on a sum of elementwise
        exponentials, then PICOS enforces :math:`t \geq 0` through an auxiliary
        constraint during solution search. When an additional expression
        :math:`y` is given, PICOS enforces :math:`y \geq 0` as well.
    """

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    @convert_and_refine_arguments("x", "y", allowNone=True)
    def __init__(self, x, y=None):
        """Construct a :class:`SumExponentials`.

        :param x: The affine expression :math:`x`.
        :type x: ~picos.expressions.AffineExpression
        :param y: An additional affine expression :math:`y`. If necessary, PICOS
            will attempt to reshape or broadcast it to the shape of :math:`x`.
        :type y: ~picos.expressions.AffineExpression
        """
        if not isinstance(x, AffineExpression):
            raise TypeError("Can only sum the elementwise exponentials of a "
                "real affine expression, not of {}.".format(x.string))

        if y is not None:
            if not isinstance(y, AffineExpression):
                raise TypeError("The additional parameter y must be a real "
                    "affine expression, not {}.".format(y.string))
            elif x.shape != y.shape:
                y = y.reshaped_or_broadcasted(x.shape)

            if y.is1:
                y = None

        self._x = x
        self._y = y

        if len(x) == 1:
            if y is None:
                typeStr = "Exponential"
                symbStr = glyphs.exp(x.string)
            else:
                typeStr = "Exponential Perspective"
                symbStr = glyphs.mul(
                    y.string, glyphs.exp(glyphs.div(x.string, y.string)))
        else:
            if y is None:
                typeStr = "Sum of Exponentials"
                symbStr = glyphs.make_function("sum", "exp")(x.string)
            else:
                typeStr = "Sum of Exponential Perspectives"
                symbStr = glyphs.sum(glyphs.mul(glyphs.slice(y.string, "i"),
                    glyphs.exp(glyphs.div(glyphs.slice(x.string, "i"),
                    glyphs.slice(y.string, "i")))))

        Expression.__init__(self, typeStr, symbStr)

    # --------------------------------------------------------------------------
    # Abstract method implementations and method overridings, except _predict.
    # --------------------------------------------------------------------------

    def _get_refined(self):
        if self._x.constant and (self._y is None or self._y.constant):
            return AffineExpression.from_constant(self.value, 1, self._symbStr)
        else:
            return self

    Subtype = namedtuple("Subtype", ("argdim", "y"))

    def _get_subtype(self):
        return self.Subtype(len(self._x), self._y is not None)

    def _get_value(self):
        x = numpy.ravel(cvx2np(self._x._get_value()))

        if self._y is None:
            s = numpy.sum(numpy.exp(x))
        else:
            y = numpy.ravel(cvx2np(self._y._get_value()))
            s = y.dot(numpy.exp(x / y))

        return cvxopt.matrix(s)

    @cached_unary_operator
    def _get_mutables(self):
        if self._y is None:
            return self._x._get_mutables()
        else:
            return self._x._get_mutables().union(self._y.mutables)

    def _is_convex(self):
        return True

    def _is_concave(self):
        return False

    def _replace_mutables(self, mapping):
        return self.__class__(self._x._replace_mutables(mapping),
            None if self._y is None else self._y._replace_mutables(mapping))

    def _freeze_mutables(self, freeze):
        return self.__class__(self._x._freeze_mutables(freeze),
            None if self._y is None else self._y._freeze_mutables(freeze))

    # --------------------------------------------------------------------------
    # Python special method implementations, except constraint-creating ones.
    # --------------------------------------------------------------------------

    @classmethod
    def _add(cls, self, other, forward):
        if isinstance(other, AffineExpression) and other.constant:
            value = other.value

            if not value:
                return self
            elif value > 0:
                if self._y is None:
                    result = cls(self._x // math.log(value))
                else:
                    result = cls(self._x // value, self._y // 1)

                if forward:
                    string = glyphs.clever_add(self.string, other.string)
                else:
                    string = glyphs.clever_add(other.string, self.string)

                result._typeStr = "Offset " + result._typeStr
                result._symbStr = string

                return result
        elif isinstance(other, cls):
            assert forward, "Encountered __radd__ on equal types."

            if self._y is None and other._y is None:
                result = cls(self._x.vec // other._x.vec)
            elif self._y is not None and other._y is None:
                one = AffineExpression.from_constant(1.0, (other.n, 1))
                result = cls(self._x.vec // other._x.vec, self._y.vec // one)
            elif self._y is None and other._y is not None:
                one = AffineExpression.from_constant(1.0, (self.n, 1))
                result = cls(self._x.vec // other._x.vec, one // other._y.vec)
            else:
                result = cls(
                    self._x.vec // other._x.vec, self._y.vec // other._y.vec)

            result._symbStr = glyphs.clever_add(self.string, other.string)

            return result

        if forward:
            return Expression.__add__(self, other)
        else:
            return Expression.__radd__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __add__(self, other):
        return SumExponentials._add(self, other, True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __radd__(self, other):
        return SumExponentials._add(self, other, False)

    @classmethod
    def _mul_div(cls, self, other, div, forward):
        assert not div or forward

        if isinstance(other, AffineExpression) and other.constant:
            factor = other.safe_value

            if not factor:
                if div:
                    raise ZeroDivisionError(
                        "Cannot divide {} by zero.".format(self.string))
                else:
                    return AffineExpression.zero()
            elif factor == 1:
                return self
            elif factor > 0:
                if div:
                    factor = 1 / factor
                    string = glyphs.div(self.string, other.string)
                elif forward:
                    string = glyphs.clever_mul(self.string, other.string)
                else:
                    string = glyphs.clever_mul(other.string, self.string)

                if self._y is None:
                    result = cls(self._x + math.log(factor))
                else:
                    result = cls(other*self._x, other*self._y)

                result._typeStr = "Scaled " + result._typeStr
                result._symbStr = string

                return result

        if div:
            return Expression.__div__(self, other)
        elif forward:
            return Expression.__mul__(self, other)
        else:
            return Expression.__rmul__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __mul__(self, other):
        """Denote scaling from the right hand side."""
        return SumExponentials._mul_div(self, other, div=False, forward=True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __rmul__(self, other):
        """Denote scaling from the left hand side."""
        return SumExponentials._mul_div(self, other, div=False, forward=False)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __truediv__(self, other):
        """Denote division by a constant scalar."""
        return SumExponentials._mul_div(self, other, div=True, forward=True)

    # --------------------------------------------------------------------------
    # Methods and properties that return expressions.
    # --------------------------------------------------------------------------

    @property
    def x(self):
        """The expression :math:`x`."""
        return self._x

    @property
    def y(self):
        """The additional expression :math:`y`, or :obj:`None`."""
        return self._y

    @cached_property
    def log(self):
        """The logarithm of the expression."""
        from . import LogSumExp

        if self._y is not None:
            raise NotImplementedError("May only take the logarithm of a sum of"
                " exponentials, not of a sum of exponential perspectives.")

        return LogSumExp(self._x)

    # --------------------------------------------------------------------------
    # Methods and properties that describe the expression.
    # --------------------------------------------------------------------------

    @property
    def n(self):
        """Length of :attr:`x`."""
        return len(self._x)

    # --------------------------------------------------------------------------
    # Constraint-creating operators, and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        if relation == operator.__le__:
            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                return SumExponentialsConstraint.make_type(
                    argdim=subtype.argdim,
                    lse_representable=(not subtype.y and other.subtype.nonneg))
            elif issubclass(other.clstype, SumExponentials):
                if subtype.y or other.subtype.y:
                    return NotImplemented

                if other.subtype.argdim != 1:
                    return NotImplemented

                return LogSumExpConstraint.make_type(argdim=subtype.argdim)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        from . import LogSumExp

        if isinstance(other, AffineExpression):
            return SumExponentialsConstraint(self, other)
        elif isinstance(other, SumExponentials):
            if self._y is not None or other._y is not None:
                raise NotImplementedError("Comparing two sums of exponentials "
                    "is not supported if either expression has the additional "
                    "perspectives parameter y set.")

            if other.n != 1:
                raise NotImplementedError("You may only upper bound a sum of "
                    "exponentials by a single exponential, not by another sum.")

            return LogSumExp(self._x) <= other._x
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
