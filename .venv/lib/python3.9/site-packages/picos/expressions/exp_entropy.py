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

"""Implements :class:`Entropy` and :class:`NegativeEntropy`."""

# TODO: Common base class for Entropy and NegativeEntropy.

import math
import operator
from collections import namedtuple

import cvxopt
import numpy

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_selfinverse_unary_operator, cached_unary_operator
from ..constraints import KullbackLeiblerConstraint
from .data import convert_and_refine_arguments, convert_operands, cvx2np
from .exp_affine import AffineExpression
from .expression import Expression, refine_operands, validate_prediction

_API_START = api_start(globals())
# -------------------------------


class Entropy(Expression):
    r"""Entropy or negative relative entropy of an affine expression.

    Negative relative entropy is also known as the perspective of the logarithm.

    :Definition:

    Let :math:`x` be an :math:`n`-dimensional real affine expression.

    1.  If no additional expression :math:`y` is given, this is the entropy

        .. math::

            -\sum_{i = 1}^n \operatorname{vec}(x)_i
            \log(\operatorname{vec}(x)_i).

    2.  If an additional affine expression :math:`y` of same shape as :math:`x`
        is given, this is the negative relative entropy (or logarithmic
        perspective)

        .. math::

            -\sum_{i = 1}^n \operatorname{vec}(x)_i
            \log\left(
                \frac{\operatorname{vec}(x)_i}{\operatorname{vec}(y)_i}
            \right)
            &= -\sum_{i = 1}^n \operatorname{vec}(x)_i
            \left[
                \log(\operatorname{vec}(x)_i) - \log(\operatorname{vec}(y)_i)
            \right] \\
            &= \sum_{i = 1}^n \operatorname{vec}(x)_i
            \left[
                \log(\operatorname{vec}(y)_i) - \log(\operatorname{vec}(x)_i
            \right] \\
            &= \sum_{i = 1}^n \operatorname{vec}(x)_i
            \log\left(
                \frac{\operatorname{vec}(y)_i}{\operatorname{vec}(x)_i}
            \right).

    .. warning::

        When you pose a lower bound on this expression, then PICOS enforces
        :math:`x \geq 0` through an auxiliary constraint during solution search.
        When an additional expression :math:`y` is given, PICOS enforces
        :math:`y \geq 0` as well.
    """

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    @convert_and_refine_arguments("x", "y", allowNone=True)
    def __init__(self, x, y=None):
        """Construct an :class:`Entropy`.

        :param x: The affine expression :math:`x`.
        :type x: ~picos.expressions.AffineExpression
        :param y: An additional affine expression :math:`y`. If necessary, PICOS
            will attempt to reshape or broadcast it to the shape of :math:`x`.
        :type y: ~picos.expressions.AffineExpression
        """
        if not isinstance(x, AffineExpression):
            raise TypeError("Can only take the elementwise logarithm of a real "
                "affine expression, not of {}.".format(type(x).__name__))

        if y is not None:
            if not isinstance(y, AffineExpression):
                raise TypeError("The additional parameter y must be a real "
                    "affine expression, not {}.".format(type(x).__name__))
            elif x.shape != y.shape:
                y = y.reshaped_or_broadcasted(x.shape)

            if y.is1:
                y = None

        self._x = x
        self._y = y

        if y is None:
            typeStr = "Entropy"
            if len(x) == 1:
                symbStr = glyphs.neg(glyphs.mul(x.string, glyphs.log(x.string)))
            else:
                symbStr = glyphs.neg(
                    glyphs.dotp(x.string, glyphs.log(x.string)))
        else:
            typeStr = "Logarithmic Perspective"
            if len(x) == 1:
                symbStr = glyphs.mul(
                    x.string, glyphs.log(glyphs.div(y.string, x.string)))
            else:
                symbStr = glyphs.dotp(x.string,
                    glyphs.sub(glyphs.log(y.string), glyphs.log(x.string)))

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
            s = -numpy.dot(x, numpy.log(x))
        else:
            y = numpy.ravel(cvx2np(self._y._get_value()))
            s = numpy.dot(x, numpy.log(y / x))

        return cvxopt.matrix(s)

    @cached_unary_operator
    def _get_mutables(self):
        if self._y is None:
            return self._x._get_mutables()
        else:
            return self._x._get_mutables().union(self._y.mutables)

    def _is_convex(self):
        return False

    def _is_concave(self):
        return True

    def _replace_mutables(self, mapping):
        return self.__class__(self._x._replace_mutables(mapping),
            None if self._y is None else self._y._replace_mutables(mapping))

    def _freeze_mutables(self, freeze):
        return self.__class__(self._x._freeze_mutables(freeze),
            None if self._y is None else self._y._freeze_mutables(freeze))

    # --------------------------------------------------------------------------
    # Python special method implementations, except constraint-creating ones.
    # --------------------------------------------------------------------------

    @cached_selfinverse_unary_operator
    def __neg__(self):
        return NegativeEntropy(self._x, self._y)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __add__(self, other):
        other_str = other.string

        if isinstance(other, AffineExpression):
            if other.is0:
                return self

            other = Entropy(other, other * math.e)

        if isinstance(other, Entropy):
            if self._y is None and other._y is None:
                entropy = Entropy(self._x.vec // other._x.vec)
            elif self._y is not None and other._y is None:
                one = AffineExpression.from_constant(1.0, (other.n, 1))
                entropy = Entropy(
                    self._x.vec // other._x.vec, self._y.vec // one)
            elif self._y is None and other._y is not None:
                one = AffineExpression.from_constant(1.0, (self.n, 1))
                entropy = Entropy(
                    self._x.vec // other._x.vec, one // other._y.vec)
            else:
                entropy = Entropy(
                    self._x.vec // other._x.vec, self._y.vec // other._y.vec)

            entropy._symbStr = glyphs.clever_add(self.string, other_str)

            return entropy

        return Expression.__add__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __radd__(self, other):
        if isinstance(other, (AffineExpression, Entropy)):
            entropy = Entropy.__add__(self, other)

            if entropy is not self:
                entropy._symbStr = glyphs.clever_add(other.string, self.string)

            return entropy

        return Expression.__radd__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __sub__(self, other):
        if isinstance(other, (AffineExpression, NegativeEntropy)):
            return self + (-other)

        return Expression.__sub__(self, other)

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

        if relation == operator.__ge__:
            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                return KullbackLeiblerConstraint.make_type(
                    argdim=subtype.argdim)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __ge__(self, other):
        if isinstance(other, AffineExpression):
            return KullbackLeiblerConstraint(-self, -other)
        else:
            return NotImplemented


class NegativeEntropy(Expression):
    r"""Negative or relative entropy of an affine expression.

    Relative entropy is also known as the Kullback-Leibler divergence.

    :Definition:

    Let :math:`x` be an :math:`n`-dimensional real affine expression.

    1.  If no additional expression :math:`y` is given, this is the negative
        entropy

        .. math::

            \sum_{i = 1}^n \operatorname{vec}(x)_i
            \log(\operatorname{vec}(x)_i).

    2.  If an additional affine expression :math:`y` of same shape as :math:`x`
        is given, this is the relative entropy (or Kullback-Leibler divergence)

        .. math::

            \sum_{i = 1}^n \operatorname{vec}(x)_i
            \log\left(
                \frac{\operatorname{vec}(x)_i}{\operatorname{vec}(y)_i}
            \right)
            = \sum_{i = 1}^n \operatorname{vec}(x)_i
            \left[
                \log(\operatorname{vec}(x)_i) - \log(\operatorname{vec}(y)_i)
            \right].

    .. warning::

        When you pose an upper bound on this expression, then PICOS enforces
        :math:`x \geq 0` through an auxiliary constraint during solution search.
        When an additional expression :math:`y` is given, PICOS enforces
        :math:`y \geq 0` as well.
    """

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    @convert_and_refine_arguments("x", "y", allowNone=True)
    def __init__(self, x, y=None):
        """Construct a :class:`NegativeEntropy`.

        :param x: The affine expression :math:`x`.
        :type x: ~picos.expressions.AffineExpression
        :param y: An additional affine expression :math:`y`. If necessary, PICOS
            will attempt to reshape or broadcast it to the shape of :math:`x`.
        :type y: ~picos.expressions.AffineExpression
        """
        if not isinstance(x, AffineExpression):
            raise TypeError("Can only take the elementwise logarithm of a real "
                "affine expression, not of {}.".format(type(x).__name__))

        if y is not None:
            if not isinstance(y, AffineExpression):
                raise TypeError("The additional parameter y must be a real "
                    "affine expression, not {}.".format(type(x).__name__))
            elif x.shape != y.shape:
                y = y.reshaped_or_broadcasted(x.shape)

            if y.is1:
                y = None

        self._x = x
        self._y = y

        if y is None:
            typeStr = "Negative Entropy"
            if len(x) == 1:
                symbStr = glyphs.mul(x.string, glyphs.log(x.string))
            else:
                symbStr = glyphs.dotp(x.string, glyphs.log(x.string))
        else:
            typeStr = "Relative Entropy"
            if len(x) == 1:
                symbStr = glyphs.mul(
                    x.string, glyphs.log(glyphs.div(x.string, y.string)))
            else:
                symbStr = glyphs.dotp(x.string,
                    glyphs.sub(glyphs.log(x.string), glyphs.log(y.string)))

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
            s = numpy.dot(x, numpy.log(x))
        else:
            y = numpy.ravel(cvx2np(self._y._get_value()))
            s = numpy.dot(x, numpy.log(x / y))

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

    @cached_selfinverse_unary_operator
    def __neg__(self):
        return Entropy(self._x, self._y)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __add__(self, other):
        other_str = other.string

        if isinstance(other, AffineExpression):
            if other.is0:
                return self

            other = NegativeEntropy(other, other / math.e)

        if isinstance(other, NegativeEntropy):
            if self._y is None and other._y is None:
                negent = NegativeEntropy(self._x.vec // other._x.vec)
            elif self._y is not None and other._y is None:
                one = AffineExpression.from_constant(1.0, (other.n, 1))
                negent = NegativeEntropy(
                    self._x.vec // other._x.vec, self._y.vec // one)
            elif self._y is None and other._y is not None:
                one = AffineExpression.from_constant(1.0, (self.n, 1))
                negent = NegativeEntropy(
                    self._x.vec // other._x.vec, one // other._y.vec)
            else:
                negent = NegativeEntropy(
                    self._x.vec // other._x.vec, self._y.vec // other._y.vec)

            negent._symbStr = glyphs.clever_add(self.string, other_str)

            return negent

        return Expression.__add__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __radd__(self, other):
        if isinstance(other, (AffineExpression, NegativeEntropy)):
            negent = NegativeEntropy.__add__(self, other)

            if negent is not self:
                negent._symbStr = glyphs.clever_add(other.string, self.string)

            return negent

        return Expression.__radd__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __sub__(self, other):
        if isinstance(other, (AffineExpression, Entropy)):
            return self + (-other)

        return Expression.__sub__(self, other)

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
                return KullbackLeiblerConstraint.make_type(
                    argdim=subtype.argdim)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        if isinstance(other, AffineExpression):
            return KullbackLeiblerConstraint(self, other)
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
