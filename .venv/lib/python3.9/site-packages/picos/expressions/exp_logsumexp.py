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

"""Implements :class:`LogSumExp`."""

import operator
from collections import namedtuple

import cvxopt
import numpy

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from ..constraints import LogSumExpConstraint
from .data import convert_and_refine_arguments, convert_operands, cvx2np
from .exp_affine import AffineExpression
from .expression import Expression, refine_operands, validate_prediction

_API_START = api_start(globals())
# -------------------------------


class LogSumExp(Expression):
    r"""Logarithm of the sum of elementwise exponentials of an expression.

    :Definition:

    For an :math:`n`-dimensional real affine expression :math:`x`, this is the
    logarithm of the sum of elementwise exponentials

    .. math::

        \log\sum_{i = 1}^n \exp(\operatorname{vec}(x)_i).
    """

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    @convert_and_refine_arguments("x")
    def __init__(self, x):
        """Construct a :class:`LogSumExp`.

        :param x: The affine expression :math:`x`.
        :type x: ~picos.expressions.AffineExpression
        """
        if not isinstance(x, AffineExpression):
            raise TypeError("Can only form the logarithm of the sum of "
                "elementwise exponentials of a real affine expression, not of "
                "{}.".format(type(x).__name__))

        self._x = x

        typeStr = "Logarithm of Sum of Exponentials"
        symbStr = glyphs.make_function("log", "sum", "exp")(x.string)

        Expression.__init__(self, typeStr, symbStr)

    # --------------------------------------------------------------------------
    # Abstract method implementations and method overridings, except _predict.
    # --------------------------------------------------------------------------

    def _get_refined(self):
        if self._x.constant:
            return AffineExpression.from_constant(self.value, 1, self._symbStr)
        elif len(self._x) == 1:
            return self._x  # Don't carry the string for an identity.
        else:
            return self

    Subtype = namedtuple("Subtype", ("argdim"))

    def _get_subtype(self):
        return self.Subtype(len(self._x))

    def _get_value(self):
        x = numpy.ravel(cvx2np(self._x._get_value()))
        s = numpy.log(numpy.sum(numpy.exp(x)))
        return cvxopt.matrix(s)

    def _get_mutables(self):
        return self._x._get_mutables()

    def _is_convex(self):
        return True

    def _is_concave(self):
        return False

    def _replace_mutables(self, mapping):
        return self.__class__(self._x._replace_mutables(mapping))

    def _freeze_mutables(self, freeze):
        return self.__class__(self._x._freeze_mutables(freeze))

    # --------------------------------------------------------------------------
    # Python special method implementations, except constraint-creating ones.
    # --------------------------------------------------------------------------

    @classmethod
    def _add(cls, self, other, forward):
        if isinstance(other, AffineExpression):
            if other.is0:
                return self

            lse = cls(self._x + other)

            if forward:
                lse._symbStr = glyphs.clever_add(self.string, other.string)
            else:
                lse._symbStr = glyphs.clever_add(other.string, self.string)

            return lse

        if forward:
            return Expression.__add__(self, other)
        else:
            return Expression.__radd__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __add__(self, other):
        return LogSumExp._add(self, other, True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __radd__(self, other):
        return LogSumExp._add(self, other, False)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __sub__(self, other):
        if isinstance(other, AffineExpression):
            if other.is0:
                return self

            lse = LogSumExp(self._x - other)
            lse._symbStr = glyphs.clever_sub(self.string, other.string)

            return lse

        return Expression.__sub__(self, other)

    # --------------------------------------------------------------------------
    # Methods and properties that return expressions.
    # --------------------------------------------------------------------------

    @property
    def x(self):
        """The expression :math:`x`."""
        return self._x

    @cached_property
    def exp(self):
        """The elementwise sum of exponentials of :math:`x`."""
        from . import SumExponentials
        return SumExponentials(self._x)

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
                return LogSumExpConstraint.make_type(subtype.argdim)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        if isinstance(other, AffineExpression):
            return LogSumExpConstraint(self, other)
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
