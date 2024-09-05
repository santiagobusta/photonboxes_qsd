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

"""Implements :class:`Logarithm`."""

import math
import operator
from collections import namedtuple

import cvxopt

from .. import glyphs
from ..apidoc import api_end, api_start
from ..constraints import LogConstraint
from .data import convert_and_refine_arguments, convert_operands
from .exp_affine import AffineExpression
from .expression import Expression, refine_operands, validate_prediction

_API_START = api_start(globals())
# -------------------------------


class Logarithm(Expression):
    r"""Logarithm of a scalar affine expression.

    :Definition:

    For a real scalar affine expression :math:`x`, this is :math:`\log(x)`.

    .. warning::

        When you pose a lower bound on a logarithm :math:`\log(x)`, then PICOS
        enforces :math:`x \geq 0` through an auxiliary constraint during
        solution search.
    """

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    @convert_and_refine_arguments("x")
    def __init__(self, x):
        """Construct a :class:`Logarithm`.

        :param x: The scalar affine expression :math:`x`.
        :type x: ~picos.expressions.AffineExpression
        """
        if not isinstance(x, AffineExpression):
            raise TypeError("Can only take the logarithm of a real affine "
                "expression, not of {}.".format(type(x).__name__))
        elif not x.scalar:
            raise TypeError("Can only take the logarithm of a scalar expression"
                "but {} is shaped {}.".format(x.string, glyphs.shape(x.shape)))

        self._x = x

        Expression.__init__(self, "Logarithm", glyphs.log(x.string))

    # --------------------------------------------------------------------------
    # Abstract method implementations and method overridings, except _predict.
    # --------------------------------------------------------------------------

    def _get_refined(self):
        if self._x.constant:
            return AffineExpression.from_constant(self.value, 1, self._symbStr)
        else:
            return self

    Subtype = namedtuple("Subtype", ())

    def _get_subtype(self):
        return self.Subtype()

    def _get_value(self):
        value = cvxopt.matrix(self._x._get_value())  # Must be dense for log.
        return cvxopt.log(value)

    def _get_mutables(self):
        return self._x._get_mutables()

    def _is_convex(self):
        return False

    def _is_concave(self):
        return True

    def _replace_mutables(self, mapping):
        return self.__class__(self._x._replace_mutables(mapping))

    def _freeze_mutables(self, freeze):
        return self.__class__(self._x._freeze_mutables(freeze))

    # --------------------------------------------------------------------------
    # Python special method implementations, except constraint-creating ones.
    # --------------------------------------------------------------------------

    @classmethod
    def _add(cls, self, other, forward):
        if isinstance(other, AffineExpression) and other.constant:
            if other.is0:
                return self

            log = cls(self._x * math.exp(other.value))
            log._typeStr = "Offset " + log._typeStr

            if forward:
                log._symbStr = glyphs.clever_add(self.string, other.string)
            else:
                log._symbStr = glyphs.clever_add(other.string, self.string)

            return log

        if forward:
            return Expression.__add__(self, other)
        else:
            return Expression.__radd__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __add__(self, other):
        return Logarithm._add(self, other, True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __radd__(self, other):
        return Logarithm._add(self, other, False)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __sub__(self, other):
        if isinstance(other, AffineExpression) and other.constant:
            log = Logarithm(self._x / math.exp(other.value))
            log._typeStr = "Offset " + log._typeStr
            log._symbStr = glyphs.clever_sub(self.string, other.string)

            return log

        return Expression.__sub__(self, other)

    @classmethod
    def _mul(cls, self, other, forward):
        from . import Entropy, NegativeEntropy

        if isinstance(other, AffineExpression):
            if other.is0:
                return AffineExpression.zero()
            elif other.is1:
                return self
            elif other.equals(self._x):
                return NegativeEntropy(self._x)
            elif other.equals(-self._x):
                return Entropy(self._x)

        if forward:
            return Expression.__mul__(self, other)
        else:
            return Expression.__rmul__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __mul__(self, other):
        return Logarithm._mul(self, other, True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __rmul__(self, other):
        return Logarithm._mul(self, other, False)

    # --------------------------------------------------------------------------
    # Methods and properties that return expressions.
    # --------------------------------------------------------------------------

    @property
    def x(self):
        """The expression :math:`x`."""
        return self._x

    @property
    def exp(self):
        """The exponential of the logarithm, equal to :math:`x`."""
        return self._x

    # --------------------------------------------------------------------------
    # Constraint-creating operators, and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        if relation == operator.__ge__:
            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                return LogConstraint.make_type()

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __ge__(self, other):
        if isinstance(other, AffineExpression):
            return LogConstraint(self, other)
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
