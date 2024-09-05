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

"""Implements :class:`GeometricMean`."""

import operator
from collections import namedtuple

import cvxopt
import numpy

from .. import glyphs
from ..apidoc import api_end, api_start
from ..constraints import GeometricMeanConstraint
from .data import convert_and_refine_arguments, convert_operands, cvx2np
from .exp_affine import AffineExpression
from .expression import Expression, refine_operands, validate_prediction

_API_START = api_start(globals())
# -------------------------------


class GeometricMean(Expression):
    r"""Geometric mean of an affine expression.

    :Definition:

    For an :math:`n`-dimensional affine expression :math:`x` with
    :math:`x \geq 0`, the geometric mean is given as

    .. math::

        \left(\prod_{i = 1}^n x_i \right)^{\frac{1}{n}}.

    .. warning::

        When you pose a lower bound on a geometric mean, then PICOS enforces
        :math:`x \geq 0` through an auxiliary constraint during solution search.
    """

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    @convert_and_refine_arguments("x")
    def __init__(self, x):
        """Construct a :class:`GeometricMean`.

        :param x: The affine expression to form the geometric mean of.
        :type x: ~picos.expressions.AffineExpression
        """
        if not isinstance(x, AffineExpression):
            raise TypeError("Can only form the geometrtic mean of a real affine"
                " expression, not of {}.".format(type(x).__name__))

        self._x = x

        Expression.__init__(
            self, "Geometric Mean", glyphs.make_function("geomean")(x.string))

    # --------------------------------------------------------------------------
    # Abstract method implementations and method overridings, except _predict.
    # --------------------------------------------------------------------------

    def _get_refined(self):
        if self._x.constant:
            return AffineExpression.from_constant(self.value, 1, self._symbStr)
        elif len(self._x) == 1:
            return self._x.renamed(self._symbStr)
        else:
            return self

    Subtype = namedtuple("Subtype", ("argdim"))

    def _get_subtype(self):
        return self.Subtype(len(self._x))

    def _get_value(self):
        value = self._x._get_value()
        value = numpy.prod(cvx2np(value))**(1.0 / len(self._x))
        return cvxopt.matrix(value)

    def _get_mutables(self):
        return self._x._get_mutables()

    def _is_convex(self):
        return False

    def _is_concave(self):
        return True  # Only for nonnegative x.

    def _replace_mutables(self, mapping):
        return self.__class__(self._x._replace_mutables(mapping))

    def _freeze_mutables(self, freeze):
        return self.__class__(self._x._freeze_mutables(freeze))

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

                product = cls(other*self._x)
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
        return GeometricMean._mul(self, other, True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __rmul__(self, other):
        return GeometricMean._mul(self, other, False)

    # --------------------------------------------------------------------------
    # Methods and properties that return modified copies.
    # --------------------------------------------------------------------------

    @property
    def x(self):
        """The expression under the mean."""
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
                return GeometricMeanConstraint.make_type(subtype.argdim)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __ge__(self, other):
        if isinstance(other, AffineExpression):
            return GeometricMeanConstraint(self, other)
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
