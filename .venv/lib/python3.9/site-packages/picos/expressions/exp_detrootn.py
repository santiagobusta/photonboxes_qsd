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

"""Implements :class:`DetRootN`."""

import operator
from collections import namedtuple

import cvxopt
import numpy

from .. import glyphs
from ..apidoc import api_end, api_start
from ..constraints import DetRootNConstraint
from .data import convert_and_refine_arguments, convert_operands, cvx2np
from .exp_affine import AffineExpression, ComplexAffineExpression
from .expression import Expression, refine_operands, validate_prediction

_API_START = api_start(globals())
# -------------------------------


class DetRootN(Expression):
    r"""The :math:`n`-th root of the determinant of an :math:`n\times n` matrix.

    :Definition:

    For an :math:`n \times n` positive semidefinite hermitian matrix :math:`X`,
    this is

    .. math::

        \sqrt[n]{\det X}.

    .. warning::

        When you pose a lower bound on the :math:`n`-th root of a determinant of
        the matrix :math:`X`, then PICOS enforces positive semidefiniteness
        :math:`X \succeq 0` through an auxiliary constraint during solution
        search.
    """

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    @convert_and_refine_arguments("x")
    def __init__(self, x):
        """Construct a :class:`DetRootN`.

        :param x: The matrix concerned. Must be hermitian by definition.
        :type x: ~picos.expressions.ComplexAffineExpression
        """
        if not isinstance(x, ComplexAffineExpression):
            raise TypeError("Can only form the determinant of an affine "
                "expression, not of {}.".format(type(x).__name__))
        elif not x.square:
            raise TypeError("Can't take the determinant of non-square {0}."
                .format(x.string))
        elif not x.hermitian:
            raise NotImplementedError("Taking the n-th root of the determinant "
                "of {0} is not supported as {0} is not necessarily hermitian."
                .format(x.string))

        self._x = x

        Expression.__init__(self, "n-th Root of a Determinant",
            glyphs.power(glyphs.det(x.string), glyphs.div(1, x.shape[0])))

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

    Subtype = namedtuple("Subtype", ("diag", "complex"))

    def _get_subtype(self):
        return self.Subtype(self.n, self._x.complex)

    def _get_value(self):
        value = self._x._get_value()

        det = numpy.linalg.det(cvx2np(value))

        if det < 0:
            raise ArithmeticError("Cannot evaluate {}: {} is negative."
                .format(self.string, glyphs.eq(glyphs.det(self.x.string), det)))

        return cvxopt.matrix(det**(1.0 / self._x.shape[0]))

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

                result = cls(other*self._x)
                result._typeStr = "Scaled " + result._typeStr
                result._symbStr = string

                return result

        if forward:
            return Expression.__mul__(self, other)
        else:
            return Expression.__rmul__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __mul__(self, other):
        return DetRootN._mul(self, other, True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __rmul__(self, other):
        return DetRootN._mul(self, other, False)

    # --------------------------------------------------------------------------
    # Methods and properties that return modified copies.
    # --------------------------------------------------------------------------

    @property
    def x(self):
        """The matrix concerned."""
        return self._x

    # --------------------------------------------------------------------------
    # Methods and properties that describe the expression.
    # --------------------------------------------------------------------------

    @property
    def n(self):
        """Diagonal length of :attr:`x`."""
        return self._x.shape[0]

    # --------------------------------------------------------------------------
    # Constraint-creating operators, and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        if relation == operator.__ge__:
            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                return DetRootNConstraint.make_type(
                    diag=subtype.diag, complex=subtype.complex)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __ge__(self, other):
        if isinstance(other, AffineExpression):
            return DetRootNConstraint(self, other)
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
