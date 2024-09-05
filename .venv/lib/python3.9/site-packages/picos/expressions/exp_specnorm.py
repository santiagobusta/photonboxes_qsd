# ------------------------------------------------------------------------------
# Copyright (C) 2020 Guillaume Sagnol
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

"""Implements :class:`SpectralNorm`."""

import operator
from collections import namedtuple

import cvxopt
import numpy

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_unary_operator
from ..constraints import AbsoluteValueConstraint, SpectralNormConstraint
from .data import convert_and_refine_arguments, convert_operands, cvx2np
from .exp_affine import AffineExpression, ComplexAffineExpression
from .exp_norm import Norm
from .expression import Expression, refine_operands, validate_prediction

_API_START = api_start(globals())
# -------------------------------


class SpectralNorm(Expression):
    r"""The spectral norm of a matrix.

    This class can represent the spectral norm of a matrix-affine expression
    (real- or complex valued). The spectral norm is convex, so we can form
    expressions of the form ``SpectralNorm(X) <= t`` which are typically
    reformulated as LMIs that can be handled by SDP solvers.

    :Definition:

    If the normed expression is a matrix :math:`X`, then its spectral norm is

        .. math::

            \|X\|_2 = \max \{  \|Xu\|_2 : \|u\| \leq  1\}
                    = \sqrt{\lambda_{\max}(XX^*)},

    where :math:`\lambda_{\max}(\cdot)` denotes the largest eigenvalue of
    a matrix, and :math:`X^*` denotes the adjoint matrix of :math:`X`
    (i.e., the transposed matrix :math:`X^T` if :math:`X` is real-valued).

    Special cases:

    -   If :math:`X` is scalar, then :math:`\|X\|_2` reduces to the the absolute
        value (or modulus) :math:`|X|`.
    -   If :math:`X` is scalar, then :math:`\|X\|_2` coincides with the
        Euclidean norm of :math:`X`.

    """

    @convert_and_refine_arguments("x")
    def __init__(self, x):
        """Construct a :class:`SpectralNorm`.

        :param x: The affine expression to take the norm of.
        :type x: ~picos.expressions.ComplexAffineExpression
        """
        # Validate x.
        if not isinstance(x, ComplexAffineExpression):
            raise TypeError("Can only form the spectral norm of an affine "
                "expression, not of {}.".format(type(x).__name__))

        complex = not isinstance(x, AffineExpression)

        # Build the string representations.
        if len(x) == 1:
            typeStr = "Modulus" if complex else "Absolute Value"
            symbStr = glyphs.abs(x.string)
        elif 1 in x.shape:
            typeStr = "Euclidean Norm"
            symbStr = glyphs.norm(x.string)
        else:
            typeStr = "Spectral Norm"
            symbStr = glyphs.spnorm(x.string)

        if complex:
            typeStr = "Complex " + typeStr

        self._x = x
        self._complex = complex
        Expression.__init__(self, typeStr, symbStr)

    # --------------------------------------------------------------------------
    # Abstract method implementations and method overridings, except _predict.
    # --------------------------------------------------------------------------

    @cached_unary_operator
    def _get_refined(self):
        if self._x.constant:
            return AffineExpression.from_constant(self.value, 1, self.string)
        elif len(self._x) == 1 or (1 in self._x.shape):
            return Norm(self._x)
        else:
            return self

    Subtype = namedtuple("Subtype", ("argshape", "complex", "hermitian"))

    def _get_subtype(self):
        return self.Subtype(self._x.shape, self._complex, self._x.hermitian)

    def _get_value(self):
        value = self._x._get_value()
        value = cvx2np(value)
        value = numpy.linalg.norm(value, 2)
        return cvxopt.matrix(value)

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

                norm = cls(other*self._x)
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
        return SpectralNorm._mul(self, other, True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __rmul__(self, other):
        return SpectralNorm._mul(self, other, False)

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

        arg_shape, arg_complex, arg_hermitian = subtype
        xLen = arg_shape[0] * arg_shape[1]

        if relation == operator.__le__:
            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                if xLen == 1:
                    return AbsoluteValueConstraint.make_type()
                elif 1 in arg_shape:
                    assert False, "Unexpected case (should have been refined)"
                else:
                    return SpectralNormConstraint.make_type(
                        arg_shape, arg_complex, arg_hermitian)
        elif relation == operator.__ge__:
            return NotImplemented  # Not concave.

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        assert self.convex

        if isinstance(other, AffineExpression):
            if len(self._x) == 1:
                return AbsoluteValueConstraint(self._x, other)
            elif 1 in self._x.shape:
                assert False, "Unexpected case (should have been refined)"
            else:
                return SpectralNormConstraint(self, other)
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
