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

"""Implements :class:`SquaredNorm`."""

import operator
from collections import namedtuple

import cvxopt

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property, cached_unary_operator
from ..constraints import SquaredNormConstraint
from .data import convert_and_refine_arguments, convert_operands, cvxopt_hcat
from .exp_affine import AffineExpression, ComplexAffineExpression
from .exp_quadratic import QuadraticExpression
from .expression import Expression, refine_operands, validate_prediction

_API_START = api_start(globals())
# -------------------------------


class SquaredNorm(QuadraticExpression):
    """A squared Euclidean or Frobenius norm.

    This is a lightweight wrapper around
    :class:`~picos.expressions.QuadraticExpression` that can handle common
    constraint formulations more efficiently.
    """

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    @convert_and_refine_arguments("x")
    def __init__(self, x):
        """Create a squared Euclidean or Frobenius norm.

        :param x:
            The (complex) affine expression under the squared norm.
        :type affinePart:
            ~picos.expressions.ComplexAffineExpression
        """
        # Validate x.
        if not isinstance(x, ComplexAffineExpression):
            raise TypeError("Can only form the squared norm of an affine "
                "expression, not of {}.".format(type(x).__name__))

        if len(x) == 1:
            typeStr = "Squared Scalar"
            symbStr = glyphs.squared(x.string)
        else:
            typeStr = "Squared Norm"
            symbStr = glyphs.squared(glyphs.norm(x.string))

        Expression.__init__(self, typeStr, symbStr)

        # TODO: Add a nonzero-vectorization to BiaffineExpression that returns
        #       a scalar zero for an all-zero expression and the vectorization
        #       of the expression with zero rows removed otherwise.
        if x.is0:
            self._x = AffineExpression.zero()
        else:
            # Vectorize and stack real and imaginary parts.
            vec = x.vec if x.isreal else x.vec.real // x.vec.imag

            # Remove zero rows from the vectorization.
            A = abs(cvxopt_hcat(vec._coefs.values()))
            a = cvxopt.sparse(sum(A[:, j] for j in range(A.size[1])))
            nonzero = a.I
            nnz = len(nonzero)
            B = cvxopt.spmatrix([1.0]*nnz, range(nnz), nonzero, (nnz, len(x)))

            self._x = B*vec

    # --------------------------------------------------------------------------
    # Allow inheriting from QuadraticExpression.
    # --------------------------------------------------------------------------

    @cached_property
    def _quadratic_form(self):
        """The squared norm as a pure quadratic expression.

        If the expression under the norm is constant, then this is :obj:`None`.
        """
        # HACK: Make a shallow copy of self._x so that the product does not
        #       recognize that the operation below represents a squared norm.
        #       This only works as long as the product checks for operand
        #       equality with the "is" keyword.
        result = (self._x.renamed("HACK") | self._x)

        if isinstance(result, AffineExpression):
            return None
        else:
            assert isinstance(result, QuadraticExpression)
            result._symbStr = self.string
            return result

    @cached_property
    def _quads(self):
        if self._quadratic_form:
            return self._quadratic_form._quads
        else:
            return {}

    @cached_property
    def _aff(self):
        if self._quadratic_form:
            return self._quadratic_form._aff
        else:
            refined = self.refined
            assert isinstance(refined, ComplexAffineExpression)
            return refined

    @cached_property
    def _sf(self):
        if len(self._x) == 1:
            return (self._x, self._x)
        else:
            return None

    # --------------------------------------------------------------------------
    # Squared norm specific properties.
    # --------------------------------------------------------------------------

    @property
    def argdim(self):
        """Number of nonzero elements of the expression under the norm."""
        return len(self._x)

    # --------------------------------------------------------------------------
    # Abstract method implementations and method overridings, except _predict.
    # --------------------------------------------------------------------------

    @cached_unary_operator
    def _get_refined(self):
        if self._x.constant:
            value = self._x.value_as_matrix
            return AffineExpression.from_constant(
                value.T*value, (1, 1), self._symbStr)
        else:
            return self

    Subtype = namedtuple("Subtype", ("argdim", "quadratic_subtype"))

    @cached_unary_operator
    def _get_subtype(self):
        return self.Subtype(
            len(self._x),
            self._quadratic_form.subtype if self._quadratic_form else None)

    def _get_value(self):
        value = self._x._get_value()
        return value.T*value

    @cached_unary_operator
    def _get_variables(self):
        return self._x.variables

    def _is_convex(self):
        return True

    def _is_concave(self):
        return self._x.constant

    def _replace_variables(self, var_map):
        return self.__class__(self._x._replace_variables(var_map))

    # --------------------------------------------------------------------------
    # Python special method implementations, except constraint-creating ones.
    # --------------------------------------------------------------------------

    @convert_operands(sameShape=True)
    @refine_operands()
    def __add__(self, other):
        """Denote addition from the right hand side."""
        if isinstance(other, SquaredNorm):  # No need to have __radd__ for this.
            result = SquaredNorm(self._x // other._x)
            result._symbStr = glyphs.clever_add(self.string, other.string)

            return result

        return QuadraticExpression.__add__(self, other)

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

                result = cls(self._x*factor**0.5)
                result._typeStr = "Scaled " + result._typeStr
                result._symbStr = string

                return result

        if div:
            return QuadraticExpression.__truediv__(self, other)
        elif forward:
            return QuadraticExpression.__mul__(self, other)
        else:
            return QuadraticExpression.__rmul__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __mul__(self, other):
        """Denote scaling from the right hand side."""
        return SquaredNorm._mul_div(self, other, div=False, forward=True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __rmul__(self, other):
        """Denote scaling from the left hand side."""
        return SquaredNorm._mul_div(self, other, div=False, forward=False)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __truediv__(self, other):
        """Denote division by a constant scalar."""
        return SquaredNorm._mul_div(self, other, div=True, forward=True)

    # --------------------------------------------------------------------------
    # Method overridings for QuadraticExpression.
    # --------------------------------------------------------------------------

    @cached_property
    def fullroot(self):
        """Affine expression whose squared norm equals the expression.

        Overrides :meth:`~.exp_quadratic.QuadraticExpression.fullroot` of
        :class:`~.exp_quadratic.QuadraticExpression`.
        """
        return self._x.renamed(glyphs.Fn("fullroot({})")(self.string))

    @property
    def is_squared_norm(self):
        """Always :obj:`True` for squared norm instances.

        Overrides :meth:`~.exp_quadratic.QuadraticExpression.is_squared_norm`
        of :class:`~.exp_quadratic.QuadraticExpression`.
        """
        return True

    @property
    def is0(self):
        """Whether the expression is zero.

        Overrides :meth:`~.exp_quadratic.QuadraticExpression.is0` of
        :class:`~.exp_quadratic.QuadraticExpression`.
        """
        return self._x.is0

    # --------------------------------------------------------------------------
    # Constraint-creating operators, and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        if relation == operator.__le__:
            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                return SquaredNormConstraint.make_type(
                    subtype.argdim, other.subtype.constant)

        return QuadraticExpression._predict(
            subtype.quadratic_subtype, relation, other)

    @convert_operands(sameShape=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        if isinstance(other, AffineExpression):
            return SquaredNormConstraint(self, other)

        # NOTE: The following should handle the case where the upper bound has a
        #       scalar factorization efficiently by virtue of
        #       SquaredNorm.fullroot. See ConicQuadraticConstraint.
        return QuadraticExpression.__le__(self, other)


# --------------------------------------
__all__ = api_end(_API_START, globals())
