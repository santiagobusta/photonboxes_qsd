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

"""Implements :class:`PowerTrace`."""

import operator
from collections import namedtuple

import cvxopt
import numpy

from .. import glyphs
from ..apidoc import api_end, api_start
from ..constraints import Constraint, PowerTraceConstraint
from .data import (convert_and_refine_arguments, convert_operands, cvx2np,
                   cvxopt_hpsd, make_fraction)
from .exp_affine import AffineExpression, ComplexAffineExpression, Constant
from .expression import Expression, refine_operands, validate_prediction

_API_START = api_start(globals())
# -------------------------------


class PowerTrace(Expression):
    r"""The trace of the :math:`p`-th power of a hermitian matrix.

    :Definition:

    Let :math:`p \in \mathbb{Q}`.

    1.  If the base expressions is a real scalar :math:`x` and no additional
        constant :math:`m` is given, then this is the power :math:`x^p`.

    2.  If the base expressions is a real scalar :math:`x`,
        :math:`p \in [0, 1]`, and a positive scalar constant :math:`m` is given,
        then this is the scaled power :math:`m x^p`.

    3.  If the base expression is a hermitian matrix :math:`X` and no additional
        constant :math:`M` is given, then this is the trace of power
        :math:`\operatorname{tr}(X^p)`.

    4.  If the base expression is a hermitian matrix :math:`X`,
        :math:`p \in [0, 1]`, and a hermitian positive semidefinite constant
        matrix :math:`M` of same shape as :math:`X` is given, then this is the
        trace of a scaled power :math:`\operatorname{tr}(M X^p)`.

    No other case is supported. In particular, if :math:`p \not\in [0, 1]`, then
    :math:`m`/:math:`M` must be undefined (:obj:`None`).

    .. warning::

        1. For a constraint of the form :math:`x^p \leq t` with :math:`p < 1`
           and :math:`p \neq 0`, PICOS enforces :math:`x \geq 0` during solution
           search.

        2. For a constraint of the form :math:`\operatorname{tr}(X^p) \leq t` or
           :math:`\operatorname{tr}(M X^p) \leq t` with :math:`p < 1` and
           :math:`p \neq 0`, PICOS enforces :math:`X \succeq 0` during solution
           search.

        3. For a constraint of the form :math:`\operatorname{tr}(X^p) \leq t`
           or :math:`\operatorname{tr}(M X^p) \leq t` with :math:`p > 1`, PICOS
           enforces :math:`t \geq 0` during solution search.
    """

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    @convert_and_refine_arguments("x")
    def __init__(self, x, p, m=None, denominator_limit=1000):
        """Construct a :class:`PowerTrace`.

        :param x: The scalar or symmetric matrix to form a power of.
        :type x: ~picos.expressions.AffineExpression
        :param float p: The value for :math:`p`, which is cast to a limited
            precision fraction.
        :param m: An additional positive semidefinite constant to multiply the
            power with.
        :type m: :class:`~picos.expressions.AffineExpression` or anything
            recognized by :func:`~picos.expressions.data.load_data`
        :param int denominator_limit: The largest allowed denominator when
            casting :math:`p` to a fraction. Higher values can yield a greater
            precision at reduced performance.
        """
        # Validate x.
        if not isinstance(x, ComplexAffineExpression):
            raise TypeError("Can only form the power of an affine expression, "
                "not of {}.".format(x.string))
        elif not x.square:
            raise TypeError(
                "Can't form the power of non-square {}.".format(x.string))
        elif not x.hermitian:
            raise NotImplementedError("Taking {} to a power is not supported "
                "as it is not necessarily hermitian.".format(x.string))

        # Load p.
        pNum, pDen, p, pStr = make_fraction(p, denominator_limit)

        # Load m.
        if m is not None:
            mStr = "m" if len(x) == 1 else "M"

            if p < 0 or p > 1:
                raise ValueError(
                    "p-th power with an additional factor {} requires {}."
                    .format(mStr, glyphs.le(0, glyphs.le("p", 1))))

            if not isinstance(m, ComplexAffineExpression):
                try:
                    m = Constant(mStr, m, x.shape)
                except Exception as error:
                    raise TypeError(
                        "Failed to load the additional factor {} as a matrix of"
                        " same shape as {}.".format(mStr, x.string)) from error
            else:
                m = m.refined

            if not m.constant:
                raise TypeError("The additional factor {} is not constant."
                    .format(m.string))
            elif not cvxopt_hpsd(m.safe_value_as_matrix):
                raise ValueError("The additional factor {} is not hermitian "
                    "positive semidefinite.".format(m.string))

        self._x     = x
        self._num   = pNum
        self._den   = pDen
        self._m     = m
        self._limit = denominator_limit

        if m is None:
            if len(x) == 1:
                typeStr = "Power"
                if p == 2:
                    symbStr = glyphs.squared(x.string)
                elif p == 3:
                    symbStr = glyphs.cubed(x.string)
                else:
                    symbStr = glyphs.power(x.string, pStr)
            else:
                typeStr = "Trace of Power"
                symbStr = glyphs.trace(glyphs.power(x.string, pStr))
        else:
            if len(x) == 1:
                typeStr = "Scaled Power"
                symbStr = glyphs.mul(m.string, glyphs.power(x.string, pStr))
            else:
                typeStr = "Trace of Scaled Power"
                symbStr = glyphs.trace(glyphs.mul(
                    m.string, glyphs.power(x.string, pStr)))

        Expression.__init__(self, typeStr, symbStr)

    # --------------------------------------------------------------------------
    # Abstract method implementations and method overridings, except _predict.
    # --------------------------------------------------------------------------

    def _get_refined(self):
        if self._x.constant:
            return Constant(self._symbStr, self.value)
        elif self.p == 0:
            if self._m is not None:
                return self._m.tr
            else:
                return Constant(
                    glyphs.Fn("diaglen({})")(self._x.string), self._x.shape[0])
        elif self.p == 1:
            if self._m is not None:
                # NOTE: No hermitian transpose as both m and x are hermitian.
                return (self._m | self._x)
            else:
                return self._x.tr
        elif self.p == 2 and self._x.scalar and self._m is None:
            return (self._x | self._x)
        else:
            return self

    Subtype = namedtuple("Subtype", ("diag", "num", "den", "hasM", "complex"))

    def _get_subtype(self):
        return self.Subtype(
            self._x.shape[0], self._num, self._den, self._m is not None,
            self._x.complex)

    def _get_value(self):
        x = cvx2np(self._x._get_value())
        p = self.p

        eigenvalues = numpy.linalg.eigvalsh(x)

        if p != int(p) and any(value < 0 for value in eigenvalues):
            raise ArithmeticError("Cannot evaluate {}: {} is not positive "
                "semidefinite and the exponent is fractional."
                .format(self.string, self._x.string))

        if self._m is None:
            trace = sum([value**p for value in eigenvalues])
        else:
            m = cvx2np(self._m._get_value())

            U, S, V = numpy.linalg.svd(x)
            power = U*numpy.diag(S**p)*V
            trace = numpy.trace(m * power)

        return cvxopt.matrix(trace)

    def _get_mutables(self):
        return self._x._get_mutables()

    def _is_convex(self):
        return self.p >= 1 or self.p <= 0

    def _is_concave(self):
        return self.p >= 0 and self.p <= 1

    def _replace_mutables(self, mapping):
        return self.__class__(
            self._x._replace_mutables(mapping), self.p, self._m, self._limit)

    def _freeze_mutables(self, freeze):
        return self.__class__(
            self._x._freeze_mutables(freeze), self.p, self._m, self._limit)

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
            elif factor > 0 and self.p >= 0 and self.p <= 1:
                if self._m is None:
                    m = other.dupdiag(self.n).renamed(other.string)
                else:
                    m = other*self._m

                return cls(self._x, self.p, m, self._limit)

        if forward:
            return Expression.__mul__(self, other)
        else:
            return Expression.__rmul__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __mul__(self, other):
        return PowerTrace._mul(self, other, True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __rmul__(self, other):
        return PowerTrace._mul(self, other, False)

    # --------------------------------------------------------------------------
    # Methods and properties that return expressions.
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

    @property
    def p(self):
        """The parameter :math:`p`.

        This is a limited precision version of the parameter used when the
        expression was constructed.
        """
        return float(self._num) / float(self._den)

    @property
    def num(self):
        """The limited precision fraction numerator of :math:`p`."""
        return self._num

    @property
    def den(self):
        """The limited precision fraction denominator of :math:`p`."""
        return self._den

    @property
    def m(self):
        """An additional factor to multiply the power with."""
        return self._m

    # --------------------------------------------------------------------------
    # Constraint-creating operators, and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        p = float(subtype.num) / float(subtype.den)

        if relation == operator.__le__:
            if p > 0 and p < 1:  # Not convex.
                return NotImplemented

            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                return PowerTraceConstraint.make_type(*subtype)
        elif relation == operator.__ge__:
            if p < 0 or p > 1:  # Not concave.
                return NotImplemented

            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                return PowerTraceConstraint.make_type(*subtype)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        if not self.convex:
            raise TypeError("Cannot upper-bound a nonconvex (trace of) power.")

        if isinstance(other, AffineExpression):
            return PowerTraceConstraint(self, Constraint.LE, other)
        else:
            return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __ge__(self, other):
        if not self.concave:
            raise TypeError("Cannot lower-bound a nonconcave (trace of) power.")

        if isinstance(other, AffineExpression):
            return PowerTraceConstraint(self, Constraint.GE, other)
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
