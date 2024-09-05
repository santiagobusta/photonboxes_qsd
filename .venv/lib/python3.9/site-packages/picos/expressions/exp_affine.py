# ------------------------------------------------------------------------------
# Copyright (C) 2019-2022 Maximilian Stahlberg
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

"""Implements affine expression types."""

import operator
from collections import namedtuple

import cvxopt
import numpy

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property, cached_unary_operator
from ..constraints import (AffineConstraint, ComplexAffineConstraint,
                           ComplexLMIConstraint, Constraint, LMIConstraint)
from .data import convert_operands, cvx2np, cvx2csc, load_data
from .exp_biaffine import BiaffineExpression
from .expression import (Expression, ExpressionType, refine_operands,
                         validate_prediction)

_API_START = api_start(globals())
# -------------------------------


class ComplexAffineExpression(BiaffineExpression):
    """A multidimensional (complex) affine expression.

    Base class for the real :class:`AffineExpression`.
    """

    # --------------------------------------------------------------------------
    # Abstract method implementations for Expression.
    # --------------------------------------------------------------------------

    Subtype = namedtuple("Subtype", ("shape", "constant", "nonneg"))
    Subtype.dim = property(lambda self: self.shape[0] * self.shape[1])

    def _get_subtype(self):
        """Implement :meth:`~.expression.Expression._get_subtype`."""
        nonneg = self.constant and self.isreal \
            and all(x >= 0 for x in self.value_as_matrix)

        return self.Subtype(self._shape, self.constant, nonneg)

    # --------------------------------------------------------------------------
    # Method overridings for Expression.
    # --------------------------------------------------------------------------

    @cached_unary_operator
    def _get_refined(self):
        """Implement :meth:`~.expression.Expression._get_refined`."""
        if self.isreal:
            return AffineExpression(self._symbStr, self._shape, self._coefs)
        else:
            return self

    @convert_operands(sameShape=True, allowNone=True)
    def _set_value(self, value):
        """Override :meth:`~.expression.Expression._set_value`."""
        if value is None:
            for var in self._linear_coefs:
                var.value = None
            return

        # Since all variables are real-valued, prevent NumPy from finding
        # complex-valued solutions that do not actually work.
        (self.real // self.imag).renamed(self.string).value \
            = (value.real // value.imag)

    # --------------------------------------------------------------------------
    # Abstract method implementations for BiaffineExpression.
    # --------------------------------------------------------------------------

    @classmethod
    def _get_bilinear_terms_allowed(cls):
        """Implement for :class:`~.exp_biaffine.BiaffineExpression`."""
        return False

    @classmethod
    def _get_parameters_allowed(cls):
        """Implement for :class:`~.exp_biaffine.BiaffineExpression`."""
        return False

    @classmethod
    def _get_basetype(cls):
        """Implement :meth:`~.exp_biaffine.BiaffineExpression._get_basetype`."""
        return ComplexAffineExpression

    @classmethod
    def _get_typecode(cls):
        """Implement :meth:`~.exp_biaffine.BiaffineExpression._get_typecode`."""
        return "z"

    # --------------------------------------------------------------------------
    # Method overridings for BiaffineExpression: Binary operators.
    # --------------------------------------------------------------------------

    @convert_operands(sameShape=True)
    @refine_operands(stop_at_affine=True)
    def __or__(self, other):
        from .exp_quadratic import QuadraticExpression
        from .exp_sqnorm import SquaredNorm

        if isinstance(other, ComplexAffineExpression) \
        and not self.constant and not other.constant:
            # Create a squared norm if possible.
            # NOTE: Must not check self.equals(other) here; see SquaredNorm.
            # TODO: Consider creating a helper function for __or__ that always
            #       returns a QuadraticExpression instead of a SquaredNorm to be
            #       used within SquaredNorm. Then equals would be possible here.
            if self is other:
                return SquaredNorm(self)

            string = glyphs.clever_dotp(
                self.string, other.string, other.complex, self.scalar)

            # Handle the complex case: Conjugate the right hand side.
            other = other.conj

            Cs, Co = self._constant_coef, other._constant_coef

            # Compute the affine part of the product.
            affString = glyphs.affpart(string)
            affCoefs = {(): Cs.T * Co}
            for var in self.variables.union(other.variables):
                if var not in other._linear_coefs:
                    affCoefs[var] = Co.T * self._linear_coefs[var]
                elif var not in self._linear_coefs:
                    affCoefs[var] = Cs.T * other._linear_coefs[var]
                else:
                    affCoefs[var] = Co.T * self._linear_coefs[var] + \
                        Cs.T * other._linear_coefs[var]
            affPart = self._common_basetype(other)(affString, (1, 1), affCoefs)

            # Compute the quadratic part of the product.
            quadPart = {(v, w): self._linear_coefs[v].T * other._linear_coefs[w]
                for v in self._linear_coefs for w in other._linear_coefs}

            # Don't create quadratic expressions without a quadratic part.
            if not any(quadPart.values()):
                affPart._symbStr = string
                return affPart

            # Remember a factorization into two real scalars if applicable.
            # NOTE: If the user enters a multiplication a*b of two scalar affine
            #       expressions, then we have, at this point, self == a.T == a
            #       and other == b.conj.conj == b.
            if len(self) == 1 and len(other) == 1 \
            and self.isreal and other.isreal:
                factors = (self.refined, other.refined)
            else:
                factors = None

            return QuadraticExpression(
                string, quadPart, affPart, scalarFactors=factors)
        else:
            return BiaffineExpression.__or__(self, other)

    @convert_operands(rMatMul=True)
    @refine_operands(stop_at_affine=True)
    def __mul__(self, other):
        if isinstance(other, ComplexAffineExpression) \
        and not self.constant and not other.constant:
            # If the result is scalar, allow for quadratic terms.
            if self._shape[0] == 1 and other._shape[1] == 1 \
            and self._shape[1] == other._shape[0]:
                result = self.T.__or__(other.conj)

                # NOTE: __or__ always creates a fresh expression.
                result._symbStr = glyphs.clever_mul(self.string, other.string)

                return result
            else:
                raise NotImplementedError(
                    "PICOS does not support multidimensional quadratic "
                    "expressions at this point. More precisely, one factor must"
                    " be constant or the result must be scalar.")
        else:
            return BiaffineExpression.__mul__(self, other)

    @convert_operands(sameShape=True)
    @refine_operands(stop_at_affine=True)
    def __xor__(self, other):
        if isinstance(other, ComplexAffineExpression) \
        and not self.constant and not other.constant:
            # If the result is scalar, allow for quadratic terms.
            if self._shape == (1, 1):
                result = self.__or__(other.conj)

                # NOTE: __or__ always creates a fresh expression.
                result._symbStr = glyphs.hadamard(self.string, other.string)

                return result
            else:
                raise NotImplementedError(
                    "PICOS does not support multidimensional quadratic "
                    "expressions at this point. More precisely, one factor must"
                    " be constant or the result must be scalar.")
        else:
            return BiaffineExpression.__xor__(self, other)

    # TODO: Create a quadratic expression from a scalar Kronecker prod.

    # --------------------------------------------------------------------------
    # Method overridings for BiaffineExpression: Unary operators.
    # --------------------------------------------------------------------------

    @cached_property
    def real(self):
        """Override :meth:`~.exp_biaffine.BiaffineExpression.real`.

        The result is returned as an :meth:`AffineExpression`.
        """
        return AffineExpression(glyphs.real(self.string), self._shape,
            {vars: coef.real() for vars, coef in self._coefs.items()})

    @cached_property
    def imag(self):
        """Override :meth:`~.exp_biaffine.BiaffineExpression.imag`.

        The result is returned as an :meth:`AffineExpression`.
        """
        return AffineExpression(glyphs.imag(self.string), self._shape,
            {vars: coef.imag() for vars, coef in self._coefs.items()})

    # --------------------------------------------------------------------------
    # Additional unary operators.
    # --------------------------------------------------------------------------

    @cached_unary_operator
    def __abs__(self):
        from . import Norm

        return Norm(self)

    # --------------------------------------------------------------------------
    # Constraint-creating operators, and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        from .set import Set

        if relation == operator.__eq__:
            if issubclass(other.clstype, ComplexAffineExpression):
                return ComplexAffineConstraint.make_type(dim=subtype.dim)
        elif relation == operator.__lshift__:
            if issubclass(other.clstype, ComplexAffineExpression):
                return ComplexLMIConstraint.make_type(int(subtype.dim**0.5))
            elif issubclass(other.clstype, Set):
                other_type = ExpressionType(cls, subtype)
                return other.predict(operator.__rshift__, other_type)
        elif relation == operator.__rshift__:
            if issubclass(other.clstype, ComplexAffineExpression):
                return ComplexLMIConstraint.make_type(int(subtype.dim**0.5))

        return NotImplemented

    @convert_operands(sameShape=True)
    @validate_prediction
    @refine_operands()
    def __eq__(self, other):
        if isinstance(other, ComplexAffineExpression):
            return ComplexAffineConstraint(self, other)
        else:
            return NotImplemented

    # Since we define __eq__, __hash__ is not inherited. Do this manually.
    __hash__ = Expression.__hash__

    def _lshift_implementation(self, other):
        if isinstance(other, ComplexAffineExpression):
            return ComplexLMIConstraint(self, Constraint.LE, other)
        else:
            return NotImplemented

    def _rshift_implementation(self, other):
        if isinstance(other, ComplexAffineExpression):
            return ComplexLMIConstraint(self, Constraint.GE, other)
        else:
            return NotImplemented

    # --------------------------------------------------------------------------
    # Interface for PICOS-internal use.
    # --------------------------------------------------------------------------

    def sparse_rows(self, varOffsetMap):
        r"""Yield a sparse list representation of the expression.

        This is similar to :meth:`sparse_matrix_form` (with default arguments)
        but instead of returning :math:`A` and :math:`b` at once, this yields
        for every row of :math:`[A \mid b]`, each representing a scalar entry of
        the expression's vectorization, a triplet containing a list of column
        indices and values of that row of :math:`A` and the entry of :math:`b`.

        :param varOffsetMap:
            Maps variables to column offsets.

        :yields tuple(list, list, float):
            Triples ``(J, V, c)`` where ``J`` contains column indices
            (representing scalar variables), ``V`` contains coefficients for
            each column index, and where ``c`` is a constant term.
        """
        A, b = self.sparse_matrix_form(varOffsetMap, dense_b=True)
        R, C, V = A.T.CCS

        for r in range(len(self)):
            u, v = R[r:r + 2]
            yield list(C[u:v]), list(V[u:v]), b[r]

    def sparse_matrix_form(
            self, varOffsetMap, *, offset=0, padding=0, dense_b=False):
        """Return a representation suited for embedding in constraint matrices.

        This computes a sparse matrix :math:`A` and a sparse column vector
        :math:`b` such that :math:`Ax + b` represents the vectorized expression,
        where :math:`x` is a vertical concatenation of a number of variables,
        including those that appear in the expression. The size and ordering of
        :math:`x` is given through ``varOffsetMap``, which maps PICOS variables
        to their starting position within :math:`x`.

        If the optional parameters ``offset`` and ``padding`` are given, then
        both :math:`A` and :math:`b` are padded with zero rows from above and
        below, respectively.

        This method is used by PICOS internally to assemble constraint matrices.

        :param dict varOffsetMap:
            Maps variables to column offsets.
        :param int offset:
            Number of zero rows to insert at the top of :math:`A` and :math:`b`.
        :param int offset:
            Number of zero rows to insert at the bottom of :math:`A` and
            :math:`b`.
        :param bool dense_b:
            Whether to return :math:`b` as a dense vector. Not compatible with
            nonzero ``offset`` or ``padding``.

        :returns tuple(cvxopt.spmatrix):
            A pair ``(A, b)`` of CVXOPT sparse matrices representing the matrix
            :math:`A` and the column vector :math:`b`.  (If ``dense_b=True``,
            then ``b`` is returned as a dense CVXOPT column vector instead.)
        """
        lin = self._sparse_linear_coefs
        cst = self._constant_coef

        tc = self._typecode

        k = len(self)
        m = offset + k + padding
        n = sum(var.dim for var in varOffsetMap)

        ordered_vars = sorted(varOffsetMap, key=varOffsetMap.__getitem__)

        blocks = []
        for var in ordered_vars:
            if var in lin:
                coef = lin[var]
            else:
                coef = cvxopt.spmatrix([], [], [], size=(k, var.dim), tc=tc)

            blocks.append([coef])

        if blocks:
            A = cvxopt.sparse(blocks, tc=tc)
        else:
            A = cvxopt.spmatrix([], [], [], size=(k, 0), tc=tc)

        b = cvxopt.matrix(cst, tc=tc) if dense_b else cvxopt.sparse(cst, tc=tc)

        if offset or padding:
            if dense_b:
                raise ValueError("Refusing to return a dense vector if a "
                    "nonzero offset or padding is given.")

            A = cvxopt.sparse(
                [
                    cvxopt.spmatrix([], [], [], size=(offset, n), tc=tc),
                    A,
                    cvxopt.spmatrix([], [], [], size=(padding, n), tc=tc)
                ], tc=tc
            )

            b = cvxopt.sparse(
                [
                    cvxopt.spmatrix([], [], [], size=(offset, 1), tc=tc),
                    b,
                    cvxopt.spmatrix([], [], [], size=(padding, 1), tc=tc)
                ], tc=tc
            )

        assert A.size == (m, n)
        assert b.size == (m, 1)

        return A, b

    def scipy_sparse_matrix_form(
            self, varOffsetMap, *, offset=0, padding=0, dense_b=False):
        """Like :meth:`sparse_matrix_form` but returns SciPy types.

        See :meth:`sparse_matrix_form` for details and arguments.

        :returns tuple(scipy.sparse.csc_matrix):
            A pair ``(A, b)`` of SciPy sparse matrices in CSC format
            representing the matrix :math:`A` and the column vector :math:`b`.
            (If ``dense_b=True``, then ``b`` is returned as a 1-D :class:`NumPy
            array <numpy.ndarray>` instead.)

        :raises ModuleNotFoundError:
            If the optional dependency :mod:`scipy` is not installed.
        """
        import scipy.sparse

        lin = self._linear_coefs
        cst = self._constant_coef

        dtype = type(cst[0])

        k = len(self)
        m = offset + k + padding
        n = sum(var.dim for var in varOffsetMap)

        ordered_vars = sorted(varOffsetMap, key=varOffsetMap.__getitem__)

        blocks = []
        for var in ordered_vars:
            if var in lin:
                coef = cvx2csc(lin[var])
            else:
                coef = scipy.sparse.csc_matrix((k, var.dim), dtype=dtype)

            blocks.append(coef)

        if blocks:
            A = scipy.sparse.hstack(blocks, format="csc", dtype=dtype)
        else:
            A = scipy.sparse.csc_matrix((k, 0), dtype=dtype)

        b = numpy.ravel(cvx2np(cst)) if dense_b else cvx2csc(cst)

        if offset or padding:
            if dense_b:
                raise ValueError("Refusing to return a dense vector if a "
                    "nonzero offset or padding is given.")

            A = scipy.sparse.vstack(
                [
                    scipy.sparse.csc_matrix((offset, n), dtype=dtype),
                    A,
                    scipy.sparse.csc_matrix((padding, n), dtype=dtype)
                ], format="csc", dtype=dtype
            )

            b = scipy.sparse.vstack(
                [
                    scipy.sparse.csc_matrix((offset, 1), dtype=dtype),
                    b,
                    scipy.sparse.csc_matrix((padding, 1), dtype=dtype)
                ], format="csc", dtype=dtype
            )

        assert A.shape == (m, n)
        assert b.shape in ((m,), (m, 1))

        return A, b


class AffineExpression(ComplexAffineExpression):
    """A multidimensional real affine expression."""

    # --------------------------------------------------------------------------
    # Method overridings for BiaffineExpression.
    # --------------------------------------------------------------------------

    @property
    def isreal(self):
        """Always true for :class:`AffineExpression` instances."""  # noqa
        return True

    @property
    def real(self):
        """The :class:`AffineExpression` as is."""  # noqa
        return self

    @cached_property
    def imag(self):
        """A zero of same shape as the :class:`AffineExpression`."""  # noqa
        return self._basetype.zero(self._shape)

    @property
    def conj(self):
        """The :class:`AffineExpression` as is."""  # noqa
        return self

    @property
    def H(self):
        """The regular transpose of the :class:`AffineExpression`."""  # noqa
        return self.T

    # --------------------------------------------------------------------------
    # Method overridings for ComplexAffineExpression.
    # --------------------------------------------------------------------------

    @classmethod
    def _get_basetype(cls):
        return AffineExpression

    @classmethod
    def _get_typecode(cls):
        return "d"

    def _get_refined(self):
        return self

    @convert_operands(sameShape=True, allowNone=True)
    def _set_value(self, value):
        if value is None:
            for var in self._linear_coefs:
                var.value = None
            return

        if not isinstance(value, AffineExpression) or not value.constant:
            raise TypeError("Cannot set the value of {} to {}: Not real or not "
                "a constant.".format(repr(self), repr(value)))

        if self.constant:
            raise TypeError("Cannot set the value on a constant expression.")

        y = cvx2np(value._constant_coef)

        A = []
        for var, coef in self._linear_coefs.items():
            A.append(cvx2np(coef))
        assert A

        A = numpy.hstack(A)
        b = y - cvx2np(self._constant_coef)

        try:
            solution, residual, _, _ = numpy.linalg.lstsq(A, b, rcond=None)
        except numpy.linalg.LinAlgError as error:
            raise RuntimeError("Setting a value on {} by means of a least-"
                "squares solution failed.".format(self.string)) from error

        if not numpy.allclose(residual, 0):
            raise ValueError("Setting a value on {} failed: No exact solution "
                "to the associated linear system found.".format(self.string))

        offset = 0
        for var in self._linear_coefs:
            var.internal_value = solution[offset:offset+var.dim]
            offset += var.dim

    # --------------------------------------------------------------------------
    # Additional unary operators.
    # --------------------------------------------------------------------------

    @cached_property
    def exp(self):
        """The exponential function applied to the expression."""  # noqa
        from . import SumExponentials
        return SumExponentials(self)

    @cached_property
    def log(self):
        """The Logarithm of the expression."""  # noqa
        from . import Logarithm
        return Logarithm(self)

    # --------------------------------------------------------------------------
    # Constraint-creating operators, and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        if relation in (operator.__eq__, operator.__le__, operator.__ge__):
            if issubclass(other.clstype, AffineExpression):
                return AffineConstraint.make_type(
                    dim=subtype.dim, eq=(relation is operator.__eq__))
        elif relation == operator.__lshift__:
            if issubclass(other.clstype, AffineExpression):
                return LMIConstraint.make_type(int(subtype.dim**0.5))
            elif issubclass(other.clstype, ComplexAffineExpression):
                return ComplexLMIConstraint.make_type(int(subtype.dim**0.5))
        elif relation == operator.__rshift__:
            if issubclass(other.clstype, AffineExpression):
                return LMIConstraint.make_type(int(subtype.dim**0.5))
            elif issubclass(other.clstype, ComplexAffineExpression):
                return ComplexLMIConstraint.make_type(int(subtype.dim**0.5))

        return NotImplemented

    @convert_operands(sameShape=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        if isinstance(other, AffineExpression):
            return AffineConstraint(self, Constraint.LE, other)
        else:
            return NotImplemented

    @convert_operands(sameShape=True)
    @validate_prediction
    @refine_operands()
    def __ge__(self, other):
        if isinstance(other, AffineExpression):
            return AffineConstraint(self, Constraint.GE, other)
        else:
            return NotImplemented

    @convert_operands(sameShape=True)
    @validate_prediction
    @refine_operands()
    def __eq__(self, other):
        if isinstance(other, AffineExpression):
            return AffineConstraint(self, Constraint.EQ, other)
        else:
            return NotImplemented

    # Since we define __eq__, __hash__ is not inherited. Do this manually.
    __hash__ = Expression.__hash__

    def _lshift_implementation(self, other):
        if isinstance(other, AffineExpression):
            return LMIConstraint(self, Constraint.LE, other)
        elif isinstance(other, ComplexAffineExpression):
            return ComplexLMIConstraint(self, Constraint.LE, other)
        else:
            return NotImplemented

    def _rshift_implementation(self, other):
        if isinstance(other, AffineExpression):
            return LMIConstraint(self, Constraint.GE, other)
        elif isinstance(other, ComplexAffineExpression):
            return ComplexLMIConstraint(self, Constraint.GE, other)
        else:
            return NotImplemented


def Constant(name_or_value, value=None, shape=None):
    """Create a constant PICOS expression.

    Loads the given numeric value as a constant
    :class:`~picos.expressions.ComplexAffineExpression` or
    :class:`~picos.expressions.AffineExpression`, depending on the value.
    Optionally, the value is broadcasted or reshaped according to the shape
    argument.

    :param str name_or_value: Symbolic string description of the constant. If
        :obj:`None` or the empty string, a string will be generated. If this is
        the only positional parameter (i.e.``value`` is not given), then this
        position is used as the value argument instead!
    :param value: The numeric constant to load.

    See :func:`~.data.load_data` for supported data formats and broadcasting and
    reshaping rules.

    :Example:

    >>> from picos import Constant
    >>> Constant(1)
    <1×1 Real Constant: 1>
    >>> Constant(1, shape=(2, 2))
    <2×2 Real Constant: [1]>
    >>> Constant("one", 1)
    <1×1 Real Constant: one>
    >>> Constant("J", 1, (2, 2))
    <2×2 Real Constant: J>
    """
    if value is None:
        value = name_or_value
        name  = None
    else:
        name  = name_or_value

    value, valStr = load_data(value, shape)

    if value.typecode == "z":
        cls = ComplexAffineExpression
    else:
        cls = AffineExpression

    return cls(name if name else valStr, value.size, {(): value})


# --------------------------------------
__all__ = api_end(_API_START, globals())
