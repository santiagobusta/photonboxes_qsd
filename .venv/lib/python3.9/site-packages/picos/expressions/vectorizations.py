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

r"""Implement special matrix vectorization formats.

These formats are used to efficiently store structured mutable types such as
symmetric matrix variables in the form of real vectors.
"""

import random
from abc import ABC, abstractmethod
from copy import copy

import cvxopt

from .. import glyphs, settings
from ..apidoc import api_end, api_start
from .data import cvxopt_equals, cvxopt_hcat, cvxopt_vcat, load_shape

_API_START = api_start(globals())
# -------------------------------


#: Number of instances to cache per vectorization format.
CACHE_SIZE = 100

#: Number of cached instances to drop at random when the cache is full.
CACHE_BULK_REMOVE = CACHE_SIZE // 4


class BaseVectorization(ABC, object):
    """Abstract base class for special matrix vectorization formats.

    Subclass instances are cached: If multiple instances of the same
    vectorization format and concerning matrices of the same shape are requested
    successively, then the instance created to serve the first request is
    retrieved from a cache on successive requests. The module attributes
    :data:`CACHE_SIZE` and :data:`CACHE_BULK_REMOVE` control the size of the
    cache for each vectorization format.

    .. warning::

        Due to how caching is implemented, derived classes may not inherit from
        each other but only from :class:`BaseVectorization` directly!
    """

    def __new__(cls, shape):
        """Lookup or create a vectorization format for a fixed matrix shape."""
        shape = load_shape(shape, squareMatrix=cls._square_input())

        if not hasattr(cls, "_cache"):
            cls._cache = {}

        if shape not in cls._cache:
            if len(cls._cache) >= CACHE_SIZE:
                for remove in random.sample(
                        list(cls._cache.keys()), CACHE_BULK_REMOVE):
                    cls._cache.pop(remove)

            cls._cache[shape] = object.__new__(cls)

        return cls._cache[shape]

    def __init__(self, shape):
        """Initialize a vectorization format for a fixed matrix shape."""
        self._shape = load_shape(shape, squareMatrix=self._square_input())
        self._special2full = self._make_special_to_full()
        self._full2special = self._make_full_to_special()

    def __len__(self):
        return self._shape[0] * self._shape[1]

    @property
    def shape(self):
        """The shape of matrices being vectorized."""
        return self._shape

    @property
    def dim(self):
        """The length of the vectorization.

        This corresponds to the dimension of a matrix mutable being vectorized.
        """
        return self._special2full.size[1]

    @classmethod
    @abstractmethod
    def _square_input(cls):
        """Whether input matrices must be square."""
        pass

    @abstractmethod
    def _validate_matrix(self, matrix):
        """Raise an exception if the given matrix cannot be vectorized."""
        if not isinstance(matrix, (cvxopt.matrix, cvxopt.spmatrix)):
            raise TypeError("May only vectorize CVXOPT matrix types.")

        if matrix.size != self._shape:
            raise TypeError("Cannot vectorize a matrix of shape {} according "
                "to a vectorization recipe for {} matrices."
                .format(glyphs.shape(matrix.size), glyphs.shape(self._shape)))

    def _ensure_real(self, matrix):
        """Raise a :exc:`TypeError` if the given matrix is not real."""
        if matrix.typecode == "z":
            raise TypeError(
                "The vectorization format does not support complex input.")

    def _validate_vector(self, vector):
        """Raise an exception if the given vector cannot be devectorized."""
        if not isinstance(vector, (cvxopt.matrix, cvxopt.spmatrix)):
            raise TypeError("May only devectorize CVXOPT matrix types.")

        if vector.typecode == "z":
            raise TypeError("Cannot devectorize a complex vector: All "
                "vectorizations are expected to be real.")

        if vector.size != (self.dim, 1):
            raise TypeError(
                "Invalid shape of vectorized data: Expected {} but got {}."
                .format(glyphs.shape((self.dim, 1)), glyphs.shape(vector.size)))

    @abstractmethod
    def _make_special_to_full(self):
        """Return a mapping from the special to the full vectorization."""
        pass

    @abstractmethod
    def _make_full_to_special(self):
        """Return a mapping from the full to the special vectorization.

        Returns :obj:`None` if the input matrix is complex. Then,
        :meth:`vectorize` needs to be overridden.
        """
        pass

    @property
    def identity(self):
        """A linear mapping from the special to the full vectorization.

        The term *identity* comes from the fact that these matrices are used
        as the coefficients that map the internal (vectorized) representation of
        a :class:`~.mutable.Mutable` object to the
        :class:`~.exp_biaffine.BiaffineExpression` instance that represents the
        mutable in algebraic operations.
        """
        return self._special2full

    def vectorize(self, matrix):
        """Given a matrix, return its special vectorization.

        :raises TypeError: If the input isn't a CVXOPT matrix or does not have
            the expected numeric type or shape.
        :raises ValueError: If the matrix does not have the expected structure.
        """
        self._validate_matrix(matrix)
        return self._full2special*matrix[:]

    def devectorize(self, vector):
        """Given a special vectorization, return the corresponding matrix.

        :raises TypeError: If the input isn't a CVXOPT column vector or does not
            have the expected numeric type or length.
        """
        self._validate_vector(vector)
        M = self._special2full*vector
        M.size = self._shape
        return M


class FullVectorization(BaseVectorization):
    """A basic column-major matrix vectorization."""

    @classmethod
    def _square_input(cls):
        return False

    def _validate_matrix(self, matrix):
        BaseVectorization._validate_matrix(self, matrix)
        self._ensure_real(matrix)

    def _make_full_to_special(self):
        n = len(self)
        return cvxopt.spmatrix([1]*n, range(n), range(n), tc="d")

    def vectorize(self, matrix):
        """Override :meth:`BaseVectorization.vectorize` for speed reasons."""
        self._validate_matrix(matrix)

        if matrix.typecode == "d":
            return matrix[:]
        else:
            assert matrix.typecode == "i"
            return self._full2special*matrix[:]

    def _make_special_to_full(self):
        # Not actually used, see devectorize.
        return self._make_full_to_special()

    def devectorize(self, vector):
        """Override :meth:`BaseVectorization.devectorize` for speed reasons."""
        self._validate_vector(vector)
        M = copy(vector)
        M.size = self._shape
        return M


class ComplexVectorization(BaseVectorization):
    """An isometric vectorization that stacks real and imaginary parts."""

    @classmethod
    def _square_input(cls):
        return False

    def _validate_matrix(self, matrix):
        BaseVectorization._validate_matrix(self, matrix)

    def _make_full_to_special(self):
        # No such matrix exists as the full vectorization is complex.
        return None

    def vectorize(self, matrix):
        """Override :meth:`BaseVectorization.vectorize`.

        This is necessary because extracting the real and the imaginary part
        cannot be done with a linear transformation matrix as is done for other
        vectorization formats.
        """
        self._validate_matrix(matrix)
        fullVectorization = matrix[:]
        return cvxopt_vcat([fullVectorization.real(), fullVectorization.imag()])

    def _make_special_to_full(self):
        cRank = len(self)  # Rank on the complex field.
        rRank = 2*cRank    # Rank on the real field.
        return cvxopt.spmatrix([1]*cRank + [1j]*cRank,
            list(range(cRank))*2, range(rRank), tc="z")


class SymmetricVectorization(BaseVectorization):
    """An isometric symmetric matrix vectorization.

    See [svec]_ for the precise vectorization used.

    .. [svec] Dattorro, J. (2018). Isomorphism of symmetric matrix subspace.
        In *Convex Optimization & Euclidean Distance Geometry (2nd ed.)*
        (pp. 47f.). California, Meboo Publishing USA. Retrieved from
        `<https://meboo.convexoptimization.com/Meboo.html>`_.
    """

    @classmethod
    def _square_input(cls):
        return True

    def _validate_matrix(self, matrix):
        BaseVectorization._validate_matrix(self, matrix)
        self._ensure_real(matrix)

        if not cvxopt_equals(matrix, matrix.T,
                relTol=settings.RELATIVE_HERMITIANNESS_TOLERANCE):
            raise ValueError("The given matrix is not numerically symmetric.")

    def _make_full_to_special(self):
        n = self._shape[0]
        m = n*(n + 1) // 2
        I, J, V = [], [], []
        for i in range(m):
            c = int((1 + 8*i)**0.5 - 1) // 2
            r = i - c*(c + 1) // 2

            # Version 1: Average elements in lower and upper triangular part.
            #            This could work around noisy input data.
            if c == r:
                I.append(i)
                J.append(c*n + r)
                V.append(1)
            else:
                I.extend([i, i])
                J.extend([c*n + r, r*n + c])
                V.extend([2**0.5 / 2, 2**0.5 / 2])

            # Version 2: Ignore elements below the main diagonal.
            #            This is the faster approach.
            # I.append(i)
            # J.append(c*n + r)
            # V.append(1 if c == r else 2**0.5)

        return cvxopt.spmatrix(V, I, J, (m, n**2), tc="d")

    def _make_special_to_full(self):
        n = self._shape[0]
        I, J, V = range(n**2), [], []
        for i in I:
            c, r = divmod(i, n)
            if c < r:  # Entries below the diagonal are infered.
                c, r = r, c
            J.append(c*(c + 1) // 2 + r)
            V.append(1 if c == r else 1 / 2**0.5)
        return cvxopt.spmatrix(V, I, J, (n**2, n*(n + 1) // 2), tc="d")


class SkewSymmetricVectorization(BaseVectorization):
    """An isometric skew-symmetric matrix vectorization."""

    @classmethod
    def _square_input(cls):
        return True

    def _validate_matrix(self, matrix):
        BaseVectorization._validate_matrix(self, matrix)
        self._ensure_real(matrix)

        # NOTE: Use hermitianess tolerance.
        if not cvxopt_equals(matrix, -matrix.T,
                relTol=settings.RELATIVE_HERMITIANNESS_TOLERANCE):
            raise ValueError(
                "The given matrix is not numerically skew-symmetric.")

    def _make_full_to_special(self):
        n = self._shape[0]
        m = n*(n - 1) // 2
        I, J, V = [], [], []
        for i in range(m):
            c = int((1 + 8*i)**0.5 + 1) // 2
            r = i - c*(c - 1) // 2
            I.append(i)
            J.append(c*n + r)
            V.append(2**0.5)
        return cvxopt.spmatrix(V, I, J, (m, n**2), tc="d")

    def _make_special_to_full(self):
        n = self._shape[0]
        I, J, V = [], [], []
        for i in range(n**2):
            c, r = divmod(i, n)
            if c == r:  # Entries on the diagonal are zero.
                continue
            elif c < r:  # Entries below the diagonal are infered.
                I.append(i)
                J.append(r*(r - 1) // 2 + c)
                V.append(-1 / 2**0.5)
            else:
                I.append(i)
                J.append(c*(c - 1) // 2 + r)
                V.append(1 / 2**0.5)
        return cvxopt.spmatrix(V, I, J, (n**2, n*(n - 1) // 2), tc="d")


class HermitianVectorization(BaseVectorization):
    r"""An isometric hermitian matrix vectorization.

    The vectorization is isometric with respect to the Hermitian inner product
    :math:`\langle A, B \rangle = \operatorname{tr}(B^H A)` on the matrices and
    the real dot product on their vectorizations.
    """

    def __init__(self, shape):
        """Initialize a vectorization format for hermitian matrices.

        Uses :class:`SymmetricVectorization` (for the real part) and
        :class:`SkewSymmetricVectorization` (for the imaginary part) internally.
        """
        self._sym = SymmetricVectorization(shape)
        self._skw = SkewSymmetricVectorization(shape)
        super(HermitianVectorization, self).__init__(shape)

    @classmethod
    def _square_input(cls):
        return True

    def _validate_matrix(self, matrix):
        BaseVectorization._validate_matrix(self, matrix)

        if not cvxopt_equals(matrix, matrix.H,
                relTol=settings.RELATIVE_HERMITIANNESS_TOLERANCE):
            raise ValueError("The given matrix is not numerically hermitian.")

    def _make_full_to_special(self):
        # No such matrix exists as the full vectorization is complex.
        return None

    def vectorize(self, matrix):
        """Override :meth:`BaseVectorization.vectorize`.

        This is necessary because extracting the real and the imaginary part
        cannot be done with a linear transformation matrix as is done for other
        vectorization formats.
        """
        self._validate_matrix(matrix)
        fullVectorization = matrix[:]
        return cvxopt_vcat([
            self._sym._full2special*fullVectorization.real(),
            self._skw._full2special*fullVectorization.imag()])

    def _make_special_to_full(self):
        A = cvxopt_hcat([
            self._sym._special2full,
            cvxopt.spmatrix([], [], [], self._skw._special2full.size)])
        B = cvxopt_hcat([
            cvxopt.spmatrix([], [], [], self._sym._special2full.size),
            self._skw._special2full])
        return A + 1j*B


class LowerTriangularVectorization(BaseVectorization):
    """An isometric lower triangular matrix vectorization."""

    @classmethod
    def _square_input(cls):
        return True

    def _validate_matrix(self, matrix):
        BaseVectorization._validate_matrix(self, matrix)
        self._ensure_real(matrix)

        n = self._shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j]:
                    raise ValueError(
                        "The given matrix is not lower triangular.")

    def _make_full_to_special(self):
        n = self._shape[0]
        m = n*(n + 1) // 2
        I, J, V = [], [], []
        for i in range(m):
            c = int((1 + 8*i)**0.5 - 1) // 2
            r = i - c*(c + 1) // 2
            I.append(i)
            J.append(r*n + c)
            V.append(1)
        return cvxopt.spmatrix(V, I, J, (m, n**2), tc="d")

    def _make_special_to_full(self):
        n = self._shape[0]
        m = n*(n + 1) // 2
        I, J, V = [], [], []
        for j in range(m):
            c = int((1 + 8*j)**0.5 - 1) // 2
            r = j - c*(c + 1) // 2
            I.append(r*n + c)
            J.append(j)
            V.append(1)
        return cvxopt.spmatrix(V, I, J, (n**2, m), tc="d")


class UpperTriangularVectorization(BaseVectorization):
    """An isometric upper triangular matrix vectorization."""

    @classmethod
    def _square_input(cls):
        return True

    def _validate_matrix(self, matrix):
        BaseVectorization._validate_matrix(self, matrix)
        self._ensure_real(matrix)

        n = self._shape[0]
        for i in range(n):
            for j in range(i):
                if matrix[i, j]:
                    raise ValueError(
                        "The given matrix is not upper triangular.")

    def _make_full_to_special(self):
        n = self._shape[0]
        m = n*(n + 1) // 2
        I, J, V = [], [], []
        for i in range(m):
            c = int((1 + 8*i)**0.5 - 1) // 2
            r = i - c*(c + 1) // 2
            I.append(i)
            J.append(c*n + r)
            V.append(1)
        return cvxopt.spmatrix(V, I, J, (m, n**2), tc="d")

    def _make_special_to_full(self):
        n = self._shape[0]
        m = n*(n + 1) // 2
        I, J, V = [], [], []
        for j in range(m):
            c = int((1 + 8*j)**0.5 - 1) // 2
            r = j - c*(c + 1) // 2
            I.append(c*n + r)
            J.append(j)
            V.append(1)
        return cvxopt.spmatrix(V, I, J, (n**2, m), tc="d")


# --------------------------------------
__all__ = api_end(_API_START, globals())
