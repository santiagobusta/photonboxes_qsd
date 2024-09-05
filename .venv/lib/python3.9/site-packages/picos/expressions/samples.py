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

"""Implements :class:`Samples`."""

import random

import cvxopt

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from .data import cvxopt_hcat, load_data, load_shape
from .exp_affine import ComplexAffineExpression, Constant
from .expression import Expression

_API_START = api_start(globals())
# -------------------------------


class Samples():
    """A collection of data points.

    :Example:

    >>> from picos.expressions import Samples
    >>> # Load the column-major vectorization of six matrices.
    >>> data = [[[1*i, 3*i],
    ...          [2*i, 4*i]] for i in range(1, 7)]
    >>> S = Samples(data)
    >>> S
    <Samples: (6 4-dimensional samples)>
    >>> [S.num, S.dim, S.original_shape]  # Metadata.
    [6, 4, (2, 2)]
    >>> S.matrix  # All samples as the columns of one matrix.
    <4×6 Real Constant: [4×6]>
    >>> print(S.matrix)
    [ 1.00e+00  2.00e+00  3.00e+00  4.00e+00  5.00e+00  6.00e+00]
    [ 2.00e+00  4.00e+00  6.00e+00  8.00e+00  1.00e+01  1.20e+01]
    [ 3.00e+00  6.00e+00  9.00e+00  1.20e+01  1.50e+01  1.80e+01]
    [ 4.00e+00  8.00e+00  1.20e+01  1.60e+01  2.00e+01  2.40e+01]
    >>> print(S[0].T)  # The first sample (transposed for brevity).
    [ 1.00e+00  2.00e+00  3.00e+00  4.00e+00]
    >>> print(S.mean.T)  # The sample mean (transposed for brevity).
    [ 3.50e+00  7.00e+00  1.05e+01  1.40e+01]
    >>> print(S.covariance)  # The sample covariance matrix.
    [ 3.50e+00  7.00e+00  1.05e+01  1.40e+01]
    [ 7.00e+00  1.40e+01  2.10e+01  2.80e+01]
    [ 1.05e+01  2.10e+01  3.15e+01  4.20e+01]
    [ 1.40e+01  2.80e+01  4.20e+01  5.60e+01]
    >>> print(S.original[0])  # The first sample in its original shape.
    [ 1.00e+00  3.00e+00]
    [ 2.00e+00  4.00e+00]
    >>> U = S.select([0, 2, 4])  # Select a subset of samples by indices.
    >>> print(U.matrix)
    [ 1.00e+00  3.00e+00  5.00e+00]
    [ 2.00e+00  6.00e+00  1.00e+01]
    [ 3.00e+00  9.00e+00  1.50e+01]
    [ 4.00e+00  1.20e+01  2.00e+01]
    >>> T, V = S.partition()  # Split into training and validation samples.
    >>> print(T.matrix)
    [ 1.00e+00  2.00e+00  3.00e+00]
    [ 2.00e+00  4.00e+00  6.00e+00]
    [ 3.00e+00  6.00e+00  9.00e+00]
    [ 4.00e+00  8.00e+00  1.20e+01]
    >>> print(V.matrix)
    [ 4.00e+00  5.00e+00  6.00e+00]
    [ 8.00e+00  1.00e+01  1.20e+01]
    [ 1.20e+01  1.50e+01  1.80e+01]
    [ 1.60e+01  2.00e+01  2.40e+01]
    """

    def __new__(cls, samples=None, forced_original_shape=None, **kwargs):
        """Prepare a :class:`Samples` instance."""
        if isinstance(samples, cls):
            if forced_original_shape is not None:
                forced_shape = load_shape(forced_original_shape)
                if forced_shape[0]*forced_shape[1] != samples.dim:
                    raise ValueError("Incompatible forced original shape.")

                if forced_shape == samples.original_shape:
                    # Shapes are consistent, return the existing instance.
                    return samples
                else:
                    # Make a shallow copy and change only the original shape.
                    self = object.__new__(cls)
                    self._cached_cvx_mat = samples._cached_cvx_mat
                    self._cached_cvx_vec = samples._cached_cvx_vec
                    self._cached_pic_mat = samples._cached_pic_mat
                    self._cached_pic_vec = samples._cached_pic_vec
                    self._original_shape = forced_shape
                    return self
            else:
                # Return the existing instance.
                return samples
        else:
            # Return a new instance.
            self = object.__new__(cls)
            self._cached_cvx_mat = None
            self._cached_cvx_vec = None
            self._cached_pic_mat = None
            self._cached_pic_vec = None
            self._original_shape = None
            return self

    def __init__(self, samples, forced_original_shape=None, always_copy=True):
        """Load a number of data points (samples).

        :param samples:
            Any of the following:

            - A tuple or list of constants, each of which denotes a sample
              vector. Matrices are vectorized but their :attr:`original_shape`
              is stored and may be used by PICOS internally.
            - A constant row or column vector whose entries denote scalar
              samples.
            - A constant matrix whose columns denote the samples.
            - Another :class:`Samples` instance. If possible, it is returned as
              is (:class:`Samples` instances are immutable), otherwise a shallow
              copy with the necessary modifications is returned instead.

            In any case, constants may be given as constant numeric data values
            (anything recognized by :func:`~.data.load_data`) or as constant
            PICOS expressions.

        :param forced_original_shape:
            Overwrites :attr:`original_shape` with the given shape.

        :param bool always_copy:
            If this is :obj:`False`, then data that is provided in the form of
            CVXOPT types is not copied but referenced if possible. This can
            speed up instance creation but will introduce inconsistencies if the
            original data is modified. Note that this argument has no impact if
            the ``samples`` argument already is a :class:`Samples` instance; in
            this case data is never copied.
        """
        if isinstance(samples, Samples):
            # Handled in __new__.
            return
        elif isinstance(samples, (tuple, list)):
            if not samples:
                raise ValueError("Need at least one sample.")

            if all(isinstance(s, (int, float, complex)) for s in samples):
                # Efficiently handle a list of scalars.
                self._cached_cvx_mat = load_data(samples)[0].T
            elif all(isinstance(s, ComplexAffineExpression)
                    and s.constant for s in samples):
                if len(set(s.shape for s in samples)) != 1:
                    raise ValueError("Cannot load samples of differing shapes.")

                self._original_shape = samples[0].shape
                self._cached_pic_vec = tuple(s.vec for s in samples)
            else:
                samples = tuple(
                    load_data(s, alwaysCopy=always_copy)[0] for s in samples)

                if len(set(s.size for s in samples)) != 1:
                    raise ValueError("Cannot load samples of differing shapes.")

                self._original_shape = samples[0].size
                self._cached_cvx_vec = tuple(
                    s if s.size[1] == 1 else s[:] for s in samples)
        elif isinstance(samples, Expression):
            samples = samples.refined

            if not isinstance(samples, ComplexAffineExpression):
                raise TypeError("Can only extract samples from a (constant) "
                    "affine expression, not from an instance of {}."
                    .format(type(samples).__name__))

            if not samples.constant:
                raise TypeError("Can only extract samples from a constant "
                    "expression, {} is not constant.".format(samples.string))

            self._cached_pic_mat = samples

            # Treat any vector as a number of scalar samples.
            if self._cached_pic_mat.shape[1] == 1:
                self._cached_pic_mat = self._cached_pic_mat.T
        else:
            self._cached_cvx_mat = load_data(samples, alwaysCopy=always_copy)[0]

            # Treat any vector as a number of scalar samples.
            if self._cached_cvx_mat.size[1] == 1:
                self._cached_cvx_mat = self._cached_cvx_mat.T

        assert any(samples is not None for samples in (
            self._cached_cvx_vec,
            self._cached_pic_vec,
            self._cached_cvx_mat,
            self._cached_pic_mat))

        if forced_original_shape is not None:
            forced_shape = load_shape(forced_original_shape)
            if forced_shape[0]*forced_shape[1] != self.dim:
                raise ValueError("Incompatible forced original shape.")
            self._original_shape = forced_shape

    def __len__(self):
        """Number of samples."""
        return self.num

    def __str__(self):
        return glyphs.parenth(
            "{} {}-dimensional samples".format(self.num, self.dim))

    def __repr__(self):
        return glyphs.repr2("Samples", self.__str__())

    def __getitem__(self, i):
        return self.vectors[i]

    def __iter__(self):
        for vector in self.vectors:
            yield vector

    @property
    def _cvxopt_matrix(self):
        """A CVXOPT dense or sparse matrix whose columns are the samples.

        This cached property is used by PICOS internally as accessing the CVXOPT
        value of a constant PICOS expression would create a copy of the data.

        .. warning::

            :class:`Sample` instances are supposed to be immutable, so you are
            expected not to modify the returned CVXOPT objects.
        """
        if self._cached_cvx_mat is not None:
            pass
        elif self._cached_pic_mat is not None:
            self._cached_cvx_mat = self._cached_pic_mat.value_as_matrix
        elif self._cached_cvx_vec is not None:
            self._cached_cvx_mat = cvxopt_hcat(self._cached_cvx_vec)
        else:
            self._cached_cvx_mat = cvxopt_hcat(
                [s.value_as_matrix for s in self._cached_pic_vec])

        return self._cached_cvx_mat

    @property
    def _cvxopt_vectors(self):
        """A :class:`tuple` containing the samples as CVXOPT column vectors.

        This cached property is used by PICOS internally as accessing the CVXOPT
        value of a constant PICOS expression would create a copy of the data.

        .. warning::

            :class:`Sample` instances are supposed to be immutable, so you are
            expected not to modify the returned CVXOPT objects.
        """
        if self._cached_cvx_vec is not None:
            pass
        elif self._cached_cvx_mat is not None:
            self._cached_cvx_vec = tuple(self._cached_cvx_mat[:, i]
                for i in range(self._cached_cvx_mat.size[1]))
        elif self._cached_pic_vec is not None:
            self._cached_cvx_vec = tuple(
                s.value_as_matrix for s in self._cached_pic_vec)
        else:
            # We need to convert from a PICOS to a CVXOPT matrix, do so in a way
            # that caches the result.
            _ = self._cvxopt_matrix
            assert self._cached_cvx_mat is not None

            self._cached_cvx_vec = tuple(self._cached_cvx_mat[:, i]
                for i in range(self._cached_cvx_mat.size[1]))

        return self._cached_cvx_vec

    @property
    def matrix(self):
        """A matrix whose columns are the samples."""
        if self._cached_pic_mat is not None:
            pass
        else:
            self._cached_pic_mat = Constant(self._cvxopt_matrix)

        return self._cached_pic_mat

    @property
    def vectors(self):
        """A :class:`tuple` containing the samples as column vectors."""
        if self._cached_pic_vec is not None:
            pass
        else:
            self._cached_pic_vec = tuple(
                Constant(s) for s in self._cvxopt_vectors)

        return self._cached_pic_vec

    @cached_property
    def original(self):
        """A :class:`tuple` containing the samples in their original shape."""
        shape = self.original_shape

        if shape[1] == 1:
            return self.vectors
        else:
            return tuple(sample.reshaped(shape) for sample in self)

    @property
    def dim(self):
        """Sample dimension."""
        if self._cached_cvx_mat is not None:
            return self._cached_cvx_mat.size[0]
        elif self._cached_pic_mat is not None:
            return self._cached_pic_mat.shape[0]
        elif self._cached_cvx_vec is not None:
            # NOTE: len() counts nonzero entries for sparse matrices.
            return self._cached_cvx_vec[0].size[0]
        else:
            return len(self._cached_pic_vec[0])

    @property
    def num(self):
        """Number of samples."""
        if self._cached_cvx_mat is not None:
            return self._cached_cvx_mat.size[1]
        elif self._cached_pic_mat is not None:
            return self._cached_pic_mat.shape[1]
        elif self._cached_cvx_vec is not None:
            return len(self._cached_cvx_vec)
        else:
            return len(self._cached_pic_vec)

    @property
    def original_shape(self):
        """Original shape of the samples before vectorization."""
        if self._original_shape is None:
            self._original_shape = (self.dim, 1)

        return self._original_shape

    @cached_property
    def mean(self):
        """The sample mean as a column vector."""
        return Constant(sum(self._cvxopt_vectors) / self.num)

    @cached_property
    def covariance(self):
        """The sample covariance matrix."""
        if self.num == 1:
            return cvxopt.spmatrix([], [], [], (1, 1))

        mu = self.mean.value_as_matrix
        X = self._cvxopt_matrix
        Y = mu*cvxopt.matrix(1, (1, self.num))
        Z = X - Y

        return Constant(Z * Z.T / (self.num - 1))

    def shuffled(self, rng=None):
        """Return a randomly shuffled instance of the samples.

        :param rng:
            A function that generates a random :class:`float` in :math:`[0, 1)`.
            Defaults to whatever :func:`random.shuffle` defaults to.

        :Example:

        >>> from picos.expressions import Samples
        >>> S = Samples(range(6))
        >>> print(S.matrix)
        [ 0.00e+00  1.00e+00  2.00e+00  3.00e+00  4.00e+00  5.00e+00]
        >>> rng = lambda: 0.5  # Fake RNG for reproducibility.
        >>> print(S.shuffled(rng).matrix)
        [ 0.00e+00  5.00e+00  1.00e+00  4.00e+00  2.00e+00  3.00e+00]
        """
        order = list(range(self.num))
        random.shuffle(order, rng)

        S = self.__class__.__new__(self.__class__)

        if self._cached_cvx_mat is not None:
            S._cached_cvx_mat = self._cached_cvx_mat[:, order]

        if self._cached_cvx_vec is not None:
            S._cached_cvx_vec = tuple(self._cached_cvx_vec[i] for i in order)

        if self._cached_pic_mat is not None:
            # NOTE: Rename to a default string for consistency.
            S._cached_pic_mat = self._cached_pic_mat[:, order].renamed(
                glyphs.matrix(glyphs.shape(self._cached_pic_mat.shape)))

        if self._cached_pic_vec is not None:
            S._cached_pic_vec = tuple(self._cached_pic_vec[i] for i in order)

        return S

    def partition(self, after_or_fraction=0.5):
        """Split the samples into two parts.

        :param after_or_fraction:
            Either a fraction strictly between zero and one that denotes the
            relative size of the first partition or an integer that denotes the
            number of samples to put in the first partition.
        :type after_or_fraction:
            int or float
        """
        if isinstance(after_or_fraction, float):
            if after_or_fraction <= 0 or after_or_fraction >= 1:
                raise ValueError(
                    "A partitioning fraction must be strictly between 0 and 1.")

            n = round(self.num * after_or_fraction)
            n = min(n, self.num - 1)
            n = max(1, n)
        else:
            n = int(after_or_fraction)

        if n < 1 or n >= self.num:
            raise ValueError("Partitioning would leave one partition empty.")

        A = self.__class__.__new__(self.__class__)
        B = self.__class__.__new__(self.__class__)

        if self._cached_cvx_mat is not None:
            A._cached_cvx_mat = self._cached_cvx_mat[:, :n]
            B._cached_cvx_mat = self._cached_cvx_mat[:, n:]

        if self._cached_cvx_vec is not None:
            A._cached_cvx_vec = self._cached_cvx_vec[:n]
            B._cached_cvx_vec = self._cached_cvx_vec[n:]

        if self._cached_pic_mat is not None:
            A._cached_pic_mat = self._cached_pic_mat[:, :n]
            B._cached_pic_mat = self._cached_pic_mat[:, n:]

        if self._cached_pic_vec is not None:
            A._cached_pic_vec = self._cached_pic_vec[:n]
            B._cached_pic_vec = self._cached_pic_vec[n:]

        A._original_shape = self._original_shape
        B._original_shape = self._original_shape

        return A, B

    def kfold(self, k):
        r"""Perform :math:`k`-fold cross-validation (without shuffling).

        If random shuffling is desired, write ``S.shuffled().kfold(k)`` where
        ``S`` is your :class:`Samples` instance. To make the shuffling
        reproducible, see :meth:`shuffled`.

        :returns list(tuple):
            A list of :math:`k` training set and validation set pairs.

        .. warning::

            If the number of samples :math:`n` is not a multiple of :math:`k`,
            then the last :math:`n \bmod k` samples will appear in every
            training but in no validation set.

        :Example:

        >>> from picos.expressions import Samples
        >>> n, k = 7, 3
        >>> S = Samples(range(n))
        >>> for i, (T, V) in enumerate(S.kfold(k)):
        ...     print("Partition {}:\nT = {}V = {}"
        ...           .format(i + 1, T.matrix, V.matrix))
        Partition 1:
        T = [ 2.00e+00  3.00e+00  4.00e+00  5.00e+00  6.00e+00]
        V = [ 0.00e+00  1.00e+00]
        <BLANKLINE>
        Partition 2:
        T = [ 0.00e+00  1.00e+00  4.00e+00  5.00e+00  6.00e+00]
        V = [ 2.00e+00  3.00e+00]
        <BLANKLINE>
        Partition 3:
        T = [ 0.00e+00  1.00e+00  2.00e+00  3.00e+00  6.00e+00]
        V = [ 4.00e+00  5.00e+00]
        <BLANKLINE>

        """
        if not isinstance(k, int):
            raise TypeError("k must be an integer.")

        if k < 2:
            raise ValueError("k must be at least two.")

        if k > self.num:
            raise ValueError("k must not exceed the number of samples.")

        n = self.num // k

        assert n >= 1 and n < self.num

        fold = []
        indices = list(range(self.num))

        for i in range(k):
            t = indices[:i*n] + indices[(i+1)*n:]
            v = indices[i*n:(i+1)*n]

            T = self.__class__.__new__(self.__class__)
            V = self.__class__.__new__(self.__class__)

            if self._cached_cvx_mat is not None:
                T._cached_cvx_mat = self._cached_cvx_mat[:, t]
                V._cached_cvx_mat = self._cached_cvx_mat[:, v]

            if self._cached_cvx_vec is not None:
                T._cached_cvx_vec = tuple(self._cached_cvx_vec[i] for i in t)
                V._cached_cvx_vec = tuple(self._cached_cvx_vec[i] for i in v)

            if self._cached_pic_mat is not None:
                T._cached_pic_mat = self._cached_pic_mat[:, t]
                V._cached_pic_mat = self._cached_pic_mat[:, v]

            if self._cached_pic_vec is not None:
                T._cached_pic_vec = tuple(self._cached_pic_vec[i] for i in t)
                V._cached_pic_vec = tuple(self._cached_pic_vec[i] for i in v)

            fold.append((T, V))

        return fold

    def select(self, indices):
        """Return a new :class:`Samples` instance with only selected samples.

        :param indices:
            The indices of the samples to select.
        """
        indices = list(indices)

        S = self.__class__.__new__(self.__class__)

        if self._cached_cvx_mat is not None:
            S._cached_cvx_mat = self._cached_cvx_mat[:, indices]

        if self._cached_cvx_vec is not None:
            S._cached_cvx_vec = tuple(self._cached_cvx_vec[i] for i in indices)

        if self._cached_pic_mat is not None:
            S._cached_pic_mat = self._cached_pic_mat[:, indices]

        if self._cached_pic_vec is not None:
            S._cached_pic_vec = tuple(self._cached_pic_vec[i] for i in indices)

        if len(S) == 0:
            raise ValueError("Empty susbet of samples selected.")

        S._original_shape = self._original_shape
        return S


# --------------------------------------
__all__ = api_end(_API_START, globals())
