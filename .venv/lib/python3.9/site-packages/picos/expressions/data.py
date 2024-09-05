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

"""Functions to load and work with raw numeric data."""

import functools
import inspect
import math
import re
import sys
from fractions import Fraction
from functools import lru_cache

import cvxopt
import numpy

from .. import glyphs, settings
from ..apidoc import api_end, api_start

_API_START = api_start(globals())
# -------------------------------


#: Maximum entrywise absolute deviation allowed for numeric equality checks.
TOLERANCE = 1e-6


def load_shape(shape, squareMatrix=False, wildcards=False):
    """Parse the argument as a matrix shape.

    PICOS uses this function whenever you supply a shape parameter to a method.

    A scalar argument is treated as the length of a column-vector. If the shape
    contains :obj:`None`, it is treated as a wildcard (any dimensionality).

    :param bool squareMatrix: If :obj:`True`, a scalar argument is treated as
        the side/diagonal length of a square matrix, and any other argument is
        validated to be square. If :obj:`False`, a scalar argument is treated
        as the length of a column vector.
    :param bool wildcards: Whether the wildcard token :obj:`None` is allowed.
    """
    if shape is None:
        shape = (None, None)
    elif isinstance(shape, int):
        if squareMatrix:
            shape = (shape, shape)
        else:
            shape = (shape, 1)
    elif not isinstance(shape, tuple) and not isinstance(shape, list):
        raise TypeError("Shapes must be given as None, int, tuple or list.")
    elif len(shape) == 1:
        shape = (shape[0], 1)
    elif len(shape) == 0:
        shape = (1, 1)
    elif len(shape) != 2:
        raise TypeError("Shapes must be two-dimensional.")

    shape = (
        None if shape[0] is None else int(shape[0]),
        None if shape[1] is None else int(shape[1]))

    if not wildcards and None in shape:
        raise ValueError("Invalid shape (wildcards not allowed): {}."
            .format(glyphs.shape(shape)))

    if 0 in shape:
        raise ValueError("Invalid shape (zero-dimensional axis): {}."
            .format(glyphs.shape(shape)))

    if squareMatrix and shape[0] != shape[1]:
        raise ValueError("Invalid shape for a square matrix: {}."
            .format(glyphs.shape(shape)))

    return shape


def blend_shapes(baseShape, defaultShape):
    """Replace wildcards in one shape with entries of the other.

    :param baseShape: Primary shape, usually with wildcards.
    :type baseShape: tuple(int or None)
    :param defaultShape: Secondary shape with fallback entries.
    :type defaultShape: tuple(int or None)
    """
    return (
        defaultShape[0] if baseShape[0] is None else baseShape[0],
        defaultShape[1] if baseShape[1] is None else baseShape[1])


def should_be_sparse(shape, numNonZero):
    """Decide whether a matrix is considered sparse.

    :param tuple(int) shape: The shape of the matrix in question.
    :param int numNonZero: Number of non-zero elements of the matrix.
    """
    n, m = shape
    l = (n*m)**0.5
    return numNonZero < l * math.log(l)


_LOAD_DATA_REGEX = re.compile("^{}{}{}$".format(
    r"([\-0-9.+j]+)?",  # Leading coefficient.
    "(" + "|".join((  # Matrix type:
        r"e_[0-9]+(?:,[0-9]+)?",  # Single nonzero element.
        r"\|[\-0-9.+j]+\|",  # All equal elements.
        "I",  # Identity matrix.
    )) + ")",
    r"(\([0-9]+(?:,[0-9]+)?\))?"  # Fixed shape.
))


def is_scipy_spmat(value):
    """Report whether value is a SciPy sparse matrix without importing scipy."""
    return hasattr(value, "__module__") \
        and value.__module__.startswith("scipy.sparse")


def load_data(value, shape=None, typecode=None, sparse=None,
        alwaysCopy=True, legacy=False):
    r"""Load a constant numeric data value as a CVXOPT (sparse) matrix.

    As a user, you never need to call this manually, but you should be aware
    that PICOS uses this function on any raw data you supply as an operand when
    working with PICOS expressions. For instance, you can just add a NumPy
    matrix or an algebraic string such as ``"I"`` to such an expression without
    worrying about any conversion.

    :Supported data types:

        - A NumPy matrix: :class:`numpy.ndarray` or :class:`numpy.matrix`.
        - A SciPy sparse matrix: All from :mod:`scipy.sparse`.
        - A CVXOPT matrix: :obj:`cvxopt.matrix` or :obj:`cvxopt.spmatrix`.
        - A constant PICOS expression: :class:`~.exp_affine.AffineExpression` or
          :class:`~.exp_affine.ComplexAffineExpression`.
        - A Python scalar: :class:`int`, :class:`float` or :class:`complex`.
        - A flat :class:`tuple` or :class:`list` containing scalars or a
          :class:`range`, all representing a column vector.
        - A nested :class:`tuple` or :class:`list` containing scalars. The outer
          level represents rows and the inner level represents the rows'
          entries. Allows you to define a :math:`2 \times 3` matrix like this:

          .. code-block:: python

              A = [[1, 2, 3],
                   [4, 5, 6]]

        - A verbatim string description, with rows separated by newline and
          columns separated by whitespace. The same :math:`2 \times 3` matrix as
          above:

          .. code-block:: python

              A = '''1 2 3
                     4 5 6'''

        - An algebraic string description:

          .. list-table::
            :widths: 1 99

            * - ``"|a|"``
              - A matrix with all entries equal to :math:`a`.
            * - ``"|a|(m,n)"``
              - A :math:`m \times n` matrix with all entries equal to :math:`a`.
            * - ``"e_i,j(m,n)"``
              - A :math:`m \times n` matrix with a :math:`1` at :math:`(i,j)`,
                indexed from :math:`(1,1)`` to :math:`(m,n)``.
            * - ``"e_i(m,n)"``
              - A :math:`m \times n` matrix with a single :math:`1` on the
                :math:`i`-th coordinate, indexed from :math:`1` in column-major
                order.
            * - ``"I"``
              - The identity matrix.
            * - ``"I(n)"``
              - The :math:`n \times n` identiy matrix.
            * - ``"a…"``
              - The matrix given by ``"…"`` but multiplied by :math:`a`.

        Different matrix operations such as addition or multiplication have
        different requirements with respect to the operands' shapes. The shape
        of any PICOS expression involved will always be maintained. But when an
        operation involves both a PICOS expression and raw data, then PICOS will
        try to broadcast or reshape the raw data such that the operation can be
        performed.

    :Broadcasting and reshaping rules:

        - An input vector without a second axis (for instance a non-nested
          :class:`tuple` or :class:`list` or a :class:`range`) is interpreted as
          a row vector if the target shape is of the form ``(None, n)`` with
          :math:`n > 1`, otherwise it is interpreted as a column vector.
        - If the target shape is :obj:`None` or ``(None, None)``, then the
          input's shape is maintained.
        - If the target shape contains :obj:`None` exactly once, that occurence
          is replaced by the respective dimensionality of the input data shape.
        - A scalar is copied (broadcasted) to fill the target shape.
        - A column (row) vector is copied horizontally (vertically) if its
          length matches the target row (column) count.
        - Reshaping from matrix to vector: A matrix is vectorized in
          column-major (row-major) order if the target shape is a column (row)
          vector whose length matches the number of matrix entries.
        - Reshaping from vector to matrix: A column (row) vector is treated as
          the column-major (row-major) vectorization of the target matrix if the
          former's length matches the number of the latter's entries.
        - All other combinations of input and target shape raise an exception.
        - When an algebraic string description specifies no shape, then the
          shape argument must be supplied. When both the string and the shape
          argument specify a shape, then they must be consistent (no
          broadcasting or reshaping is applied in this case).

    :param shape: The shape of the resulting matrix. If the input data is of
        another shape, broadcasting or reshaping is used if possible, otherwise
        an exception is raised. An integer is treated as the length of a column
        vector. If this is :obj:`None`, then the target's shape is the input's.
        If only the target number of rows or columns is :obj:`None`, then only
        that quantity is chosen according to the input.
    :type shape: int or tuple or list or None

    :param str typecode: The numeric type of the resulting matrix. Either
        ``'d'`` (float), ``'z'`` (complex) or ``'i'`` (integer). If the input
        data is not already of this type, then it will be converted if possible.
        If this is not possible, then an exception is raised. If this is
        :obj:`None`, then the output type is chosen based on the input data.

    :param sparse: If :obj:`True`, a sparse matrix is returned. If :obj:`False`,
        a dense matrix is returned. If :obj:`None`, it depends on the sparsity
        pattern of the input data. If the typecode argument is ``'i'``, then
        a value of :obj:`True` is not allowed and the returned matrix is dense.
    :type sparse: bool or None

    :param bool alwaysCopy: If :obj:`True`, then a copy of the input data is
        returned even if it already equals the output data. If :obj:`False`,
        the input value can be returned as such if it is already a CVXOPT matrix
        of the desired shape, typecode, and sparsity.

    :param bool legacy: Be compatible with the old ``retrieve_matrix`` function.
        In particular, if the target shape contains :obj:`None` exactly once and
        the input data is scalar, treat this as a matrix multiplication case and
        return the scalar times an identity matrix of appropriate size.

    :returns: A :class:`tuple` whose first entry is the loaded matrix and whose
        second argument is a short string for representing the data within
        algebraic expression strings.

    :Example:

    >>> from picos.expressions.data import load_data
    >>> # Data as (nested) list:
    >>> load_data([1,2,3])
    (<3x1 matrix, tc='i'>, '[3×1]')
    >>> load_data([[1,2,3]])
    (<1x3 matrix, tc='i'>, '[1×3]')
    >>> A = [[1,2,3],
    ...      [4,5,6]]
    >>> load_data(A)
    (<2x3 matrix, tc='i'>, '[2×3]')
    >>> # Data as string:
    >>> value, string = load_data('e_14(7,2)')
    >>> print(string)
    [7×2:e_7,2]
    >>> print(value) #doctest: +NORMALIZE_WHITESPACE
    [   0        0       ]
    [   0        0       ]
    [   0        0       ]
    [   0        0       ]
    [   0        0       ]
    [   0        0       ]
    [   0        1.00e+00]
    >>> load_data('5.3I', (2,2))
    (<2x2 sparse matrix, tc='d', nnz=2>, '5.3·I')
    """
    from .exp_affine import ComplexAffineExpression
    from .expression import Expression

    def load_sparse(V, I, J, shape, typecode):
        """Create a CVXOPT sparse matrix."""
        # HACK: Work around CVXOPT not supporting integer sparse matrices:
        #       Create a real sparse matrix for now (will be converted later).
        #       Note that users may not request both sparsity and integrality.
        typecode = "d" if typecode == "i" else typecode

        try:
            if typecode:
                return cvxopt.spmatrix(V, I, J, shape, typecode)
            else:
                return cvxopt.spmatrix(V, I, J, shape)
        except TypeError as error:
            # Attempt to convert complex typed but real valued input to a real
            # typed output matrix.
            if typecode == "d":
                realV = [x.real for x in V]
                if realV == V:
                    try:
                        return cvxopt.spmatrix(realV, I, J, shape, typecode)
                    except Exception:
                        pass

            raise TypeError(
                "Failed to create a CVXOPT sparse matrix of shape {} and type "
                "'{}'.".format(glyphs.shape(shape), typecode)) from error

    def load_dense(value, shape, typecode):
        """Create a CVXOPT dense matrix."""
        try:
            if typecode:
                return cvxopt.matrix(value, shape, typecode)
            else:
                return cvxopt.matrix(value, shape)
        except TypeError as error:
            # Attempt to convert complex (real) typed but real/integer (integer)
            # valued input to a real/integer (integer) typed output matrix.
            if typecode in "id":
                try:
                    complexValue = cvxopt.matrix(value, shape, "z")
                except Exception:
                    pass
                else:
                    if not any(complexValue.imag()):
                        if typecode == "d":
                            return complexValue.real()
                        else:
                            realData = list(complexValue.real())
                            intData  = [int(x) for x in realData]
                            if intData == realData:
                                try:
                                    return cvxopt.matrix(intData, shape, "i")
                                except Exception:
                                    pass

            raise TypeError(
                "Failed to create a CVXOPT dense matrix of shape {} and type "
                "'{}'.".format(glyphs.shape(shape), typecode)) from error

    def simple_vector_as_row(shape):
        """Whether a single-axis vector should be a row or column vector."""
        return shape[0] is None and shape[1] is not None and shape[1] > 1

    def broadcast_error():
        """Raise a broadcasting related :class:TypeError."""
        raise TypeError("Cannot broadcast or reshape from {} to {}{}."
            .format(glyphs.shape(inShape), glyphs.shape(shape), " read as {}"
                .format(glyphs.shape(outShape)) if shape != outShape else ""))

    def scalar(x, typecode=None):
        """Load a scalar as either a complex, float, or int."""
        x = complex(x)

        if typecode == "z":
            return x  # We are content with complex.
        else:
            if not x.imag:
                x = x.real

                if typecode == "d":
                    return x  # We are content with real.
                else:
                    if int(x) == x:
                        return int(x)
                    elif not typecode:
                        return x  # We are not strict and real is the best.
            elif not typecode:
                return x  # We are not strict and complex is the best.

        raise TypeError(
            "Cannot cast {} according to typecode '{}'.".format(x, typecode))

    # Normalize the shape argument to a two-dimensional tuple of int or None,
    # where None means any dimensionality.
    shape = load_shape(shape, wildcards=True)

    # Validate the typecode argument.
    if typecode is not None and typecode not in "dzi":
        raise ValueError("Typecode argument not a valid CVXOPT typecode: {}."
            .format(typecode))

    # Validate the sparsity argument.
    if sparse not in (None, True, False):
        raise ValueError("Sparsity argument must be True, False or None.")

    # CVXOPT sparse matrices may not be integer typed.
    if typecode == "i":
        if sparse:
            raise TypeError("Sparse integer matrices are not implemented with "
                "CVXOPT, which PICOS uses as its matrix backed.")
        else:
            sparse = False

    # Conversions must retrieve the input shape for further processing.
    inShape = None

    # Allow conversions to specify their own string description.
    string = None

    # Convert from range to list.
    if isinstance(value, range):
        value = list(value)

    # Try to refine a PICOS expression to a constant affine expression.
    if isinstance(value, Expression):
        value = value.refined

    # Convert from a SciPy sparse to a CVXOPT sparse matrix.
    if is_scipy_spmat(value):
        value = sp2cvx(value)
        alwaysCopy = False

    # Convert the data to a CVXOPT (sparse) matrix of proper shape and type.
    if isinstance(value, (int, float, complex, numpy.number)):
        # Convert NumPy numbers to Python ones.
        if isinstance(value, numpy.number):
            if isinstance(value, numpy.integer):
                value = int(value)
            elif isinstance(value, numpy.floating):
                value = float(value)
            elif isinstance(value, numpy.complexfloating):
                value = complex(value)
            else:
                assert False, "Unexpected NumPy numeric type {}.".format(
                    type(value).__name__)

        # CVXOPT is limited by the system's word length.
        if isinstance(value, int):
            if value > sys.maxsize or value < -sys.maxsize - 1:
                raise ValueError("The number {} is too large to be loaded by "
                    "PICOS.".format(value))

        inShape  = (1, 1)
        outShape = blend_shapes(shape, inShape)

        string = glyphs.scalar(value)
        if value and outShape != (1, 1):  # Don't use the matrix glyph on zero.
            string = glyphs.matrix(string)

        if not value and sparse is not False:
            value = load_sparse([], [], [], outShape, typecode)
        else:
            value = load_dense(value, outShape, typecode)

            if sparse:
                value = cvxopt.sparse(value)
    elif isinstance(value, cvxopt.matrix):
        # NOTE: Since the input is already a CVXOPT data type, it is possible
        #       that it can be returned without modification. The 'alwaysCopy'
        #       parameter, if set to True, prevents this. As soon as a copy is
        #       made during processing of the input we set 'alwaysCopy' to False
        #       to prevent an unnecessary second copy of the processed input to
        #       be made later on.

        # If sparse is requested, let CVXOPT handle the transformation first.
        if sparse:
            value = cvxopt.sparse(value)
            return load_data(value, shape, typecode, sparse, False, legacy)

        # Refine the output shape.
        inShape  = value.size
        outShape = blend_shapes(shape, inShape)

        # Define shorthands.
        inLength  = inShape[0]  * inShape[1]
        outLength = outShape[0] * outShape[1]

        # If the input is stored as complex and no complex output is explicitly
        # requested, try to remove an imaginary part of zero.
        if value.typecode == "z" and typecode != "z" and not any(value.imag()):
            value = value.real()
            alwaysCopy = False

        # Broadcast or reshape the data if necessary and possible.
        reshaped = True
        if inShape == outShape:
            reshaped = False
        elif inShape == (1, 1):
            # Broadcast a scalar.
            value = load_dense(value[0], outShape, typecode)
        elif inShape == (outShape[0], 1):
            # Copy columns horizontally.
            value = load_dense(list(value)*outShape[1], outShape, typecode)
        elif inShape == (1, outShape[1]):
            # Copy rows vertically.
            value = load_dense(
                list(value)*outShape[0], (outShape[1], outShape[0]), typecode).T
        elif (inLength, 1) == outShape:
            # Vectorize in column-major order.
            if not typecode or value.typecode == typecode:
                # Quick method.
                value = value[:]
            else:
                # With typecode change.
                value = load_dense(value, outShape, typecode)
        elif (1, inLength) == outShape:
            # Vectorize in row-major order.
            if not typecode or value.typecode == typecode:
                # Quick method.
                value = value.T[:].T
            else:
                # With typecode change.
                value = load_dense(value.T, outShape, typecode)
        elif (outLength, 1) == inShape:
            # Devectorize in column-major order.
            value = load_dense(value, outShape, typecode)
        elif (1, outLength) == inShape:
            # Devectorize in row-major order.
            value = load_dense(value, (outShape[1], outShape[0]), typecode).T
        else:
            broadcast_error()
        if reshaped:
            alwaysCopy = False

        # The data now has the desired shape.
        assert value.size == outShape

        # Ensure the proper typecode.
        if typecode and value.typecode != typecode:
            value = load_dense(value, outShape, typecode)
            alwaysCopy = False

        # Copy the data if requested and not already a copy.
        if alwaysCopy:
            value = load_dense(value, outShape, typecode)
    elif isinstance(value, cvxopt.spmatrix):
        # NOTE: See case above.

        # Refine the output shape.
        inShape  = value.size
        outShape = blend_shapes(shape, inShape)

        # Define shorthands.
        inLength  = inShape[0]  * inShape[1]
        outLength = outShape[0] * outShape[1]

        # If the input is stored as complex and no complex output is explicitly
        # requested, try to remove an imaginary part of zero.
        if value.typecode == "z" and typecode != "z" and not any(value.imag()):
            value = value.real()
            alwaysCopy = False

        # Broadcast or reshape the data if necessary and possible.
        reshaped = True
        if inShape == outShape:
            reshaped = False
        elif inShape == (1, 1):
            # Broadcast a scalar.
            if value[0] != 0:
                value = load_dense(value[0], outShape, typecode)
            else:
                value = load_sparse([], [], [], outShape, typecode)
        elif inShape == (outShape[0], 1):
            # Copy columns horizontally.
            V = list(value.V)*outShape[1]
            I = list(value.I)*outShape[1]
            J = [j for j in range(outShape[1]) for _ in range(len(value))]
            value = load_sparse(V, I, J, outShape, typecode)
        elif inShape == (1, outShape[1]):
            # Copy rows vertically.
            V = list(value.V)*outShape[0]
            I = [i for i in range(outShape[0]) for _ in range(len(value))]
            J = list(value.J)*outShape[0]
            value = load_sparse(V, I, J, outShape, typecode)
        elif (inLength, 1) == outShape:
            # Vectorize in column-major order.
            if not typecode or value.typecode == typecode:
                # Quick method.
                value = value[:]
            else:
                # With typecode change.
                n, nnz = inShape[0], len(value)
                I, J = value.I, value.J
                I = [I[k] % n + J[k]*n for k in range(nnz)]
                J = [0]*nnz
                value = load_sparse(value.V, I, J, outShape, typecode)
        elif (1, inLength) == outShape:
            # Vectorize in row-major order.
            if not typecode or value.typecode == typecode:
                # Quick method.
                value = value.T[:].T
            else:
                # With typecode change.
                m, nnz = inShape[1], len(value)
                I, J = value.I, value.J
                J = [I[k]*m + J[k] % m for k in range(nnz)]
                I = [0]*nnz
                value = load_sparse(value.V, I, J, outShape, typecode)
        elif (outLength, 1) == inShape:
            # Devectorize in column-major order.
            # TODO: Logic for also changing the typecode.
            value = value[:]
            value.size = outShape
        elif (1, outLength) == inShape:
            # Devectorize in row-major order.
            # TODO: Logic for also changing the typecode.
            value = value[:]
            value.size = (outShape[1], outShape[0])
            value = value.T
        else:
            broadcast_error()
        if reshaped:
            alwaysCopy = False

        # The data now has the desired shape.
        assert value.size == outShape

        # Ensure the proper typecode.
        if typecode and value.typecode != typecode:
            # NOTE: In the case of intential loading as a dense matrix, the
            #       typecode is already set properly.
            assert isinstance(value, cvxopt.spmatrix)
            value = load_sparse(value.V, value.I, value.J, outShape, typecode)
            alwaysCopy = False

        # Return either a (sparse) copy or the input data itself.
        if isinstance(value, cvxopt.matrix):
            # The data was intentionally copied to a dense matrix (through
            # broadcasting), so only convert it back to sparse if requested.
            assert reshaped
            if sparse:
                value = cvxopt.sparse(value)
        elif sparse is False:
            value = load_dense(value, outShape, typecode)
        elif alwaysCopy:
            value = load_sparse(value.V, value.I, value.J, outShape, typecode)
    elif isinstance(value, numpy.ndarray):
        # NumPy arrays can be tensors, we don't support those.
        if len(value.shape) > 2:
            raise TypeError("PICOS does not support tensor data.")

        # Refine the output shape.
        inShape  = load_shape(value.shape)
        outShape = blend_shapes(shape, inShape)

        # Define shorthands.
        inLength  = inShape[0]  * inShape[1]
        outLength = outShape[0] * outShape[1]

        # If the input is one-dimensional, turn it into a column or row vector.
        if inShape != value.shape:
            if simple_vector_as_row(shape):
                value = value.reshape((1, inLength))
            else:
                value = value.reshape(inShape)

        # If the input is stored as complex and no complex output is explicitly
        # requested, try to remove an imaginary part of zero.
        if value.dtype.kind == "c" and typecode != "z" \
        and not numpy.iscomplex(value).any():
            value = value.real

        # Broadcast or reshape the data if necessary and possible.
        if inShape == outShape:
            pass
        elif inShape == (1, 1):
            # Broadcast a scalar.
            value = numpy.full(outShape, value)
        elif inShape == (outShape[0], 1):
            # Copy columns horizontally.
            value = value.repeat(outShape[1], 1)
        elif inShape == (1, outShape[1]):
            # Copy rows vertically.
            value = value.repeat(outShape[0], 0)
        elif (inLength, 1) == outShape:
            # Vectorize in column-major order.
            value = value.reshape(outShape, order="F")
        elif (1, inLength) == outShape:
            # Vectorize in row-major order.
            value = value.reshape(outShape, order="C")
        elif (outLength, 1) == inShape:
            # Devectorize in column-major order.
            value = value.reshape(outShape, order="F")
        elif (1, outLength) == inShape:
            # Devectorize in row-major order.
            value = value.reshape(outShape, order="C")
        else:
            broadcast_error()

        # The data now has the desired shape.
        assert value.shape == outShape

        # Decide on whether to create a dense or sparse matrix.
        outSparse = should_be_sparse(outShape, numpy.count_nonzero(value)) \
            if sparse is None else sparse

        # Convert to CVXOPT.
        if outSparse:
            I, J  = value.nonzero()
            V     = value[I, J]
            value = load_sparse(V, I, J, outShape, typecode)
        else:
            value = load_dense(value, outShape, typecode)
    elif isinstance(value, ComplexAffineExpression):
        # Must be constant.
        if not value.constant:
            raise ValueError("Cannot load the nonconstant expression {} as a "
                "constant data value.".format(value.string))

        # Retrieve a copy of the numeric value.
        value = value.value

        # NOTE: alwaysCopy=False as the value is already a copy.
        return load_data(value, shape, typecode, sparse, False, legacy)
    elif isinstance(value, str) and re.search(r"\s", value):
        value = value.strip().splitlines()
        value = [row.strip().split() for row in value]
        value = [[scalar(x, typecode) for x in row] for row in value]

        return load_data(value, shape, typecode, sparse, alwaysCopy, legacy)
    elif isinstance(value, str):
        # Verbatim matrix strings with only one element fall throught to this
        # case as they don't (have to) contain a whitespace character.
        try:
            value = scalar(value, typecode)
        except ValueError:
            pass
        else:
            return load_data(value, shape, typecode, sparse, alwaysCopy, legacy)

        match = _LOAD_DATA_REGEX.match(value)

        if not match:
            raise ValueError("The string '{}' could not be parsed as a matrix."
                .format(value))

        # Retrieve the string tokens.
        tokens = match.groups()
        assert len(tokens) == 3
        factor, base, inShape = tokens
        assert base is not None

        # Convert the factor.
        factor = scalar(factor) if factor else 1

        # Determine whether the matrix will be square.
        square = base == "I"

        # Convert the shape.
        if inShape:
            inShape = inShape[1:-1].split(",")
            if len(inShape) == 1:
                inShape *= 2
            inShape = (int(inShape[0]), int(inShape[1]))

            if blend_shapes(shape, inShape) != inShape:
                raise ValueError(
                    "Inconsistent shapes for matrix given as '{}' with expected"
                    " shape {}.".format(value, glyphs.shape(shape)))

            outShape = inShape
        elif None in shape:
            if square and shape != (None, None):
                outShape = blend_shapes(shape, (shape[1], shape[0]))
                assert None not in outShape
            else:
                raise ValueError("Could not determine the shape of a matrix "
                    "given as '{}' because the expected size of {} contains a "
                    "wildcard. Try to give the shape explicitly with the "
                    "string.".format(value, glyphs.shape(shape)))
        else:
            outShape = shape

        # Create the base matrix.
        if base.startswith("e_"):
            position = base[2:].split(",")
            if len(position) == 1:
                index = int(position[0]) - 1
                i, j = index % outShape[0], index // outShape[0]
            else:
                i, j = int(position[0]) - 1, int(position[1]) - 1

            if i >= outShape[0] or j >= outShape[1]:
                raise ValueError("Out-of-boundary unit at row {}, column {} "
                    "in matrix of shape {} given as '{}'."
                    .format(i + 1, j + 1, glyphs.shape(outShape), value))

            value = load_sparse([1.0], [i], [j], outShape, typecode)

            if outShape[1] == 1:
                assert j == 0
                string = "e_{}".format(i + 1)
            elif outShape[0] == 1:
                assert i == 0
                string = glyphs.transp("e_{}".format(j + 1))
            else:
                string = "e_{},{}".format(i + 1, j + 1)
        elif base.startswith("|"):
            element = scalar(base[1:-1])

            # Pull the factor inside the matrix.
            element *= factor
            factor = 1

            value  = load_dense(element, outShape, typecode)
            string = glyphs.scalar(element)
        elif base == "I":
            if outShape[0] != outShape[1]:
                raise ValueError("Cannot create a non-square identy matrix.")

            n = outShape[0]

            value = load_sparse([1.0]*n, range(n), range(n), outShape, typecode)
            string = glyphs.scalar(1) if n == 1 else glyphs.idmatrix()
        else:
            assert False, "Unexpected matrix base string '{}'.".format(base)

        # Apply a coefficient.
        if factor != 1:
            value = value * factor
            string = glyphs.mul(glyphs.scalar(factor), string)

        # Finalize the string.
        if base.startswith("e_"):
            string = glyphs.matrix(
                glyphs.compsep(glyphs.shape(outShape), string))
        elif base.startswith("|"):
            if outShape != (1, 1):
                string = glyphs.matrix(string)
        elif base == "I":
            pass
        else:
            assert False, "Unexpected matrix base string."

        # Convert between dense and sparse representation.
        if sparse is True and isinstance(value, cvxopt.matrix):
            value = cvxopt.sparse(value)
        elif sparse is False and isinstance(value, cvxopt.spmatrix):
            value = cvxopt.matrix(value)
    elif isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("Cannot parse an empty tuple or list as a matrix.")

        rows   = len(value)
        cols   = None
        nested = isinstance(value[0], (tuple, list))

        if nested:
            # Both outer and inner container must be lists. Unconditionally make
            # a copy of the outer container so we can replace any inner tuples.
            value = list(value)

            for rowNum, row in enumerate(value):
                if not isinstance(row, (tuple, list)):
                    raise TypeError("Expected a tuple or list for a matrix row "
                        "but found an element of type {}.".format(
                        type(row).__name__))

                # Make sure row length is consistent.
                if cols is None:
                    cols = len(row)

                    if not cols:
                        raise TypeError("Cannot parse an empty tuple or list as"
                            " a matrix row.")
                elif len(row) != cols:
                    raise TypeError("Rows of differing size in a matrix given "
                        "as a tuple or list: {}, {}.".format(cols, len(row)))

                if isinstance(row, tuple):
                    value[rowNum] = list(row)

            value = load_dense(value, (cols, rows), typecode).T
        elif rows == 1:
            outShape = blend_shapes(shape, (1, 1))
            return load_data(
                value[0], outShape, typecode, sparse, alwaysCopy, legacy)
        else:
            outShape = (1, rows) if simple_vector_as_row(shape) else (rows, 1)
            value = load_dense(value, outShape, typecode)

        # Recurse for further transformations (broadcasting, sparsity).
        return load_data(value, shape, typecode, sparse, False, legacy)
    else:
        raise TypeError("PICOS can't load an object of type {} as a matrix: {}."
            .format(type(value).__name__, repr(value)))

    # HACK: Work around CVXOPT not supporting integer sparse matrices: If
    #       integer is requested and the matrix is currently sparse, turn dense.
    #       Note that users may not request both sparsity and integrality.
    if typecode == "i" and value.typecode == "d":
        assert not sparse
        assert isinstance(value, cvxopt.spmatrix)
        value = load_dense(value, value.size, typecode)

    if legacy:
        # Handle the case of broadcasting a scalar for matrix multiplication.
        assert inShape is not None, "Conversions must define 'inSize'."
        if inShape == (1, 1) and None in shape and shape != (None, None) \
        and 1 not in shape:
            assert 1 in value.size

            value = cvxopt.spdiag(value)

            if sparse is False:
                value = cvxopt.dense(value)

            scalarString = glyphs.scalar(value[0])
            string = glyphs.mul(scalarString, glyphs.idmatrix()) \
                if scalarString != "1" else glyphs.idmatrix()

    # Validate the output shape and type.
    assert value.size == blend_shapes(shape, value.size)
    assert not typecode or value.typecode == typecode
    if sparse is None:
        assert isinstance(value, (cvxopt.matrix, cvxopt.spmatrix))
    elif sparse:
        assert isinstance(value, cvxopt.spmatrix)
    else:
        assert isinstance(value, cvxopt.matrix)

    # Fallback to a generic matrix string if no better string was found.
    if string is None:
        if value.size == (1, 1):
            string = glyphs.scalar(value[0])
        else:
            string = glyphs.matrix(glyphs.shape(value.size))

    return value, string


def load_sparse_data(value, shape=None, typecode=None, alwaysCopy=True):
    """See :func:`~.data.load_data` with ``sparse = True``."""
    return load_data(value=value, shape=shape, typecode=typecode, sparse=True,
        alwaysCopy=alwaysCopy)


def load_dense_data(value, shape=None, typecode=None, alwaysCopy=True):
    """See :func:`~.data.load_data` with ``sparse = False``."""
    return load_data(value=value, shape=shape, typecode=typecode, sparse=False,
        alwaysCopy=alwaysCopy)


def convert_and_refine_arguments(*which, refine=True, allowNone=False):
    """Convert selected function arguments to PICOS expressions.

    If the selected arguments are already PICOS expressions, they are refined
    unless disabled. If they are not already PICOS expressions, an attempt is
    made to load them as constant expressions.

    :Decorator guarantee:

    All specified arguments are refined PICOS expressions when the function is
    exectued.

    :param bool refine:
        Whether to refine arguments that are already PICOS expressions.
    :param bool allowNone:
        Whether :obj:`None` is passed through to the function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from .exp_affine import Constant
            from .expression import Expression

            def name():
                if hasattr(func, "__qualname__"):
                    return func.__qualname__
                else:
                    return func.__name__

            callargs = inspect.getcallargs(func, *args, **kwargs)

            newargs = {}
            for key, arg in callargs.items():
                if key not in which:
                    pass
                elif allowNone and arg is None:
                    pass
                elif isinstance(arg, Expression):
                    if refine:
                        arg = arg.refined
                else:
                    try:
                        arg = Constant(arg)
                    except Exception as error:
                        raise TypeError(
                            "Failed to convert argument '{}' of {} to a PICOS "
                            "constant.".format(key, name())) from error

                newargs[key] = arg

            return func(**newargs)
        return wrapper
    return decorator


def convert_operands(
        sameShape=False, diagBroadcast=False, scalarLHS=False, scalarRHS=False,
        rMatMul=False, lMatMul=False, horiCat=False, vertCat=False,
        allowNone=False):
    """Convert binary operator operands to PICOS expressions.

    A decorator for a binary operator that converts any operand that is not
    already a PICOS expression or set into a constant one that fulfills the
    given shape requirements, if possible. See :func:`~.data.load_data` for
    broadcasting and reshaping rules that apply to raw data.

    If both operands are already PICOS expressions and at least one of them is
    an affine expression, there is a limited set of broadcasting rules to fix
    a detected shape mismatch. If this does not succeed, an exception is raised.
    If no operand is affine, the operation is performed even if shapes do not
    match. The operation is responsible for dealing with this case.

    If either operand is a PICOS set, no broadcasting or reshaping is applied as
    set instances have, in general, variable dimensionality. If a set type can
    *not* have arbitrary dimensions, then it must validate the element's shape
    on its own. In particular, no shape requirement may be given when this
    decorator is used on a set method.

    :Decorator guarantee:

    This decorator guarantees to the binary operator using it that only PICOS
    expression or set types will be passed as operands and that any affine
    expression already has the proper shape for the operation, based on the
    decorator arguments.

    :Broadcasting rules for affine expressions:

    Currently, only scalar affine expressions are broadcasted to the next
    smallest matching shape. This is more limited than the broadcasting behavior
    when one of the operands is raw data, but it ensures a symmetric behavior in
    case both operands are affine. An exception is the case of diagBroadcast,
    where a vector affine expression may be extended to a matrix.

    :param bool sameShape: Both operands must have the exact same shape.
    :param bool diagBroadcast: Both operands must be square matrices of same
        shape. If one operand is a square matrix and the other is a scalar or
        vector, the latter is put into the diagonal of a square matrix.
    :param bool scalarLHS: The left hand side operand must be scalar.
    :param bool scalarRHS: The right hand side operand must be scalar.
    :param bool rMatMul: The operation has the shape requirements of normal
        matrix multiplication with the second operand on the right side.
    :param bool lMatMul: The operation has the shape requirements of reversed
        matrix multiplication with the second operand on the left side.
    :param bool horiCat: The operation has the shape requirements of horizontal
        matrix concatenation.
    :param bool vertCat: The operation has the shape requirements of vertical
        matrix concatenation.
    :param bool allowNone: An operand of :obj:`None` is passed as-is.

    :raises TypeError: If matching shapes cannot be produced despite one of the
        operands being raw data or an affine expression.

    .. note::
        Matrix multiplication includes scalar by matrix multiplication, so
        either operand may remain a scalar.
    """
    # Fix a redundancy in allowed argument combinations.
    if sameShape and (scalarLHS or scalarRHS):
        sameShape = False
        scalarLHS = True
        scalarRHS = True

    # Check arguments; only scalarLHS and scalarRHS may appear together.
    anyScalar = scalarLHS or scalarRHS
    args = (
        sameShape, diagBroadcast, anyScalar, lMatMul, rMatMul, horiCat, vertCat)
    selected  = len([arg for arg in args if arg])
    if selected > 1:
        assert False, "Conflicting convert_operands arguments."

    def decorator(operator):
        @functools.wraps(operator)
        def wrapper(lhs, rhs, *args, **kwargs):
            def fail(reason="Operand shapes do not match.", error=None):
                opName  = operator.__qualname__ \
                    if hasattr(operator, "__qualname__") else operator.__name__
                lhsName = lhs.string if hasattr(lhs, "string") else repr(lhs)
                rhsName = rhs.string if hasattr(rhs, "string") else repr(rhs)
                raise TypeError("Invalid operation {}({}, {}): {}".format(
                    opName, lhsName, rhsName, reason)) from error

            def make_lhs_shape(rhsShape):
                if sameShape or diagBroadcast:
                    return rhsShape
                elif horiCat:
                    return (rhsShape[0], None)
                elif vertCat:
                    return (None, rhsShape[1])
                elif rMatMul:
                    return (None, rhsShape[0])
                elif lMatMul:
                    return (rhsShape[1], None)
                elif scalarLHS:
                    return (1, 1)
                else:
                    return None

            def make_rhs_shape(lhsShape):
                if sameShape or diagBroadcast:
                    return lhsShape
                elif horiCat:
                    return (lhsShape[0], None)
                elif vertCat:
                    return (None, lhsShape[1])
                elif rMatMul:
                    return (lhsShape[1], None)
                elif lMatMul:
                    return (None, lhsShape[0])
                elif scalarRHS:
                    return (1, 1)
                else:
                    return None

            from .exp_affine import Constant
            from .exp_biaffine import BiaffineExpression
            from .expression import Expression
            from .set import Set

            lhsIsExpOrSet = isinstance(lhs, (Expression, Set))
            rhsIsExpOrSet = isinstance(rhs, (Expression, Set))

            lhsIsSet = lhsIsExpOrSet and isinstance(lhs, Set)
            rhsIsSet = rhsIsExpOrSet and isinstance(rhs, Set)

            if lhsIsSet:
                assert not selected, "convert_operands when used on sets may " \
                    "not pose any shape requirements."

            if lhsIsExpOrSet and rhsIsExpOrSet:
                lhsIsAffine = isinstance(lhs, BiaffineExpression)
                rhsIsAffine = isinstance(rhs, BiaffineExpression)

                # If neither expression is biaffine, it's the operation's job to
                # deal with it.
                if not lhsIsAffine and not rhsIsAffine:
                    return operator(lhs, rhs, *args, **kwargs)

                # If there are no shape requirements, we are done as both are
                # already expressions.
                if not selected:
                    return operator(lhs, rhs, *args, **kwargs)

                assert not lhsIsSet  # Handled earlier.

                # Sets have variable shape, so no adjustment is necessary.
                if rhsIsSet:
                    return operator(lhs, rhs, *args, **kwargs)

                lhsShape, rhsShape = lhs.shape, rhs.shape

                # Check if already matching size.
                if (sameShape or horiCat or vertCat) and lhsShape == rhsShape:
                    return operator(lhs, rhs, *args, **kwargs)

                lm, ln = lhs.shape
                rm, rn = rhs.shape
                lhsSquare = lm == ln
                rhsSquare = rm == rn

                # Further check if already matching size.
                if (diagBroadcast and lhsShape == rhsShape and lhsSquare) \
                or (horiCat and lm == rm) \
                or (vertCat and ln == rn) \
                or (rMatMul and ln == rm) \
                or (lMatMul and rn == lm):
                    return operator(lhs, rhs, *args, **kwargs)

                lhsL, rhsL = len(lhs), len(rhs)

                # scalarLHS and scalarRHS are the only two shape requirements
                # that may appear together, so handle all combinations here.
                if scalarLHS and scalarRHS:
                    if lhsL == 1 and rhsL == 1:
                        return operator(lhs, rhs, *args, **kwargs)
                elif scalarLHS and lhsL == 1:
                    return operator(lhs, rhs, *args, **kwargs)
                elif scalarRHS and rhsL == 1:
                    return operator(lhs, rhs, *args, **kwargs)

                # Matrix multiplication always accepts scalars.
                if (rMatMul or lMatMul) and 1 in (lhsL, rhsL):
                    return operator(lhs, rhs, *args, **kwargs)

                # Broadcast an affine scalar on the left side to match.
                if lhsIsAffine and lhsL == 1:
                    if diagBroadcast and rhsSquare:
                        lhs = lhs.dupdiag(rm)
                    else:
                        lhs = lhs.broadcasted(make_lhs_shape(rhsShape))
                    return operator(lhs, rhs, *args, **kwargs)

                # Broadcast an affine scalar on the right side to match.
                if rhsIsAffine and rhsL == 1:
                    if diagBroadcast and lhsSquare:
                        rhs = rhs.dupdiag(lm)
                    else:
                        rhs = rhs.broadcasted(make_rhs_shape(lhsShape))
                    return operator(lhs, rhs, *args, **kwargs)

                # Diagonally broadcast an affine vector on the left hand side.
                if diagBroadcast and lhsIsAffine and rhsSquare \
                and 1 in lhsShape and rm in lhsShape:
                    lhs = lhs.diag
                    assert lhs.shape == rhs.shape
                    return operator(lhs, rhs, *args, **kwargs)

                # Diagonally broadcast an affine vector on the right hand side.
                if diagBroadcast and rhsIsAffine and lhsSquare \
                and 1 in rhsShape and lm in rhsShape:
                    rhs = rhs.diag
                    assert lhs.shape == rhs.shape
                    return operator(lhs, rhs, *args, **kwargs)

                # At least one of the expressions is affine and we didn't find a
                # way to fix a detected shape mismatch. It's our job to error.
                fail("The operand shapes of {} and {} do not match.".format(
                    glyphs.shape(lhsShape), glyphs.shape(rhsShape)))
            elif lhsIsExpOrSet:
                if allowNone and rhs is None:
                    return operator(lhs, rhs, *args, **kwargs)

                rhsShape = None if lhsIsSet else make_rhs_shape(lhs.shape)

                if diagBroadcast and lhs.shape[0] != lhs.shape[1]:
                    fail("Given that the right hand side operand is not a PICOS"
                        " expression, the left hand side operand must be square"
                        " for this operation.")

                try:
                    if diagBroadcast:
                        try:
                            rhs = Constant(rhs, shape=(1, 1))
                        except Exception:
                            try:
                                rhs = Constant(rhs, shape=(lhs.shape[0], 1))
                            except Exception:
                                rhs = Constant(rhs, shape=rhsShape)
                            else:
                                rhs = rhs.diag
                        else:
                            rhs = rhs.dupdiag(lhs.shape[0])
                    elif rMatMul or lMatMul:
                        if lhs.shape == (1, 1):
                            # Any shape works.
                            rhs = Constant(rhs)
                        else:
                            try:  # Try loading as scalar factor first.
                                rhs = Constant(rhs, shape=(1, 1))
                            except Exception:
                                try:  # Try loading as a vector.
                                    if lMatMul:
                                        s = (1, lhs.shape[0])
                                    else:
                                        s = (lhs.shape[1], 1)

                                    rhs = Constant(rhs, shape=s)
                                except TypeError:  # Try loading as a matrix.
                                    rhs = Constant(rhs, shape=rhsShape)
                    else:
                        rhs = Constant(rhs, shape=rhsShape)
                except Exception as error:
                    fail("Could not load right hand side as a constant of "
                        "matching shape.", error)
            elif rhsIsExpOrSet:
                if allowNone and lhs is None:
                    return operator(lhs, rhs, *args, **kwargs)

                lhsShape = None if rhsIsSet else make_lhs_shape(rhs.shape)

                if diagBroadcast and rhs.shape[0] != rhs.shape[1]:
                    fail("Given that the left hand side operand is not a PICOS "
                        "expression, the right hand side operand must be square"
                        " for this operation.")

                try:
                    if diagBroadcast:
                        try:
                            lhs = Constant(lhs, shape=(1, 1))
                        except Exception:
                            try:
                                lhs = Constant(lhs, shape=(rhs.shape[0], 1))
                            except Exception:
                                lhs = Constant(lhs, shape=lhsShape)
                            else:
                                lhs = lhs.diag
                        else:
                            lhs = lhs.dupdiag(rhs.shape[0])
                    if rMatMul or lMatMul:
                        if rhs.shape == (1, 1):
                            # Any shape works.
                            lhs = Constant(lhs)
                        else:
                            try:  # Try loading as scalar factor first.
                                lhs = Constant(lhs, shape=(1, 1))
                            except TypeError:
                                try:  # Try loading as a vector.
                                    if rMatMul:
                                        s = (1, rhs.shape[0])
                                    else:
                                        s = (rhs.shape[1], 1)

                                    lhs = Constant(lhs, shape=s)
                                except TypeError:  # Try loading as a matrix.
                                    lhs = Constant(lhs, shape=lhsShape)
                    else:
                        lhs = Constant(lhs, shape=lhsShape)
                except Exception as error:
                    fail("Could not load left hand side as a constant of "
                        "matching shape.", error)
            else:
                assert False, "convert_operands is supposed to decorate " \
                    "expression methods, but neither operand is a PICOS " \
                    "expression or set."

            return operator(lhs, rhs, *args, **kwargs)
        return wrapper
    return decorator


def cvxopt_equals(A, B, absTol=None, relTol=None):
    """Whether two CVXOPT (sparse) matrices are numerically equal or close.

    For every common entry of ``A`` and ``B``, it is sufficient that one of the
    two tolerances, ``absTol`` or ``relTol``, is satisfied.

    :param float absTol:
        Maximum allowed entrywise absolute difference.

    :param float relTol:
        Maximum allowed entrywise quotient of absolute difference at the entry
        divided by the largest absolute value of any entry in both matrices.
    """
    if A.size != B.size:
        return False

    Z = A - B

    if not Z:
        return True

    if not absTol and not relTol:
        return False

    M = max(abs(Z))

    if relTol:
        N = max(max(abs(A)), max(abs(B)))

    if absTol and relTol:
        if M > absTol and M / N > relTol:
            return False
    elif absTol:
        if M > absTol:
            return False
    else:
        if M / N > relTol:
            return False

    return True


def cvxopt_maxdiff(A, B):
    """Return the largest absolute difference of two (sparse) CVXOPT matrices.

    :raises TypeError: If the matrices are not of the same shape.
    """
    if A.size != B.size:
        raise TypeError("The matrices do not have the same shape.")

    # Work around "ValueError: max() arg is an empty sequence" for sparse zero.
    if not A and not B:
        return 0.0

    return max(abs(A - B))


def cvxopt_hcat(matrices):
    """Concatenate the given CVXOPT (sparse) matrices horizontally.

    The resulting matrix is sparse if any input matrix is sparse.

    :param list matrices: A list of CVXOPT (sparse) matrices.
    """
    if not isinstance(matrices, list):
        matrices = list(matrices)

    sparse = any(isinstance(M, cvxopt.spmatrix) for M in matrices)

    matrices = [[matrix] for matrix in matrices]

    if sparse:
        return cvxopt.sparse(matrices)
    else:
        return cvxopt.matrix(matrices)


def cvxopt_vcat(matrices):
    """Concatenate the given CVXOPT (sparse) matrices vertically.

    The resulting matrix is sparse if any input matrix is sparse.

    :param list matrices: A list of CVXOPT (sparse) matrices.
    """
    if not isinstance(matrices, list):
        matrices = list(matrices)

    if any(isinstance(M, cvxopt.spmatrix) for M in matrices):
        return cvxopt.sparse(matrices)
    else:
        return cvxopt.matrix(matrices)


def cvxopt_hpsd(matrix):
    """Whether the given CVXOPT matrix is hermitian positive semidefinite.

    Uses :data:`~picos.settings.RELATIVE_HERMITIANNESS_TOLERANCE` and
    :data:`~picos.settings.RELATIVE_SEMIDEFINITENESS_TOLERANCE`.

    See also :func:`cvxopt_hpd`.

    .. warning::

        The semidefiniteness tolerance allows negative, near-zero eigenvalues.
    """
    if not cvxopt_equals(matrix, matrix.H,
            relTol=settings.RELATIVE_HERMITIANNESS_TOLERANCE):
        return False

    eigenvalues = numpy.linalg.eigvalsh(cvx2np(matrix))

    minimum = -(max(list(eigenvalues) + [1.0])
        * settings.RELATIVE_SEMIDEFINITENESS_TOLERANCE)

    return all(ev >= minimum for ev in eigenvalues)


def cvxopt_hpd(matrix):
    """Whether the given CVXOPT matrix is hermitian positive definite.

    Uses :data:`~picos.settings.RELATIVE_HERMITIANNESS_TOLERANCE`.

    See also :func:`cvxopt_hpsd`.
    """
    if not cvxopt_equals(matrix, matrix.H,
            relTol=settings.RELATIVE_HERMITIANNESS_TOLERANCE):
        return False

    eigenvalues = numpy.linalg.eigvalsh(cvx2np(matrix))

    return all(ev > 0 for ev in eigenvalues)


def cvxopt_inverse(matrix):
    """Return the inverse of the given CVXOPT matrix.

    :raises ValueError:
        If the matrix is not invertible.
    """
    matrix_np = cvx2np(matrix)

    try:
        inverse_np = numpy.linalg.inv(matrix_np)
    except numpy.linalg.LinAlgError as error:
        raise ValueError("Failed to invert a {} CVXOPT matrix using NumPy."
            .format(glyphs.shape(matrix.size))) from error

    inverse, _ = load_data(inverse_np, matrix.size)

    return inverse


def cvxopt_principal_root(matrix):
    """Return the principal square root of a symmetric positive semidef. matrix.

    Given a real symmetric positive (semi)definite CVXOPT input matrix, returns
    its unique positive (semi)definite matrix square root.

    .. warning::

        Does not validate that the input matrix is symmetric positive
        semidefinite and will still return a (useless) matrix if it is not.
    """
    matrix_np = cvx2np(matrix)
    U, s, _ = numpy.linalg.svd(matrix_np, hermitian=True)
    root_np = numpy.dot(U*(s**0.5), U.T)
    root, _ = load_data(root_np, matrix.size)

    return root


@lru_cache()
def cvxopt_K(m, n, typecode="d"):
    """The commutation matrix :math:`K_{(m,n)}` as a CVXOPT sparse matrix."""
    d = m*n
    V = [1]*d
    I = range(d)
    J = [(k % n)*m + k // n for k in I]
    return cvxopt.spmatrix(V, I, J, (d, d), typecode)


def sparse_quadruple(A, reshape=None, preT=False, postT=False):
    """Return a sparse representation of the given CVXOPT (sparse) matrix.

    :param reshape: If set, then :math:`A` is reshaped on the fly.
    :param bool preT: Transpose :math:`A` before reshaping.
    :param bool postT: Transpose :math:`A` after reshaping.

    :returns: A quadruple of values, row indices, column indices, and shape.
    """
    if not isinstance(A, (cvxopt.spmatrix, cvxopt.matrix)):
        raise TypeError("Input must be a CVXOPT (sparse) matrix.")

    m, n = A.size

    if reshape:
        reshape = load_shape(reshape)

        if A.size[0] * A.size[1] != reshape[0] * reshape[1]:
            raise TypeError("Cannot reshape from {} to {}.".format(
                glyphs.shape(A.size), glyphs.shape(reshape)))

    if not A:
        V, I, J = [], [], []
        p, q = reshape if reshape else (n, m if preT else m, n)
    else:
        if isinstance(A, cvxopt.matrix):
            A = cvxopt.sparse(A)

        V, I, J = A.V, A.I, A.J

        if preT:
            I, J, m, n = J, I, n, m

        if reshape:
            p, q = reshape
            I, J = zip(*[((i + j*m) % p, (i + j*m) // p) for i, j in zip(I, J)])
        else:
            p, q = m, n

    if postT:
        I, J, p, q = J, I, q, p

    return V, I, J, (p, q)


def left_kronecker_I(A, k, reshape=None, preT=False, postT=False):
    r"""Return :math:`I_k \otimes A` for a CVXOPT (sparse) matrix :math:`A`.

    In other words, if :math:`A` is a :math:`m \times n` CVXOPT (sparse)
    matrix, returns a :math:`km \times kn` CVXOPT sparse block matrix with
    all blocks of size :math:`m \times n`, the diagonal blocks (horizontal
    block index equal to vertical block index) equal to :math:`A`, and all
    other blocks zero.

    :param reshape: If set, then :math:`A` is reshaped on the fly.
    :param bool preT: Transpose :math:`A` before reshaping.
    :param bool postT: Transpose :math:`A` after reshaping.

    :returns:
        If :math:`A` is dense and :math:`k = 1`, a
        :func:`CVXOPT dense matrix <cvxopt:cvxopt.matrix>`, otherwise a
        :func:`CVXOPT sparse matrix <cvxopt:cvxopt.spmatrix>`.
    """
    A = A.T if preT else A

    if reshape:
        A = A if preT else A[:]  # Copy if not already fresh.
        A.size = reshape

    A = A.T if postT else A

    if k == 1:
        if not preT and not reshape and not postT:
            return A + 0  # Always return a copy.
        else:
            return A

    if A.size[0] == A.size[1]:
        A = cvxopt.spdiag([A for _ in range(k)])
    else:
        Z = cvxopt.spmatrix([], [], [], A.size)
        A = cvxopt.sparse([[Z]*i + [A] + [Z]*(k-i-1) for i in range(k)])

    return A.T if postT else A


def right_kronecker_I(A, k, reshape=None, preT=False, postT=False):
    r"""Return :math:`A \otimes I_k` for a CVXOPT (sparse) matrix :math:`A`.

    :param reshape: If set, then :math:`A` is reshaped on the fly.
    :param bool preT: Transpose :math:`A` before reshaping.
    :param bool postT: Transpose :math:`A` after reshaping.

    :returns:
        If :math:`A` is dense and :math:`k = 1`, a
        :func:`CVXOPT dense matrix <cvxopt:cvxopt.matrix>`, otherwise a
        :func:`CVXOPT sparse matrix <cvxopt:cvxopt.spmatrix>`.
    """
    if isinstance(A, cvxopt.matrix):
        # Dense case: Use NumPy.
        A = numpy.array(A)

        A = A.T if preT else A
        A = A.reshape(reshape, order="F") if reshape else A
        A = A.T if postT else A

        A = numpy.kron(A, numpy.eye(k))

        return load_data(A, sparse=(k > 1))[0]
    else:
        # Sparse case: Python implementation.
        # This is slower than the NumPy approach in general but can handle
        # arbitrarily large matrices given that they are sufficiently sparse.
        V, I, J, shape = sparse_quadruple(A, reshape, preT, postT)
        m, n = shape

        if V:
            V, I, J = zip(*[(v, k*i + l, k*j + l)
                for v, i, j in zip(V, I, J) for l in range(k)])

        return cvxopt.spmatrix(V, I, J, (k*m, k*n))


def cvx2np(A, reshape=None):
    """Convert a CVXOPT (sparse) matrix to a NumPy two-dimensional array.

    :param A: The CVXOPT :func:`dense <cvxopt:cvxopt.matrix>` or
        :func:`sparse <cvxopt:cvxopt.spmatrix>` matrix to convert.
    :param bool reshape: Optional new shape for the converted matrix.

    :returns: Converted :class:`NumPy array <numpy.ndarray>`.
    """
    assert isinstance(A, (cvxopt.matrix, cvxopt.spmatrix))

    if isinstance(A, cvxopt.spmatrix):
        A = cvxopt.matrix(A)

    if reshape:
        shape = load_shape(reshape)
        return numpy.reshape(A, shape, "F")
    else:
        return numpy.array(A)


def cvx2csc(A):
    """Convert a CVXOPT matrix to a SciPy sparse matrix in CSC format."""
    import scipy.sparse

    assert isinstance(A, (cvxopt.matrix, cvxopt.spmatrix))

    if isinstance(A, cvxopt.spmatrix):
        csc = tuple(tuple(x) for x in reversed(A.CCS))
        return scipy.sparse.csc_matrix(csc, shape=A.size)
    else:
        return scipy.sparse.csc_matrix(A)


def cvx2csr(A):
    """Convert a CVXOPT matrix to a SciPy sparse matrix in CSR format."""
    import scipy.sparse

    assert isinstance(A, (cvxopt.matrix, cvxopt.spmatrix))

    if isinstance(A, cvxopt.spmatrix):
        csc = tuple(tuple(x) for x in reversed(A.CCS))
        csc = scipy.sparse.csc_matrix(csc, shape=A.size)
        return csc.tocsr()
    else:
        return scipy.sparse.csr_matrix(A)


def sp2cvx(A):
    """Convert a SciPy sparse matrix to a CVXOPT sparse matrix."""
    import scipy.sparse

    assert isinstance(A, scipy.sparse.spmatrix)

    A = A.tocoo()

    V = A.data
    I = A.row
    J = A.col

    if issubclass(A.dtype.type, numpy.complexfloating):
        tc = "z"
    else:
        tc = "d"

    return cvxopt.spmatrix(V, I, J, A.shape, tc)


def make_fraction(p, denominator_limit):
    """Convert a float :math:`p` to a limited precision fraction.

    :param float p: The float to convert, may be positive or negative infinity.
    :param int denominator_limit: The largest allowed denominator.

    :returns tuple: A quadruple ``(num, den, pNew, pStr)`` with ``pNew`` the
        limited precision version of :math:`p`, ``pStr`` a string representation
        of the fraction, and ``num`` and ``den`` the numerator and the
        denominator of the fraction, respectively.
    """
    # LEGACY: Old tools.tracepow allowed this.
    if p in ("inf", "-inf"):
        p = float(p)

    if p in (float("inf"), float("-inf")):
        return p, 1, p, glyphs.neg(glyphs.infty) if p < 0 else glyphs.infty

    frac = Fraction(p).limit_denominator(denominator_limit)
    num  = frac.numerator
    den  = frac.denominator
    pNew = float(num) / float(den)

    if den == 1:
        pStr = glyphs.scalar(num)
    else:
        pStr = glyphs.clever_div(glyphs.scalar(num), glyphs.scalar(den))

    return num, den, pNew, pStr


def value(obj, sparse=None, numpy=False):
    """Convert (nested) PICOS objects to their current value.

    :param obj:
        Either a single (PICOS) object that has a ``value`` attribute, such as a
        :class:`mutable <picos.expressions.mutable.Mutable>`,
        :class:`expression <picos.expressions.expression.Expression>` or
        :class:`~picos.modeling.Problem`, or a (nested) :class:`list`,
        :class:`tuple` or :class:`dict` thereof.

    :param sparse:
        If :obj:`None`, retrieved multidimensional values can be returned as
        either CVXOPT :func:`sparse <cvxopt:cvxopt.spmatrix>` or
        :func:`dense <cvxopt:cvxopt.matrix>` matrices, whichever PICOS stores
        internally. If :obj:`True` or :obj:`False`, multidimensional values are
        always returned as sparse or dense types, respectively.

    :param bool numpy:
        If :obj:`True`, retrieved multidimensional values are returned as a
        NumPy :class:`~numpy:numpy.ndarray` instead of a CVXOPT type. May not be
        set in combination with ``sparse=True``.

    :returns:
        An object of the same (nested) structure as ``obj``, with every
        occurence of any object with a ``value`` attribute replaced by that
        attribute's current numeric value. In the case of dictionaries, only the
        dictionary values will be converted.

    :raises TypeError:
        If some object with a ``value`` attribute has a value that cannot be
        converted to a matrix by :func:`~.load_data`. This can only happen if
        the object in question is not a PICOS object.

    :Example:

    >>> from picos import RealVariable, value
    >>> from pprint import pprint
    >>> x = {key: RealVariable(key) for key in ("foo", "bar")}
    >>> x["foo"].value = 2
    >>> x["bar"].value = 3
    >>> pprint(value(x))
    {'bar': 3.0, 'foo': 2.0}
    """
    if sparse and numpy:
        raise ValueError("NumPy does not support sparse matrices.")

    if isinstance(obj, tuple):
        return tuple(value(inner, sparse, numpy) for inner in obj)
    elif isinstance(obj, list):
        return [value(inner, sparse, numpy) for inner in obj]
    elif isinstance(obj, dict):
        return {k: value(v, sparse, numpy) for k, v in obj.items()}
    else:
        if hasattr(obj, "value"):
            val = obj.value

            if isinstance(val, (int, float, complex)):
                return val

            # PICOS objects always return their value as a CVXOPT matrix type,
            # but this function may be used on other objects with a value
            # attribute. Try to convert their value to a CVXOPT matrix first.
            if not isinstance(val, (cvxopt.matrix, cvxopt.spmatrix)):
                if numpy:
                    load_sparse = False
                elif sparse:
                    load_sparse = True
                else:
                    load_sparse = None

                val = load_data(val, sparse=load_sparse)[0]
            elif isinstance(val, cvxopt.spmatrix) \
            and (sparse is False or numpy):
                val = cvxopt.dense(val)

            if numpy:
                import numpy as the_numpy
                assert isinstance(val, cvxopt.matrix)
                val = the_numpy.array(val)

            return val
        else:
            return obj


# --------------------------------------
__all__ = api_end(_API_START, globals())
