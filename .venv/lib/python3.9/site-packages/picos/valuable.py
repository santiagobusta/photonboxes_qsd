# ------------------------------------------------------------------------------
# Copyright (C) 2021 Maximilian Stahlberg
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

"""Common interface for objects that can have a numeric value."""

from abc import ABC, abstractmethod

import cvxopt
import numpy

from .apidoc import api_end, api_start

_API_START = api_start(globals())
# -------------------------------


class NotValued(RuntimeError):
    """The operation cannot be performed due to a mutable without a value.

    Note that the :attr:`~Valuable.value`, :attr:`~Valuable.value_as_matrix`,
    :attr:`~Valuable.np`, and :attr:`~Valuable.np2d` attributes do not raise
    this exception, but return :obj:`None` instead.
    """

    pass


class Valuable(ABC):
    """Abstract base class for objects that can have a numeric value.

    This is used by all algebraic expressions through their
    :class:`~picos.expressions.expression.Expression` base class as well as by
    :class:`~picos.modeling.objective.Objective` and, referencing the latter, by
    :class:`~picos.modeling.problem.Problem` instances.
    """

    # --------------------------------------------------------------------------
    # Abstract and default-implementation methods.
    # --------------------------------------------------------------------------

    @abstractmethod
    def _get_valuable_string(self):
        """Return a short string defining the valuable object."""
        pass

    @abstractmethod
    def _get_value(self):
        """Return the numeric value of the object as a CVXOPT matrix.

        :raises NotValued: When the value is not fully defined.

        Method implementations need to return an independent copy of the value
        that the user is allowed to change without affecting the object.
        """
        pass

    def _set_value(self, value):
        raise NotImplementedError("Setting the value on an instance of {} is "
            "not supported, but you can value any mutables involved instead."
            .format(type(self).__name__))

    # --------------------------------------------------------------------------
    # Provided interface.
    # --------------------------------------------------------------------------

    def _wrap_get_value(self, asMatrix, staySafe):
        """Enhance the implementation of :attr:`_get_value`.

        Checks the type of any value returned and offers conversion options.

        :param bool asMatrix:
            Whether scalar values are returned as matrices.

        :param bool staySafe:
            Whether :exc:`NotValued` exceptions are raised. Otherwise missing
            values are returned as :obj:`None`.
        """
        try:
            value = self._get_value()
        except NotValued:
            if staySafe:
                raise
            else:
                return None

        assert isinstance(value, (cvxopt.matrix, cvxopt.spmatrix)), \
            "Expression._get_value implementations must return a CVXOPT matrix."

        if value.size == (1, 1) and not asMatrix:
            return value[0]
        else:
            return value

    value = property(
        lambda self: self._wrap_get_value(asMatrix=False, staySafe=False),
        lambda self, x: self._set_value(x),
        lambda self: self._set_value(None),
        r"""Value of the object, or :obj:`None`.

        For an expression, it is defined if the expression is constant or if all
        mutables involved in the expression are valued. Mutables can be valued
        directly by writing to their :attr:`value` attribute. Variables are also
        valued by PICOS when an optimization solution is found.

        Some expressions can also be valued directly if PICOS can find a minimal
        norm mutable assignment that makes the expression have the desired
        value. In particular, this works with affine expressions whose linear
        part has an under- or well-determined coefficient matrix.

        If you prefer the value as a NumPy, use :attr:`np` instead.

        :returns:
            The value as a Python scalar or CVXOPT matrix, or :obj:`None` if it
            is not defined.

        :Distinction:

        - Unlike :attr:`safe_value` and :attr:`safe_value_as_matrix`, an
          undefined value is returned as :obj:`None`.
        - Unlike :attr:`value_as_matrix` and :attr:`safe_value_as_matrix`,
          scalars are returned as scalar types.
        - For uncertain expressions, see also
          :meth:`~.uexpression.UncertainExpression.worst_case_value`.

        :Example:

        >>> from picos import RealVariable
        >>> x = RealVariable("x", (1,3))
        >>> y = RealVariable("y", (1,3))
        >>> e = x - 2*y + 3
        >>> print("e:", e)
        e: x - 2·y + [3]
        >>> e.value = [4, 5, 6]
        >>> print("e: ", e, "\nx: ", x, "\ny: ", y, sep = "")
        e: [ 4.00e+00  5.00e+00  6.00e+00]
        x: [ 2.00e-01  4.00e-01  6.00e-01]
        y: [-4.00e-01 -8.00e-01 -1.20e+00]
        """)

    safe_value = property(
        lambda self: self._wrap_get_value(asMatrix=False, staySafe=True),
        lambda self, x: self._set_value(x),
        lambda self: self._set_value(None),
        """Value of the object, if defined.

        Refer to :attr:`value` for when it is defined.

        :returns:
            The value as a Python scalar or CVXOPT matrix.

        :raises ~picos.NotValued:
            If the value is not defined.

        :Distinction:

        - Unlike :attr:`value`, an undefined value raises an exception.
        - Like :attr:`value`, scalars are returned as scalar types.
        """)

    value_as_matrix = property(
        lambda self: self._wrap_get_value(asMatrix=True, staySafe=False),
        lambda self, x: self._set_value(x),
        lambda self: self._set_value(None),
        r"""Value of the object as a CVXOPT matrix type, or :obj:`None`.

        Refer to :attr:`value` for when it is defined (not :obj:`None`).

        :returns:
            The value as a CVXOPT matrix, or :obj:`None` if it is not defined.

        :Distinction:

        - Like :attr:`value`, an undefined value is returned as :obj:`None`.
        - Unlike :attr:`value`, scalars are returned as :math:`1 \times 1`
          matrices.
        """)

    safe_value_as_matrix = property(
        lambda self: self._wrap_get_value(asMatrix=True, staySafe=True),
        lambda self, x: self._set_value(x),
        lambda self: self._set_value(None),
        r"""Value of the object as a CVXOPT matrix type, if defined.

        Refer to :attr:`value` for when it is defined.

        :returns:
            The value as a CVXOPT matrix.

        :raises ~picos.NotValued:
            If the value is not defined.

        :Distinction:

        - Unlike :attr:`value`, an undefined value raises an exception.
        - Unlike :attr:`value`, scalars are returned as :math:`1 \times 1`
          matrices.
        """)

    @property
    def np2d(self):
        """Value of the object as a 2D NumPy array, or :obj:`None`.

        Refer to :attr:`value` for when it is defined (not :obj:`None`).

        :returns:
            The value as a two-dimensional :class:`numpy.ndarray`, or
            :obj:`None`, if the value is not defined.

        :Distinction:

        - Like :attr:`np`, values are returned as NumPy types or :obj:`None`.
        - Unlike :attr:`np`, both scalar and vectorial values are returned as
          two-dimensional arrays. In particular, row and column vectors are
          distinguished.
        """
        value = self.value_as_matrix

        if value is None:
            return None

        # Convert CVXOPT sparse to CVXOPT dense.
        if isinstance(value, cvxopt.spmatrix):
            value = cvxopt.matrix(value)

        assert isinstance(value, cvxopt.matrix)

        # Convert CVXOPT dense to a NumPy 2D array.
        value = numpy.array(value)

        assert len(value.shape) == 2

        return value

    @np2d.setter
    def np2d(self, value):
        self._set_value(value)

    @np2d.deleter
    def np2d(self):
        self._set_value(None)

    @property
    def np(self):
        """Value of the object as a NumPy type, or :obj:`None`.

        Refer to :attr:`value` for when it is defined (not :obj:`None`).

        :returns:
            A one- or two-dimensional :class:`numpy.ndarray`, if the value is a
            vector or a matrix, respectively, or a NumPy scalar type such as
            :obj:`numpy.float64`, if the value is a scalar, or :obj:`None`,
            if the value is not defined.

        :Distinction:

        - Like :attr:`value` and :attr:`np2d`, an undefined value is returned as
          :obj:`None`.
        - Unlike :attr:`value`, scalars are returned as NumPy scalar types as
          opposed to Python builtin scalar types while vectors and matrices are
          returned as NumPy arrays as opposed to CVXOPT matrices.
        - Unlike :attr:`np2d`, scalars are returned as NumPy scalar types and
          vectors are returned as NumPy one-dimensional arrays as opposed to
          always returning two-dimensional arrays.

        :Example:

        >>> from picos import ComplexVariable
        >>> Z = ComplexVariable("Z", (3, 3))
        >>> Z.value = [i + i*1j for i in range(9)]

        Proper matrices are return as 2D arrays:

        >>> Z.value  # CVXOPT matrix.
        <3x3 matrix, tc='z'>
        >>> Z.np  # NumPy 2D array.
        array([[0.+0.j, 3.+3.j, 6.+6.j],
               [1.+1.j, 4.+4.j, 7.+7.j],
               [2.+2.j, 5.+5.j, 8.+8.j]])

        Both row and column vectors are returned as 1D arrays:

        >>> z = Z[:,0]  # First column of Z.
        >>> z.value.size  # CVXOPT column vector.
        (3, 1)
        >>> z.T.value.size  # CVXOPT row vector.
        (1, 3)
        >>> z.value == z.T.value
        False
        >>> z.np.shape  # NumPy 1D array.
        (3,)
        >>> z.T.np.shape  # Same array.
        (3,)
        >>> from numpy import array_equal
        >>> array_equal(z.np, z.T.np)
        True

        Scalars are returned as NumPy types:

        >>> u = Z[0,0]  # First element of Z.
        >>> type(u.value)  # Python scalar.
        <class 'complex'>
        >>> type(u.np)  # NumPy scalar. #doctest: +SKIP
        <class 'numpy.complex128'>

        Undefined values are returned as None:

        >>> del Z.value
        >>> Z.value is Z.np is None
        True
        """
        value = self.np2d

        if value is None:
            return None
        elif value.shape == (1, 1):
            return value[0, 0]
        elif 1 in value.shape:
            return numpy.ravel(value)
        else:
            return value

    @np.setter
    def np(self, value):
        self._set_value(value)

    @np.deleter
    def np(self):
        self._set_value(None)

    @property
    def sp(self):
        """Value as a ScipPy sparse matrix or a NumPy 2D array or :obj:`None`.

        If PICOS stores the value internally as a CVXOPT sparse matrix, or
        equivalently if :attr:`value_as_matrix` returns an instance of
        :func:`cvxopt.spmatrix`, then this returns the value as a :class:`SciPy
        sparse matrix in CSC format <scipy.sparse.csc_matrix>`. Otherwise, this
        property is equivalent to :attr:`np2d` and returns a two-dimensional
        NumPy array, or :obj:`None`, if the value is undefined.

        :Example:

        >>> import picos, cvxopt
        >>> X = picos.RealVariable("X", (3, 3))
        >>> X.value = cvxopt.spdiag([1, 2, 3])  # Stored as a sparse matrix.
        >>> type(X.value)
        <class 'cvxopt.base.spmatrix'>
        >>> type(X.sp)
        <class 'scipy.sparse._csc.csc_matrix'>
        >>> X.value = range(9)  # Stored as a dense matrix.
        >>> type(X.value)
        <class 'cvxopt.base.matrix'>
        >>> type(X.sp)
        <class 'numpy.ndarray'>
        """
        import scipy.sparse

        value = self.value_as_matrix

        if value is None:
            return None
        elif isinstance(value, cvxopt.spmatrix):
            return scipy.sparse.csc_matrix(
                tuple(list(x) for x in reversed(value.CCS)), value.size)
        else:
            return numpy.array(value)

    @property
    def valued(self):
        """Whether the object is valued.

        .. note::

            Querying this attribute is *not* faster than immediately querying
            :attr:`value` and checking whether it is :obj:`None`. Use it only if
            you do not need to know the value, but only whether it is available.

        :Example:

        >>> from picos import RealVariable
        >>> x = RealVariable("x", 3)
        >>> x.valued
        False
        >>> x.value
        >>> print((x|1))
        ∑(x)
        >>> x.value = [1, 2, 3]
        >>> (x|1).valued
        True
        >>> print((x|1))
        6.0
        """
        try:
            self._get_value()
        except NotValued:
            return False
        else:
            return True

    @valued.setter
    def valued(self, x):
        if x is False:
            self._set_value(None)
        else:
            raise ValueError("You may only assign 'False' to the 'valued' "
                "attribute, which is the same as setting 'value' to 'None'.")

    def __index__(self):
        """Propose the value as an index."""
        value = self.value_as_matrix

        if value is None:
            raise NotValued("Cannot use unvalued {} as an index."
                .format(self._get_valuable_string()))

        if value.size != (1, 1):
            raise TypeError("Cannot use multidimensional {} as an index."
                .format(self._get_valuable_string()))

        value = value[0]

        if value.imag:
            raise ValueError(
                "Cannot use {} as an index as its value of {} has a nonzero "
                "imaginary part.".format(self._get_valuable_string(), value))

        value = value.real

        if not value.is_integer():
            raise ValueError("Cannot use {} as an index as its value of {} is "
                "not integral.".format(self._get_valuable_string(), value))

        return int(value)

    def _casting_helper(self, theType):
        assert theType in (int, float, complex)

        value = self.value_as_matrix

        if value is None:
            raise NotValued("Cannot cast unvalued {} as {}."
                .format(self._get_valuable_string(), theType.__name__))

        if value.size != (1, 1):
            raise TypeError(
                "Cannot cast multidimensional {} as {}."
                .format(self._get_valuable_string(), theType.__name__))

        value = value[0]

        return theType(value)

    def __int__(self):
        """Cast the value to an :class:`int`."""
        return self._casting_helper(int)

    def __float__(self):
        """Cast the value to a :class:`float`."""
        return self._casting_helper(float)

    def __complex__(self):
        """Cast the value to a :class:`complex`."""
        return self._casting_helper(complex)

    def __round__(self, ndigits=None):
        """Round the value to a certain precision."""
        return round(float(self), ndigits)

    def __array__(self, dtype=None):
        """Return the value as a :class:`NumPy array <numpy.ndarray>`."""
        value = self.safe_value_as_matrix

        # Convert CVXOPT sparse to CVXOPT dense.
        if isinstance(value, cvxopt.spmatrix):
            value = cvxopt.matrix(value)

        assert isinstance(value, cvxopt.matrix)

        # Convert CVXOPT dense to a NumPy 2D array.
        value = numpy.array(value, dtype)

        assert len(value.shape) == 2

        # Remove dimensions of size one.
        if value.shape == (1, 1):
            return numpy.reshape(value, ())
        elif 1 in value.shape:
            return numpy.ravel(value)
        else:
            return value

    # Prevent NumPy operators from loading PICOS expressions as arrays.
    __array_priority__ = float("inf")
    __array_ufunc__ = None


def patch_scipy_array_priority():
    """Monkey-patch scipy.sparse to make it respect ``__array_priority__``.

    This works around https://github.com/scipy/scipy/issues/4819 and is inspired
    by CVXPY's scipy_wrapper.py.
    """
    import scipy.sparse

    def teach_array_priority(operator):
        def respect_array_priority(self, other):
            if hasattr(other, "__array_priority__") \
            and self.__array_priority__ < other.__array_priority__:
                return NotImplemented
            else:
                return operator(self, other)

        return respect_array_priority

    base_type = scipy.sparse.spmatrix
    matrix_types = (type_ for type_ in scipy.sparse.__dict__.values()
        if isinstance(type_, type) and issubclass(type_, base_type))

    for matrix_type in matrix_types:
        for operator_name in (
            "__add__", "__div__", "__eq__", "__ge__", "__gt__", "__le__",
            "__lt__", "__matmul__", "__mul__", "__ne__", "__pow__", "__sub__",
            "__truediv__",
        ):
            operator = getattr(matrix_type, operator_name)

            # Wrap all binary operators of the base class and all overrides.
            if matrix_type is base_type \
            or operator is not getattr(base_type, operator_name):
                wrapped_operator = teach_array_priority(operator)
                setattr(matrix_type, operator_name, wrapped_operator)


# --------------------------------------
__all__ = api_end(_API_START, globals())
