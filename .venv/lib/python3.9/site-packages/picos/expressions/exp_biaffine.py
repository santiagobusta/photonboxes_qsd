# ------------------------------------------------------------------------------
# Copyright (C) 2019-2020 Maximilian Stahlberg
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

"""Implements :class:`BiaffineExpression`."""

import math
import operator
from abc import ABC, abstractmethod
from functools import reduce
from itertools import product

import cvxopt
import numpy

from .. import glyphs, settings
from ..apidoc import api_end, api_start
from ..caching import (cached_property, cached_selfinverse_property,
                       cached_selfinverse_unary_operator,
                       cached_unary_operator, unlocked_cached_properties)
from ..formatting import detect_range
from ..legacy import deprecated
from .data import (blend_shapes, convert_operands, cvx2np, cvxopt_equals,
                   cvxopt_K, cvxopt_vcat, left_kronecker_I, load_data,
                   load_dense_data, load_shape, right_kronecker_I)
from .expression import Expression, refine_operands
from .mutable import Mutable
from .vectorizations import FullVectorization, SymmetricVectorization

_API_START = api_start(globals())
# -------------------------------


class BiaffineExpression(Expression, ABC):
    r"""A multidimensional (complex) biaffine expression.

    Abstract base class for the affine
    :class:`~.exp_affine.ComplexAffineExpression` and
    its real subclass :class:`~.exp_affine.AffineExpression`.
    Quadratic expressions are stored in
    :class:`~.exp_quadratic.QuadraticExpression` instead.

    In general this expression has the form

    .. math::

        A(x,y) = B(x,y) + P(x) + Q(y) + C

    where :math:`x \in \mathbb{R}^p` and :math:`y \in \mathbb{R}^q` are variable
    vectors,
    :math:`B : \mathbb{R}^p \times \mathbb{R}^q \to \mathbb{C}^{m \times n}`
    is a bilinear function,
    :math:`P : \mathbb{R}^p \to \mathbb{C}^{m \times n}` and
    :math:`Q : \mathbb{R}^q \to \mathbb{C}^{m \times n}` are linear functions,
    and :math:`C \in \mathbb{C}^{m \times n}` is a constant.

    If no coefficient matrices defining :math:`B` and :math:`Q` are provided on
    subclass instanciation, then this acts as an affine function of :math:`x`.

    In a more technical sense, the notational variables :math:`x` and :math:`y`
    each represent a stack of vectorizations of a number of *actual* scalar,
    vector, or matrix variables or parameters :math:`X_i` and :math:`Y_j` with
    :math:`i, j \in \mathbb{Z}_{\geq 0}` and this class stores the nonzero
    (bi)linear functions
    :math:`B_{i,j}(\operatorname{vec}(X_i),\operatorname{vec}(Y_j))`,
    :math:`P_{i}(\operatorname{vec}(X_i))` and
    :math:`Q_{j}(\operatorname{vec}(Y_j))` in the form of separate sparse
    coefficient matrices.
    """

    # --------------------------------------------------------------------------
    # Static methods.
    # --------------------------------------------------------------------------

    @staticmethod
    def _update_coefs(coefs, mtbs, summand):
        if not summand:
            return
        elif mtbs in coefs:
            coefs[mtbs] = coefs[mtbs] + summand
        else:
            coefs[mtbs] = summand

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    @classmethod
    def _get_type_string_base(cls):
        """Return a string template for the expression type string."""
        return "Complex {}" if cls._get_typecode() == "z" else "Real {}"

    def __init__(self, string, shape=(1, 1), coefficients={}):
        """Initialize a (complex) biaffine expression.

        :param str string: A symbolic string description.

        :param shape: Shape of a vector or matrix expression.
        :type shape: int or tuple or list

        :param dict coefficients:
            Maps mutable pairs, single mutables or the empty tuple to sparse
            represetations of the coefficient matrices :math:`B`, :math:`P` and
            :math:`Q`, and :math:`C`, respectively.

        .. warning::
            If the given coefficients are already of the desired (numeric) type
            and shape, they are stored by reference. Modifying such data can
            lead to unexpected results as PICOS expressions are supposed to be
            immutable (to allow
            `caching <https://en.wikipedia.org/wiki/Cache_(computing)>`_ of
            results).

            If you create biaffine expressions by any other means than this
            constructor, PICOS makes a copy of your data to prevent future
            modifications to it from causing inconsistencies.
        """
        from .variables import BaseVariable

        shape  = load_shape(shape)
        length = shape[0]*shape[1]

        if not isinstance(coefficients, dict):
            raise TypeError("Coefficients of {} must be given as a dict."
                .format(type(self).__name__))

        # Store shape.
        self._shape = shape

        # Store coefficients.
        self._coefs = {}
        for mtbs, coef in coefficients.items():
            if isinstance(mtbs, Mutable):
                mtbs = (mtbs,)
            elif not isinstance(mtbs, tuple):
                raise TypeError("Coefficient indices must be tuples, not "
                    "objects of type {}.".format(type(mtbs).__name__))

            if not all(isinstance(mtb, Mutable) for mtb in mtbs):
                raise TypeError("Coefficients must be indexed by mutables.")

            if not self._parameters_allowed \
            and not all(isinstance(mtb, BaseVariable) for mtb in mtbs):
                raise TypeError("Coefficients of {} may only be indexed by "
                    "decision variables.".format(type(self).__name__))

            # Store only linear and potentially bilinear terms.
            if len(mtbs) != len(set(mtbs)):
                raise TypeError("Coefficients of {} must be indexed by disjoint"
                    " mutables; no quadratic terms are allowed."
                    .format(type(self).__name__))
            elif len(mtbs) > 2:
                raise TypeError("Coefficients of {} may be indexed by at "
                    "most two mutables.".format(type(self).__name__))
            elif len(mtbs) > 1 and not self._bilinear_terms_allowed:
                raise TypeError("Coefficients of {} may be indexed by at "
                    "most one mutable.".format(type(self).__name__))

            # Do not store coefficients of zero.
            if not coef:
                continue

            # Dimension of the tensor product of the vectorized mutables.
            dim = reduce(operator.mul, (mtb.dim for mtb in mtbs), 1)

            # Load the coefficient matrix in the desired format.
            coef = load_data(
                coef, (length, dim), self._typecode, alwaysCopy=False)[0]

            # Always store with respect to ordered mutables.
            if len(mtbs) == 2 and mtbs[0].id > mtbs[1].id:
                # Obtain a fitting commutation matrix.
                K = cvxopt_K(mtbs[1].dim, mtbs[0].dim, self._typecode)

                # Make coef apply to vec(y*x.T) instead of vec(x*y.T).
                coef = coef * K

                # Swap x and y.
                mtbs = (mtbs[1], mtbs[0])

            # Sum with existing coefficients submitted in opposing order.
            if mtbs in self._coefs:
                self._coefs[mtbs] = self._coefs[mtbs] + coef
            else:
                self._coefs[mtbs] = coef

        # Determine the type string.
        typeStr = self._get_type_string_base()

        if "{}" in typeStr:
            hasBilinear = bool(self._bilinear_coefs)
            hasLinear   = bool(self._linear_coefs)
            hasConstant = bool(self._constant_coef)

            if hasBilinear:
                if hasConstant:
                    typeStr = typeStr.format("Biaffine Expression")
                else:
                    typeStr = typeStr.format("Bilinear Expression")
            elif hasLinear:
                if hasConstant:
                    typeStr = typeStr.format("Affine Expression")
                else:
                    typeStr = typeStr.format("Linear Expression")
            else:
                typeStr = typeStr.format("Constant")

        typeStr = "{} {}".format(glyphs.shape(shape), typeStr)

        Expression.__init__(self, typeStr, string)

    @classmethod
    def from_constant(cls, constant, shape=None, name=None):
        """Create a class instance from the given numeric constant.

        Loads the given constant as a PICOS expression, optionally broadcasted
        or reshaped to the given shape and named as specified.

        See :func:`~.data.load_data` for supported data formats and broadcasting
        and reshaping rules.

        Unlike :func:`~.exp_affine.Constant`, this class method always creates
        an instance of the class that it is called on, instead of tailoring
        towards the numeric type of the data.

        .. note::
            When an operation involves both a PICOS expression and a constant
            value of another type, PICOS converts the constant on the fly so
            that you rarely need to use this method.
        """
        constant, string = load_data(constant, shape, cls._get_typecode())
        return cls._get_basetype()(
            name if name else string, constant.size, {(): constant[:]})

    @classmethod
    def zero(cls, shape=(1, 1)):
        """Return a constant zero expression of given shape."""
        shape  = load_shape(shape)
        string = glyphs.scalar(0)
        return cls._get_basetype()(string, shape)

    # --------------------------------------------------------------------------
    # (Abstract) class methods.
    # --------------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def _get_bilinear_terms_allowed(cls):
        """Report whether the subclass may have bilinear terms."""
        pass

    @classmethod
    @abstractmethod
    def _get_parameters_allowed(cls):
        """Report whether the subclass may depend on parameters."""
        pass

    @classmethod
    @abstractmethod
    def _get_basetype(cls):
        """Return first non-abstract :class:`Expression` subclass this bases on.

        Enables subclass objects (such as variables) to behave like the returned
        type with respect to algebraic operations. For instance, the sum of
        two :class:`ComplexVariable` is a :class:`ComplexAffineExpression`.
        """
        pass

    @classmethod
    @abstractmethod
    def _get_typecode(cls):
        """Return the CVXOPT typecode to use with coefficient matrices.

        Either ``"z"`` for complex or ``"d"`` for real.

        See also :meth:`_get_basetype`.
        """
        pass

    @classmethod
    def _common_basetype(cls, other, reverse=False):
        """Return the common basetype of two expressions.

        The parameter ``other`` may be either a :class:`BiaffineExpression`
        instance or subclass thereof.
        """
        myBasetype = cls._get_basetype()
        theirBasetype = other._get_basetype()

        if myBasetype is theirBasetype:
            return myBasetype
        elif issubclass(theirBasetype, myBasetype):
            return myBasetype
        elif issubclass(myBasetype, theirBasetype):
            return theirBasetype
        elif not reverse:
            # HACK: Handle the case where the individual base types do not have
            #       a sub- and superclass relation but where one of the types
            #       still knows what the resulting base class should be.
            return other._common_basetype(cls, reverse=True)
        else:
            raise TypeError("The expression types {} and {} do not have a "
                "common base type apart from the abstract {} so the operation "
                "cannot be performed.".format(cls.__name__, other.__name__,
                BiaffineExpression.__name__))

    # --------------------------------------------------------------------------
    # Internal use properties and methods.
    # --------------------------------------------------------------------------

    @property
    def _bilinear_terms_allowed(self):
        """Shorthand for :meth:`_get_bilinear_terms_allowed`."""
        return self._get_bilinear_terms_allowed()

    @property
    def _parameters_allowed(self):
        """Shorthand for :meth:`_get_parameters_allowed`."""
        return self._get_parameters_allowed()

    @property
    def _basetype(self):
        """Shorthand for :meth:`_get_basetype`."""
        return self._get_basetype()

    @property
    def _typecode(self):
        """Shorthand for :meth:`_get_typecode`."""
        return self._get_typecode()

    @cached_property
    def _constant_coef(self):
        """Vectorized constant term with zero represented explicitly."""
        if () in self._coefs:
            return self._coefs[()]
        else:
            return load_data(0, len(self), self._typecode)[0]

    @cached_property
    def _linear_coefs(self):
        """Linear part coefficients indexed directly by single mutables."""
        return {mtbs[0]: c for mtbs, c in self._coefs.items() if len(mtbs) == 1}

    @cached_property
    def _bilinear_coefs(self):
        """Bilinear part coefficients indexed by mutable pairs."""
        return {mtbs: c for mtbs, c in self._coefs.items() if len(mtbs) == 2}

    @cached_property
    def _sparse_coefs(self):
        """Coefficients as sparse matrices."""
        return {
            mtbs: c if isinstance(c, cvxopt.spmatrix) else cvxopt.sparse(c)
            for mtbs, c in self._coefs.items()}

    @cached_property
    def _sparse_linear_coefs(self):
        """Linear part coefficients cast to sparse and indexed by mutables."""
        return {v[0]: c for v, c in self._sparse_coefs.items() if len(v) == 1}

    def _reglyphed_string(self, mutable_name_map):
        """The symbolic string with mutable names replaced."""
        string = self.string

        if isinstance(string, glyphs.GlStr):
            string = string.reglyphed(mutable_name_map)
        elif string in mutable_name_map:
            # Handle corner cases like x + 0 for a mutable x which is not a
            # mutable any more but still has a literal string name.
            string = mutable_name_map[string]

        return string

    # --------------------------------------------------------------------------
    # Abstract method implementations and overridings for Expression.
    # --------------------------------------------------------------------------

    def _get_value(self):
        # Create a copy of the constant term.
        value = self._constant_coef[:]

        # Add value of the linear part.
        for mtb, coef in self._linear_coefs.items():
            summand = coef * mtb._get_internal_value()

            if type(value) == type(summand):
                value += summand
            else:
                # Exactly one of the matrices is sparse.
                value = value + summand

        # Add value of the bilinear part.
        for (x, y), coef in self._bilinear_coefs.items():
            xValue = x._get_internal_value()
            yValue = y._get_internal_value()
            summand = coef * (xValue*yValue.T)[:]

            if type(value) == type(summand):
                value += summand
            else:
                # Exactly one of the matrices is sparse.
                value = value + summand

        # Resize the value to the proper shape.
        value.size = self.shape

        return value

    def _get_shape(self):
        return self._shape

    @cached_unary_operator
    def _get_mutables(self):
        return frozenset(mtb for mtbs in self._coefs for mtb in mtbs)

    def _is_convex(self):
        return not self._bilinear_coefs

    def _is_concave(self):
        return not self._bilinear_coefs

    def _replace_mutables(self, mapping):
        # Handle the base case where the affine expression is a mutable.
        if self in mapping:
            return mapping[self]

        name_map = {old.name: new.string for old, new in mapping.items()}
        string = self._reglyphed_string(name_map)

        if all(isinstance(new, Mutable) for new in mapping.values()):
            # Fast implementation for the basic case.
            coefs = {tuple(mapping[mtb] for mtb in mtbs): coef
                for mtbs, coef in self._coefs.items()}
        else:
            # Turn full mapping into an effective mapping.
            mapping = {o: n for o, n in mapping.items() if o is not n}

            coefs = {}
            for old_mtbs, old_coef in self._coefs.items():
                if not any(old_mtb in mapping for old_mtb in old_mtbs):
                    self._update_coefs(coefs, old_mtbs, old_coef)
                elif len(old_mtbs) == 1:
                    assert old_mtbs[0] in mapping
                    new = mapping[old_mtbs[0]]
                    for new_mtb, new_coef in new._linear_coefs.items():
                        self._update_coefs(coefs, (new_mtb,), old_coef*new_coef)
                    self._update_coefs(coefs, (), old_coef*new._constant_coef)
                elif old_mtbs[0] in mapping and old_mtbs[1] not in mapping:
                    new = mapping[old_mtbs[0]]
                    old_mtb = old_mtbs[1]
                    for new_mtb, new_coef in new._linear_coefs.items():
                        self._update_coefs(coefs, (new_mtb, old_mtb),
                            old_coef*(left_kronecker_I(new_coef, old_mtb.dim)))
                    self._update_coefs(coefs, (old_mtb,), old_coef*(
                        left_kronecker_I(new._constant_coef, old_mtb.dim)))
                elif old_mtbs[0] not in mapping and old_mtbs[1] in mapping:
                    new = mapping[old_mtbs[1]]
                    old_mtb = old_mtbs[0]
                    for new_mtb, new_coef in new._linear_coefs.items():
                        self._update_coefs(coefs, (old_mtb, new_mtb),
                            old_coef*(right_kronecker_I(new_coef, old_mtb.dim)))
                    self._update_coefs(coefs, (old_mtb,), old_coef*(
                        right_kronecker_I(new._constant_coef, old_mtb.dim)))
                elif old_mtbs[0] in mapping and old_mtbs[1] in mapping:
                    new1 = mapping[old_mtbs[0]]
                    new2 = mapping[old_mtbs[1]]
                    if isinstance(new1, Mutable) and isinstance(new2, Mutable):
                        self._update_coefs(coefs, (new1, new2), old_coef)
                    else:
                        raise NotImplementedError(
                            "Replacing both mutables in a bilinear term is not "
                            "supported unless both are replaced with mutables. "
                            "The effective mapping was: {}".format(mapping))
                else:
                    assert False

        return self._basetype(string, self._shape, coefs)

    def _freeze_mutables(self, freeze):
        string = self._reglyphed_string(
            {mtb.name: glyphs.frozen(mtb.name) for mtb in freeze})

        coefs = {}
        for mtbs, coef in self._coefs.items():
            if not any(mtb in freeze for mtb in mtbs):
                self._update_coefs(coefs, mtbs, coef)
            elif len(mtbs) == 1:
                assert mtbs[0] in freeze
                self._update_coefs(coefs, (), coef*mtbs[0].internal_value)
            elif mtbs[0] in freeze and mtbs[1] in freeze:
                C = coef*(mtbs[0].internal_value*mtbs[1].internal_value.T)[:]
                self._update_coefs(coefs, (), C)
            elif mtbs[0] in freeze and mtbs[1] not in freeze:
                C = coef*left_kronecker_I(mtbs[0].internal_value, mtbs[1].dim)
                self._update_coefs(coefs, (mtbs[1],), C)
            elif mtbs[0] not in freeze and mtbs[1] in freeze:
                C = coef*right_kronecker_I(mtbs[1].internal_value, mtbs[0].dim)
                self._update_coefs(coefs, (mtbs[0],), C)
            else:
                assert False

        return self._basetype(string, self._shape, coefs)

    # --------------------------------------------------------------------------
    # Python special method implementations.
    # --------------------------------------------------------------------------

    def __len__(self):
        # Faster version that overrides Expression.__len__.
        return self._shape[0] * self._shape[1]

    def __getitem__(self, index):
        def slice2range(s, length):
            """Transform a :class:`slice` to a :class:`range`."""
            assert isinstance(s, slice)

            # Plug in slice's default values.
            ss = s.step if s.step else 1
            if ss > 0:
                sa = s.start if s.start is not None else 0
                sb = s.stop  if s.stop  is not None else length
            else:
                assert ss < 0
                sa = s.start if s.start is not None else length - 1
                sb = s.stop  # Keep it None as -1 would mean length - 1.

            # Wrap negative indices (once).
            ra = length + sa if sa < 0 else sa
            if sb is None:
                # This is the only case where we give a negative index to range.
                rb = -1
            else:
                rb = length + sb if sb < 0 else sb

            # Clamp out-of-bound indices.
            ra = min(max(0,  ra), length - 1)
            rb = min(max(-1, rb), length)

            r = range(ra, rb, ss)

            if not r:
                raise IndexError("Empty slice.")

            return r

        def range2slice(r, length):
            """Transform a :class:`range` to a :class:`slice`, if possible.

            :raises ValueError: If the input cannot be expressed as a slice.
            """
            assert isinstance(r, range)

            if not r:
                raise IndexError("Empty range.")

            ra = r.start
            rb = r.stop
            rs = r.step

            if rs > 0:
                if ra < 0 or rb > length:
                    raise ValueError(
                        "Out-of-bounds range cannot be represented as a slice.")
            else:
                assert rs < 0
                if ra >= length or rb < -1:
                    raise ValueError(
                        "Out-of-bounds range cannot be represented as a slice.")

                if rb == -1:
                    rb = None

            return slice(ra, rb, rs)

        def list2slice(l, length):
            """Transform a :class:`list` to a :class:`slice`, if possible.

            :raises TypeError: If the input is not an integer sequence.
            :raises ValueError: If the input cannot be expressed as a slice.
            """
            return range2slice(detect_range(l), length)

        def slice2str(s):
            """Return the short string that produced a :class:`slice`."""
            assert isinstance(s, slice)

            startStr = str(s.start) if s.start is not None else ""
            stopStr  = str(s.stop)  if s.stop  is not None else ""
            if s.step in (None, 1):
                if s.start is not None and s.stop is not None \
                and s.stop == s.start + 1:
                    return startStr
                else:
                    return "{}:{}".format(startStr, stopStr)
            else:
                return "{}:{}:{}".format(startStr, stopStr, str(s.step))

        def list2str(l, length):
            """Return a short string represnetation of a :class:`list`."""
            assert isinstance(l, list)

            # Extract integers wrapped in a list.
            if len(l) == 1:
                return str(l[0])

            # Try to convert the list to a slice.
            try:
                l = list2slice(l, length)
            except (ValueError, RuntimeError):
                pass

            if isinstance(l, list):
                if len(l) > 4:
                    return glyphs.shortint(l[0], l[-1])
                else:
                    return str(l).replace(" ", "")
            else:
                return slice2str(l)

        def any2str(a, length):
            if isinstance(a, slice):
                return slice2str(a)
            elif isinstance(a, list):
                return list2str(a, length)
            else:
                assert False

        m, n = self._shape
        indexStr = None
        isIntList = False

        # Turn the index expression into a mutable list of one index per axis.
        if isinstance(index, tuple):  # Multiple axis slicing.
            index = list(index)
        elif isinstance(index, dict):  # Arbitrary matrix element selection.
            if len(index) != 2:
                raise TypeError("When slicing with a dictionary, there must be "
                    "exactly two keys.")

            try:
                i, j = sorted(index.keys())
            except TypeError as error:
                raise TypeError("When slicing with a dictionary, the two keys "
                    "must be comparable.") from error

            I, J = index[i], index[j]

            try:
                I = load_dense_data(I, typecode="i", alwaysCopy=False)[0]
                J = load_dense_data(J, typecode="i", alwaysCopy=False)[0]

                if 1 not in I.size or 1 not in J.size:
                    raise TypeError("At least one of the objects is not flat "
                        "but a proper matrix.")

                if len(I) != len(J):
                    raise TypeError("The objects do not have the same length.")

                I, J = list(I), list(J)
            except Exception as error:
                raise TypeError("Loading a sparse index vector pair for {} from"
                    " objects of type {} and {} failed: {}".format(self.string,
                    type(index[i]).__name__, type(index[j]).__name__, error)) \
                    from None

            # Represent the selection as a global index list.
            index = [[i + j*m for i, j in zip(I, J)]]

            # Use a special index string.
            indexStr = glyphs.size(list2str(I, n), list2str(J, m))

            # Don't invoke load_dense_data on the list.
            isIntList = True
        else:  # Global indexing.
            index = [index]

        # Make sure either global or row/column indexing is used.
        if not index:
            raise IndexError("Empty index.")
        elif len(index) > 2:
            raise IndexError(
                "PICOS expressions do not have a third axis to slice.")

        # Turn the indices for each axis into either a slice or a list.
        for axis, selection in enumerate(index):
            # Convert anything that is not a slice, including scalars and lists
            # that are not confirmed integer, to an integer row or column
            # vector, then (back) to a list.
            if not isIntList and not isinstance(selection, slice):
                try:
                    matrix = load_dense_data(
                        selection, typecode="i", alwaysCopy=False)[0]

                    if 1 not in matrix.size:
                        raise TypeError("The object is not flat but a {} shaped"
                            " matrix.".format(glyphs.shape(matrix.size)))

                    selection = list(matrix)
                except Exception as error:
                    raise TypeError("Loading a slicing index vector for axis {}"
                        " of {} from an object of type {} failed: {}".format(
                        axis, self.string, type(selection).__name__, error)) \
                        from None

            index[axis] = selection

        # Build index string, retrieve new shape, finalize index.
        if len(index) == 1:  # Global indexing.
            index = index[0]

            if isinstance(index, slice):
                shape = len(slice2range(index, len(self)))
            else:
                shape = len(index)

            if indexStr is None:
                indexStr = any2str(index, len(self))
        else:  # Multiple axis slicing.
            if indexStr is None:
                indexStr = "{},{}".format(
                    any2str(index[0], m), any2str(index[1], n))

            # Convert to a global index list.
            RC, shape = [], []
            for axis, selection in enumerate(index):
                k = self._shape[axis]

                if isinstance(selection, slice):
                    # Turn the slice into an iterable range.
                    selection = slice2range(selection, k)

                    # All indices from a slice are nonnegative.
                    assert all(i >= 0 for i in selection)

                if isinstance(selection, list):
                    # Wrap once. This is consistent with CVXOPT.
                    selection = [i if i >= 0 else k + i for i in selection]

                    # Perform a partial out-of-bounds check.
                    if any(i < 0 for i in selection):
                        raise IndexError(
                            "Out-of-bounds access along axis {}.".format(axis))

                # Complete the check for out-of-bounds access.
                if any(i >= k for i in selection):
                    raise IndexError(
                        "Out-of-bounds access along axis {}.".format(axis))

                RC.append(selection)
                shape.append(len(selection))

            rows, cols = RC
            index = [i + j*m for j in cols for i in rows]

        # Finalize the string.
        string = glyphs.slice(self.string, indexStr)

        # Retrieve new coefficients and constant term.
        coefs = {mtbs: coef[index, :] for mtbs, coef in self._coefs.items()}

        return self._basetype(string, shape, coefs)

    @convert_operands(sameShape=True)
    @refine_operands(stop_at_affine=True)
    def __add__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__add__(self, other)

        string = glyphs.clever_add(self.string, other.string)

        coefs = {}
        for mtbs, coef in self._coefs.items():
            coefs[mtbs] = coef + other._coefs[mtbs] \
                if mtbs in other._coefs else coef
        for mtbs, coef in other._coefs.items():
            coefs.setdefault(mtbs, coef)

        return self._common_basetype(other)(string, self._shape, coefs)

    @convert_operands(sameShape=True)
    @refine_operands(stop_at_affine=True)
    def __radd__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__radd__(self, other)

        return other.__add__(self)

    @convert_operands(sameShape=True)
    @refine_operands(stop_at_affine=True)
    def __sub__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__sub__(self, other)

        string = glyphs.clever_sub(self.string, other.string)

        coefs = {}
        for mtbs, coef in self._coefs.items():
            coefs[mtbs] = coef - other._coefs[mtbs] \
                if mtbs in other._coefs else coef
        for mtbs, coef in other._coefs.items():
            coefs.setdefault(mtbs, -coef)

        return self._common_basetype(other)(string, self._shape, coefs)

    @convert_operands(sameShape=True)
    @refine_operands(stop_at_affine=True)
    def __rsub__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__rsub__(self, other)

        return other.__sub__(self)

    @cached_selfinverse_unary_operator
    def __neg__(self):
        string = glyphs.clever_neg(self.string)
        coefs  = {mtbs: -coef for mtbs, coef in self._coefs.items()}

        return self._basetype(string, self._shape, coefs)

    @convert_operands(rMatMul=True)
    @refine_operands(stop_at_affine=True)
    def __mul__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__mul__(self, other)

        string = glyphs.clever_mul(self.string, other.string)

        m, n = self._shape
        p, q = other._shape

        # Handle or prepare multiplication with a scalar.
        if 1 in (m*n, p*q):
            if self.constant or other.constant:
                # Handle all cases involving a constant operand immediately.
                if self.constant:
                    lhs = self._constant_coef
                    RHS = other._coefs
                else:
                    lhs = other._constant_coef
                    RHS = self._coefs

                shape = other._shape if m*n == 1 else self._shape

                # Work around CVXOPT not broadcasting sparse scalars.
                if lhs.size == (1, 1):
                    lhs = lhs[0]

                return self._common_basetype(other)(
                    string, shape, {mtbs: lhs*rhs for mtbs, rhs in RHS.items()})
            elif n == p:
                # Regular matrix multiplication already works.
                pass
            elif m*n == 1:
                # Prepare a regular matrix multiplication.
                # HACK: Replaces self.
                self = self*cvxopt.spdiag([1]*p)
                m, n = self._shape
            else:
                # Prepare a regular matrix multiplication.
                assert p*q == 1
                other = other*cvxopt.spdiag([1]*n)
                p, q = other._shape

        assert n == p

        # Handle the common row by column multiplication more efficiently.
        # This includes some scalar by scalar products (both sides nonconstant).
        row_by_column = m == 1 and q == 1

        # Regular matrix multiplication.
        coefs = {}
        for (lmtbs, lcoef), (rmtbs, rcoef) in product(
                self._coefs.items(), other._coefs.items()):
            if len(lmtbs) + len(rmtbs) > 2:
                raise NotImplementedError("Product not supported: "
                    "One operand is biaffine and the other nonconstant.")
            elif len(lmtbs) == 0:
                # Constant by constant, linear or bilinear.
                if row_by_column:
                    factor = lcoef.T
                else:
                    # Use identity vec(AB) = (I ⊗ A)vec(B).
                    factor = left_kronecker_I(lcoef, q, reshape=(m, n))

                self._update_coefs(coefs, rmtbs, factor*rcoef)
            elif len(rmtbs) == 0:
                # Constant, linear or bilinear by constant.
                if row_by_column:
                    factor = rcoef.T
                else:
                    # Use identity vec(AB) = (Bᵀ ⊗ I)vec(A).
                    factor = right_kronecker_I(
                        rcoef, m, reshape=(p, q), postT=True)

                self._update_coefs(coefs, lmtbs, factor*lcoef)
            else:
                # Linear by linear.
                assert len(lmtbs) == 1 and len(rmtbs) == 1

                if row_by_column:
                    # Use identity (Ax)ᵀ(By) = vec(AᵀB)ᵀ·vec(xyᵀ).
                    coef = (lcoef.T*rcoef)[:].T
                else:
                    # Recipe found in "Robust conic optimization in Python"
                    # (Stahlberg 2020).
                    a, b = lmtbs[0].dim, rmtbs[0].dim

                    A = cvxopt_K(m, n)*lcoef
                    B = rcoef

                    A = A.T
                    B = B.T
                    A.size = m*n*a, 1
                    B.size = p*q*b, 1

                    A = cvxopt.spdiag([cvxopt_K(a, n)]*m)*A
                    B = cvxopt.spdiag([cvxopt_K(b, p)]*q)*B
                    A.size = n, m*a
                    B.size = p, q*b
                    A = A.T

                    C = A*B
                    C.size = a, m*q*b
                    C = C*cvxopt.spdiag([cvxopt_K(b, m)]*q)
                    C.size = a*b, m*q
                    C = C.T

                    coef = C

                # Forbid quadratic results.
                if coef and lmtbs[0] is rmtbs[0]:
                    raise TypeError("The product of {} and {} is quadratic "
                        "in {} but a biaffine result is required here."
                        .format(self.string, other.string, lmtbs[0].name))

                self._update_coefs(coefs, lmtbs + rmtbs, coef)

        return self._common_basetype(other)(string, (m, q), coefs)

    @convert_operands(lMatMul=True)
    @refine_operands(stop_at_affine=True)
    def __rmul__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__rmul__(self, other)

        return other.__mul__(self)

    @convert_operands(sameShape=True)
    @refine_operands(stop_at_affine=True)
    def __or__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__or__(self, other)

        result = other.vec.H * self.vec
        result._symbStr = glyphs.clever_dotp(
            self.string, other.string, other.complex, self.scalar)
        return result

    @convert_operands(sameShape=True)
    @refine_operands(stop_at_affine=True)
    def __ror__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__ror__(self, other)

        return other.__or__(self)

    @convert_operands(sameShape=True)
    @refine_operands(stop_at_affine=True)
    def __xor__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__xor__(self, other)

        string = glyphs.hadamard(self.string, other.string)

        if other.constant:
            factor = cvxopt.spdiag(other._constant_coef)
            coefs  = {mtbs: factor*coef for mtbs, coef in self._coefs.items()}
        elif self.constant:
            factor = cvxopt.spdiag(self._constant_coef)
            coefs  = {mtbs: factor*coef for mtbs, coef in other._coefs.items()}
        else:
            return (self.diag*other.vec).reshaped(self._shape).renamed(string)

        return self._common_basetype(other)(string, self._shape, coefs)

    @convert_operands(sameShape=True)
    @refine_operands(stop_at_affine=True)
    def __rxor__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__rxor__(self, other)

        return other.__xor__(self)

    @convert_operands()
    @refine_operands(stop_at_affine=True)
    def __matmul__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__matmul__(self, other)

        cls = self._common_basetype(other)
        tc = cls._get_typecode()

        string = glyphs.kron(self.string, other.string)

        m, n  = self._shape
        p, q  = other._shape

        # Recipe in "Robust conic optimization in Python" (Stahlberg 2020).
        Kqm = cvxopt_K(q, m)
        Kqm_Ip = right_kronecker_I(Kqm, p)
        In_Kqm_Ip = left_kronecker_I(Kqm_Ip, n)

        def _kron(A, B):
            A_B = load_data(numpy.kron(cvx2np(A), cvx2np(B)), typecode=tc)[0]
            Kab = cvxopt_K(A.size[1], B.size[1])
            return In_Kqm_Ip*A_B*Kab

        coefs = {}
        for (lmtbs, lcoef), (rmtbs, rcoef) in product(
                self._coefs.items(), other._coefs.items()):
            if len(lmtbs) + len(rmtbs) <= 2:
                if len(lmtbs) == 1 and len(rmtbs) == 1 and lmtbs[0] is rmtbs[0]:
                    raise TypeError("The product of {} and {} is quadratic "
                        "in {} but a biaffine result is required here."
                        .format(self.string, other.string, lmtbs[0].name))

                self._update_coefs(coefs, lmtbs + rmtbs, _kron(lcoef, rcoef))
            else:
                raise NotImplementedError("Product not supported: "
                    "One operand is biaffine and the other nonconstant.")

        return cls(string, (m*p, n*q), coefs)

    @convert_operands()
    @refine_operands(stop_at_affine=True)
    def __rmatmul__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__rmatmul__(self, other)

        return other.__matmul__(self)

    @deprecated("2.2", "Use infix @ instead.")
    def kron(self, other):
        """Denote the Kronecker product with another expression on the right."""
        return self.__matmul__(other)

    @deprecated("2.2", "Reverse operands and use infix @ instead.")
    def leftkron(self, other):
        """Denote the Kronecker product with another expression on the left."""
        return self.__rmatmul__(other)

    @convert_operands(scalarRHS=True)
    @refine_operands(stop_at_affine=True)
    def __truediv__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__truediv__(self, other)

        if not other.constant:
            raise TypeError(
                "You may only divide {} by a constant.".format(self.string))

        if other.is0:
            raise ZeroDivisionError(
                "Tried to divide {} by zero.".format(self.string))

        divisor = other._constant_coef[0]

        string = glyphs.div(self.string, other.string)
        coefs  = {mtbs: coef / divisor for mtbs, coef in self._coefs.items()}

        return self._common_basetype(other)(string, self._shape, coefs)

    @convert_operands(scalarLHS=True)
    @refine_operands(stop_at_affine=True)
    def __rtruediv__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__rtruediv__(self, other)

        return other.__truediv__(self)

    @convert_operands(horiCat=True)
    @refine_operands(stop_at_affine=True)
    def __and__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__and__(self, other)

        string = glyphs.matrix_cat(self.string, other.string, horizontal=True)
        shape  = (self._shape[0], self._shape[1] + other._shape[1])

        coefs = {}
        for mtbs in set(self._coefs.keys()).union(other._coefs.keys()):
            l = self._coefs[mtbs]  if mtbs in self._coefs  else cvxopt.spmatrix(
                [], [], [], (len(self), other._coefs[mtbs].size[1]))
            r = other._coefs[mtbs] if mtbs in other._coefs else cvxopt.spmatrix(
                [], [], [], (len(other), self._coefs[mtbs].size[1]))

            coefs[mtbs] = cvxopt_vcat([l, r])

        return self._common_basetype(other)(string, shape, coefs)

    @convert_operands(horiCat=True)
    @refine_operands(stop_at_affine=True)
    def __rand__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__rand__(self, other)

        return other.__and__(self)

    @convert_operands(vertCat=True)
    @refine_operands(stop_at_affine=True)
    def __floordiv__(self, other):
        def interleave_columns(upper, lower, upperRows, lowerRows, cols):
            p, q = upperRows, lowerRows
            return [column for columnPairs in [
                (upper[j*p:j*p+p, :], lower[j*q:j*q+q, :]) for j in range(cols)]
                for column in columnPairs]

        if not isinstance(other, BiaffineExpression):
            return Expression.__floordiv__(self, other)

        string = glyphs.matrix_cat(self.string, other.string, horizontal=False)

        p, q, c = self._shape[0], other._shape[0], self._shape[1]
        shape = (p + q, c)

        coefs = {}
        for mtbs in set(self._coefs.keys()).union(other._coefs.keys()):
            u = self._coefs[mtbs]  if mtbs in self._coefs  else cvxopt.spmatrix(
                [], [], [], (len(self), other._coefs[mtbs].size[1]))
            l = other._coefs[mtbs] if mtbs in other._coefs else cvxopt.spmatrix(
                [], [], [], (len(other), self._coefs[mtbs].size[1]))

            coefs[mtbs] = cvxopt_vcat(interleave_columns(u, l, p, q, c))

        return self._common_basetype(other)(string, shape, coefs)

    @convert_operands(vertCat=True)
    @refine_operands(stop_at_affine=True)
    def __rfloordiv__(self, other):
        if not isinstance(other, BiaffineExpression):
            return Expression.__rfloordiv__(self, other)

        return other.__floordiv__(self)

    @convert_operands(scalarRHS=True)
    @refine_operands()  # Refine both sides to real if possible.
    def __pow__(self, other):
        from .exp_powtrace import PowerTrace

        if not self.scalar:
            raise TypeError("May only exponentiate a scalar expression.")

        if not other.constant:
            raise TypeError("The exponent must be constant.")

        if other.complex:
            raise TypeError("The exponent must be real-valued.")

        exponent = other.value

        if exponent == 2:
            return (self | self)  # Works for complex base.
        else:
            return PowerTrace(self, exponent)  # Errors on complex base.

    # --------------------------------------------------------------------------
    # Properties and functions that describe the expression.
    # --------------------------------------------------------------------------

    @cached_property
    def hermitian(self):  # noqa (D402 thinks this includes a signature)
        """Whether the expression is a hermitian (or symmetric) matrix.

        Uses :data:`~picos.settings.RELATIVE_HERMITIANNESS_TOLERANCE`.

        If PICOS rejects your near-hermitian (near-symmetric) expression as not
        hermitian (not symmetric), you can use :meth:`hermitianized` to correct
        larger numeric errors or the effects of noisy data.
        """
        return self.equals(
            self.H, relTol=settings.RELATIVE_HERMITIANNESS_TOLERANCE)

    @property
    def is0(self):
        """Whether this is a constant scalar, vector or matrix of all zeros."""
        return not self._coefs

    @cached_property
    def is1(self):
        """Whether this is a constant scalar or vector of all ones."""
        # Must be a scalar or vector.
        if self._shape[0] != 1 and self._shape[1] != 1:
            return False

        # Must be constant.
        if not self.constant:
            return False

        # Constant term must be all ones.
        return not self._constant_coef - 1

    @cached_property
    def isI(self):
        """Whether this is a constant identity matrix."""
        m, n = self._shape

        # Must be a square matrix.
        if m != n:
            return False

        # Must be constant.
        if not self.constant:
            return False

        # Constant term must be the identity.
        return not self._constant_coef - cvxopt.spdiag([1]*m)[:]

    @cached_property
    def complex(self):
        """Whether the expression can be complex-valued."""
        # NOTE: The typecode check works around a bug in CVXOPT prior to 1.2.4.
        return any(coef.typecode == "z" and coef.imag()
            for coef in self._coefs.values())

    @property
    def isreal(self):
        """Whether the expression is always real-valued."""
        return not self.complex

    @convert_operands()
    def equals(self, other, absTol=None, relTol=None):
        """Check mathematical equality with another (bi)affine expression.

        The precise type of both (bi)affine expressions may differ. In
        particular, a :class:`~.exp_affine.ComplexAffineExpression` with real
        coefficients and constant term can be equal to an
        :class:`~.exp_affine.AffineExpression`.

        If the operand is not already a PICOS expression, an attempt is made to
        load it as a constant affine expression. In this case, no reshaping or
        broadcasting is used to bring the constant term to the same shape as
        this expression. In particular,

            - ``0`` refers to a scalar zero (see also :meth:`is0`),
            - lists and tuples are treated as column vectors and
            - algebraic strings must specify a shape (see
              :func:`~.data.load_data`).

        :param other: Another PICOS expression or a constant numeric data value
            supported by :func:`~.data.load_data`.
        :param absTol: As long as all absolute differences between scalar
            entries of the coefficient matrices and the constant terms being
            compared does not exceed this bound, consider the expressions equal.
        :param relTol: As long as all absolute differences between scalar
            entries of the coefficient matrices and the constant terms being
            compared divided by the maximum absolute value found in either term
            does not exceed this bound, consider the expressions equal.

        :Example:

        >>> from picos import Constant
        >>> A = Constant("A", 0, (5,5))
        >>> repr(A)
        '<5×5 Real Constant: A>'
        >>> A.is0
        True
        >>> A.equals(0)
        False
        >>> A.equals("|0|(5,5)")
        True
        >>> repr(A*1j)
        '<5×5 Complex Constant: A·1j>'
        >>> A.equals(A*1j)
        True
        """
        if self is other:
            return True

        if not isinstance(other, BiaffineExpression):
            return False

        if self._shape != other._shape:
            return False

        assert not any((mtbs[1], mtbs[0]) in other._bilinear_coefs
                for mtbs in self._bilinear_coefs), \
            "{} and {} store bilinear terms in a different order." \
            .format(self.string, other.string)

        for mtbs in other._coefs:
            if mtbs not in self._coefs:
                return False

        for mtbs, coef in self._coefs.items():
            if mtbs not in other._coefs:
                return False

            if not cvxopt_equals(coef, other._coefs[mtbs], absTol, relTol):
                return False

        return True

    # --------------------------------------------------------------------------
    # Methods and properties that return modified copies.
    # --------------------------------------------------------------------------

    def renamed(self, string):
        """Return the expression with a modified string description."""
        return self._basetype(string, self._shape, self._coefs)

    def reshaped(self, shape, order="F"):
        r"""Return the expression reshaped in the given order.

        The default indexing order is column-major. Given an :math:`m \times n`
        matrix, reshaping in the default order is a constant time operation
        while reshaping in row-major order requires :math:`O(mn)` time. However,
        the latter allows you to be consistent with NumPy, which uses C-order (a
        generalization of row-major) by default.

        :param str order:
            The indexing order to use when reshaping. Must be either ``"F"`` for
            Fortran-order (column-major) or ``"C"`` for C-order (row-major).

        :Example:

        >>> from picos import Constant
        >>> C = Constant("C", range(6), (2, 3))
        >>> print(C)
        [ 0.00e+00  2.00e+00  4.00e+00]
        [ 1.00e+00  3.00e+00  5.00e+00]
        >>> print(C.reshaped((3, 2)))
        [ 0.00e+00  3.00e+00]
        [ 1.00e+00  4.00e+00]
        [ 2.00e+00  5.00e+00]
        >>> print(C.reshaped((3, 2), order="C"))
        [ 0.00e+00  2.00e+00]
        [ 4.00e+00  1.00e+00]
        [ 3.00e+00  5.00e+00]
        """
        if order not in "FC":
            raise ValueError("Order must be given as 'F' or 'C'.")

        shape = load_shape(shape, wildcards=True)

        if shape == self._shape:
            return self
        elif shape == (None, None):
            return self

        length = len(self)

        if shape[0] is None:
            shape = (length // shape[1], shape[1])
        elif shape[1] is None:
            shape = (shape[0], length // shape[0])

        if shape[0]*shape[1] != length:
            raise ValueError("Can only reshape to a matrix of same size.")

        if order == "F":
            coefs = self._coefs
            reshaped_glyph = glyphs.reshaped
        else:
            m, n = self._shape
            p, q = shape

            K_old = cvxopt_K(m, n, self._typecode)
            K_new = cvxopt_K(q, p, self._typecode)
            R = K_new * K_old

            coefs = {mtbs: R * coef for mtbs, coef in self._coefs.items()}
            reshaped_glyph = glyphs.reshaprm

        string = reshaped_glyph(self.string, glyphs.shape(shape))
        return self._basetype(string, shape, coefs)

    def broadcasted(self, shape):
        """Return the expression broadcasted to the given shape.

        :Example:

        >>> from picos import Constant
        >>> C = Constant("C", range(6), (2, 3))
        >>> print(C)
        [ 0.00e+00  2.00e+00  4.00e+00]
        [ 1.00e+00  3.00e+00  5.00e+00]
        >>> print(C.broadcasted((6, 6)))
        [ 0.00e+00  2.00e+00  4.00e+00  0.00e+00  2.00e+00  4.00e+00]
        [ 1.00e+00  3.00e+00  5.00e+00  1.00e+00  3.00e+00  5.00e+00]
        [ 0.00e+00  2.00e+00  4.00e+00  0.00e+00  2.00e+00  4.00e+00]
        [ 1.00e+00  3.00e+00  5.00e+00  1.00e+00  3.00e+00  5.00e+00]
        [ 0.00e+00  2.00e+00  4.00e+00  0.00e+00  2.00e+00  4.00e+00]
        [ 1.00e+00  3.00e+00  5.00e+00  1.00e+00  3.00e+00  5.00e+00]
        """
        shape = load_shape(shape, wildcards=True)
        shape = blend_shapes(shape, self._shape)

        if shape == self._shape:
            return self

        vdup = shape[0] // self._shape[0]
        hdup = shape[1] // self._shape[1]

        if (self._shape[0] * vdup, self._shape[1] * hdup) != shape:
            raise ValueError("Cannot broadcast from shape {} to {}."
                .format(glyphs.shape(self._shape), glyphs.shape(shape)))

        if self._shape == (1, 1):
            string = glyphs.matrix(self.string)
            return (self * cvxopt.matrix(1.0, shape)).renamed(string)

        string = glyphs.bcasted(self.string, glyphs.shape(shape))
        return (cvxopt.matrix(1.0, (vdup, hdup)) @ self).renamed(string)

    def reshaped_or_broadcasted(self, shape):
        """Return the expression :meth:`reshaped` or :meth:`broadcasted`.

        Unlike with :meth:`reshaped` and :meth:`broadcasted`, the target shape
        may not contain a wildcard character.

        If the wildcard-free target shape has the same number of elements as
        the current shape, then this is the same as :meth:`reshaped`, otherwise
        it is the same as :meth:`broadcasted`.
        """
        shape = load_shape(shape, wildcards=False)

        try:
            if shape[0]*shape[1] == len(self):
                return self.reshaped(shape)
            else:
                return self.broadcasted(shape)
        except ValueError:
            raise ValueError(
                "Cannot reshape or broadcast from shape {} to {}.".format(
                glyphs.shape(self._shape), glyphs.shape(shape))) from None

    @cached_property
    def hermitianized(self):
        r"""The expression projected onto the subspace of hermitian matrices.

        For a square (complex) affine expression :math:`A`, this is
        :math:`\frac{1}{2}(A + A^H)`.

        If the expression is not complex, then this is a projection onto the
        subspace of symmetric matrices.
        """
        if not self.square:
            raise TypeError("Cannot hermitianize non-square {}.".format(self))

        return (self + self.H)/2

    @cached_property
    def real(self):
        """Real part of the expression."""
        return self._basetype(glyphs.real(self.string), self._shape,
            {mtbs: coef.real() for mtbs, coef in self._coefs.items()})

    @cached_property
    def imag(self):
        """Imaginary part of the expression."""
        return self._basetype(glyphs.imag(self.string), self._shape,
            {mtbs: coef.imag() for mtbs, coef in self._coefs.items()})

    @cached_property
    def bilin(self):
        """Pure bilinear part of the expression."""
        return self._basetype(
            glyphs.blinpart(self._symbStr), self._shape, self._bilinear_coefs)

    @cached_property
    def lin(self):
        """Linear part of the expression."""
        return self._basetype(
            glyphs.linpart(self._symbStr), self._shape, self._linear_coefs)

    @cached_property
    def noncst(self):
        """Nonconstant part of the expression."""
        coefs = {mtbs: coefs for mtbs, coefs in self._coefs.items() if mtbs}

        return self._basetype(
            glyphs.ncstpart(self._symbStr), self._shape, coefs)

    @cached_property
    def cst(self):
        """Constant part of the expression."""
        coefs = {(): self._coefs[()]} if () in self._coefs else {}

        return self._basetype(glyphs.cstpart(self._symbStr), self._shape, coefs)

    @cached_selfinverse_property
    def T(self):
        """Matrix transpose."""
        if len(self) == 1:
            return self

        m, n = self._shape
        K = cvxopt_K(m, n, self._typecode)

        string = glyphs.transp(self.string)
        shape  = (self._shape[1], self._shape[0])
        coefs  = {mtbs: K * coef for mtbs, coef in self._coefs.items()}

        return self._basetype(string, shape, coefs)

    @cached_selfinverse_property
    def conj(self):
        """Complex conjugate."""
        string = glyphs.conj(self.string)
        coefs  = {mtbs: coef.H.T for mtbs, coef in self._coefs.items()}

        return self._basetype(string, self._shape, coefs)

    @cached_selfinverse_property
    def H(self):
        """Conjugate (or Hermitian) transpose."""
        return self.T.conj.renamed(glyphs.htransp(self._symbStr))

    def _square_equal_subsystem_dims(self, diagLen):
        """Support :func:`partial_trace` and :func:`partial_transpose`."""
        m, n = self._shape
        k = math.log(m, diagLen)

        if m != n or int(k) != k:
            raise TypeError("The expression has shape {} so it cannot be "
                "decomposed into subsystems of shape {}.".format(
                glyphs.shape(self._shape), glyphs.shape((diagLen,)*2)))

        return ((diagLen,)*2,)*int(k)

    def partial_transpose(self, subsystems, dimensions=2):
        r"""Return the expression with selected subsystems transposed.

        If the expression can be written as
        :math:`A_0 \otimes \cdots \otimes A_{n-1}` for matrices
        :math:`A_0, \ldots, A_{n-1}` with shapes given in ``dimensions``, then
        this returns :math:`B_0 \otimes \cdots \otimes B_{n-1}` with
        :math:`B_i = A_i^T`, if ``i in subsystems`` (with :math:`i = -1` read as
        :math:`n-1`), and :math:`B_i = A_i`, otherwise.

        :param subsystems: A collection of or a single subystem number, indexed
            from zero, corresponding to subsystems that shall be transposed.
            The value :math:`-1` refers to the last subsystem.
        :type subsystems: int or tuple or list

        :param dimensions: Either an integer :math:`d` so that the subsystems
            are assumed to be all of shape :math:`d \times d`, or a sequence of
            subsystem shapes where an integer :math:`d` within the sequence is
            read as :math:`d \times d`. In any case, the elementwise product
            over all subsystem shapes must equal the expression's shape.
        :type dimensions: int or tuple or list

        :raises TypeError: If the subsystems do not match the expression.
        :raises IndexError: If the subsystem selection is invalid.

        :Example:

        >>> from picos import Constant
        >>> A = Constant("A", range(16), (4, 4))
        >>> print(A) #doctest: +NORMALIZE_WHITESPACE
        [ 0.00e+00  4.00e+00  8.00e+00  1.20e+01]
        [ 1.00e+00  5.00e+00  9.00e+00  1.30e+01]
        [ 2.00e+00  6.00e+00  1.00e+01  1.40e+01]
        [ 3.00e+00  7.00e+00  1.10e+01  1.50e+01]
        >>> A0 = A.partial_transpose(0); A0
        <4×4 Real Constant: A.{[2×2]ᵀ⊗[2×2]}>
        >>> print(A0) #doctest: +NORMALIZE_WHITESPACE
        [ 0.00e+00  4.00e+00  2.00e+00  6.00e+00]
        [ 1.00e+00  5.00e+00  3.00e+00  7.00e+00]
        [ 8.00e+00  1.20e+01  1.00e+01  1.40e+01]
        [ 9.00e+00  1.30e+01  1.10e+01  1.50e+01]
        >>> A1 = A.partial_transpose(1); A1
        <4×4 Real Constant: A.{[2×2]⊗[2×2]ᵀ}>
        >>> print(A1) #doctest: +NORMALIZE_WHITESPACE
        [ 0.00e+00  1.00e+00  8.00e+00  9.00e+00]
        [ 4.00e+00  5.00e+00  1.20e+01  1.30e+01]
        [ 2.00e+00  3.00e+00  1.00e+01  1.10e+01]
        [ 6.00e+00  7.00e+00  1.40e+01  1.50e+01]
        """
        m, n = self._shape

        if isinstance(dimensions, int):
            dimensions = self._square_equal_subsystem_dims(dimensions)
        else:
            dimensions = [
                (d, d) if isinstance(d, int) else d for d in dimensions]

        if reduce(
                lambda x, y: (x[0]*y[0], x[1]*y[1]), dimensions) != self._shape:
            raise TypeError("Subsystem dimensions do not match expression.")

        if isinstance(subsystems, int):
            subsystems = (subsystems,)
        elif not subsystems:
            return self

        numSys     = len(dimensions)
        subsystems = set(numSys - 1 if sys == -1 else sys for sys in subsystems)

        for sys in subsystems:
            if not isinstance(sys, int):
                raise IndexError("Subsystem indices must be integer, not {}."
                    .format(type(sys).__name__))
            elif sys < 0:
                raise IndexError("Subsystem indices must be nonnegative.")
            elif sys >= numSys:
                raise IndexError(
                    "Subsystem index {} out of range for {} systems total."
                    .format(sys, numSys))

        # If all subsystems are transposed, this is regular transposition.
        if len(subsystems) == numSys:
            return self.T

        # Prepare sparse K such that K·vec(A) = vec(partial_transpose(A)).
        d = m * n
        V = [1]*d
        I = range(d)
        J = cvxopt.matrix(I)
        T = cvxopt.matrix(0, J.size)
        obh, obw   = 1, 1
        sysStrings = None

        # Apply transpositions starting with the rightmost Kronecker factor.
        for sys in range(numSys - 1, -1, -1):
            # Shape of current system.
            p, q = dimensions[sys]
            sysString = glyphs.matrix(glyphs.shape((p, q)))

            # Height/width of "inner" blocks being moved, initially scalars.
            ibh, ibw = obh, obw

            # Heigh/width of "outer" blocks whose relative position is
            # maintained but that are subject to transposition independently.
            # In the last iteration this is the shape of the resulting matrix.
            obh *= p
            obw *= q

            # Only transpose selected subsystems.
            if sys not in subsystems:
                sysStrings = glyphs.kron(sysString, sysStrings) \
                    if sysStrings else sysString
                continue
            else:
                sysStrings = glyphs.kron(glyphs.transp(sysString), sysStrings) \
                    if sysStrings else glyphs.transp(sysString)

            # Shape of outer blocks after transposition.
            obhT, obwT = obw // ibw * ibh, obh // ibh * ibw

            # Shape of full matrix after transposition.
            mT, nT = m // obh * obhT, n // obw * obwT

            for vi in I:
                # Full matrix column and row.
                c, r = divmod(vi, m)

                # Outer block vertical   index and row    within outer block,
                # outer block horizontal index and column within outer block.
                obi, obr = divmod(r, obh)
                obj, obc = divmod(c, obw)

                # Inner block vertical   index and row    within inner block,
                # inner block horizontal index and column within inner block.
                ibi, ibr = divmod(obr, ibh)
                ibj, ibc = divmod(obc, ibw)

                # (1) ibi*ibw + ibc is column within the transposed outer block;
                # adding obj*obwT yields the column in the transposed matrix.
                # (2) ibj*ibh + ibr is row within the transposed outer block;
                # adding obi*obhT yields the row in the transposed matrix.
                # (3) tvi is index within the vectorized transposed matrix.
                tvi = (obj*obwT + ibi*ibw + ibc)*mT \
                    + (obi*obhT + ibj*ibh + ibr)

                # Prepare the transposition.
                T[tvi] = J[vi]

            # Apply the transposition.
            J, T = T, J
            m, n, obh, obw = mT, nT, obhT, obwT

        # Finalize the partial commutation matrix.
        K = cvxopt.spmatrix(V, I, J, (d, d), self._typecode)

        string = glyphs.ptransp_(self.string, sysStrings)
        shape  = (m, n)
        coefs  = {mtbs: K * coef for mtbs, coef in self._coefs.items()}

        return self._basetype(string, shape, coefs)

    @cached_property
    def T0(self):
        r"""Expression with the first :math:`2 \times 2` subsystem transposed.

        Only available for a :math:`2^k \times 2^k` matrix with all subsystems
        of shape :math:`2 \times 2`. Use :meth:`partial_transpose` otherwise.
        """
        return self.partial_transpose(subsystems=0)

    @cached_property
    def T1(self):
        r"""Expression with the second :math:`2 \times 2` subsystem transposed.

        Only available for a :math:`2^k \times 2^k` matrix with all subsystems
        of shape :math:`2 \times 2`. Use :meth:`partial_transpose` otherwise.
        """
        return self.partial_transpose(subsystems=1)

    @cached_property
    def T2(self):
        r"""Expression with the third :math:`2 \times 2` subsystem transposed.

        Only available for a :math:`2^k \times 2^k` matrix with all subsystems
        of shape :math:`2 \times 2`. Use :meth:`partial_transpose` otherwise.
        """
        return self.partial_transpose(subsystems=2)

    @cached_property
    def T3(self):
        r"""Expression with the fourth :math:`2 \times 2` subsystem transposed.

        Only available for a :math:`2^k \times 2^k` matrix with all subsystems
        of shape :math:`2 \times 2`. Use :meth:`partial_transpose` otherwise.
        """
        return self.partial_transpose(subsystems=3)

    @cached_property
    def Tl(self):
        r"""Expression with the last :math:`2 \times 2` subsystem transposed.

        Only available for a :math:`2^k \times 2^k` matrix with all subsystems
        of shape :math:`2 \times 2`. Use :meth:`partial_transpose` otherwise.
        """
        return self.partial_transpose(subsystems=-1)

    @staticmethod
    def _reindex_F(indices, source, destination):
        """Convert indices between different tensor shapes in Fortran-order."""
        new = []
        offset = 0
        factor = 1

        for index, base in zip(indices, source):
            offset += factor*index
            factor *= base

        for base in destination:
            offset, remainder = divmod(offset, base)
            new.append(remainder)

        return tuple(new)

    @staticmethod
    def _reindex_C(indices, source, destination):
        """Convert indices between different tensor shapes in C-order."""
        new = []
        offset = 0
        factor = 1

        for index, base in zip(reversed(indices), reversed(source)):
            offset += factor*index
            factor *= base

        for base in reversed(destination):
            offset, remainder = divmod(offset, base)
            new.insert(0, remainder)

        return tuple(new)

    def reshuffled(self, permutation="ikjl", dimensions=None, order="C"):
        """Return the reshuffled or realigned expression.

        This operation works directly on matrices. However, it is equivalent to
        the following sequence of operations:

        1. The matrix is reshaped to a tensor with the given ``dimensions`` and
           according to ``order``.
        2. The tensor's axes are permuted according to ``permutation``.
        3. The tensor is reshaped back to the shape of the original matrix
           according to ``order``.

        For comparison, the following function applies the same operation to a
        2D NumPy :class:`~numpy:numpy.ndarray`:

        .. code::

            def reshuffle_numpy(matrix, permutation, dimensions, order):
                P = "{} -> {}".format("".join(sorted(permutation)), permutation)
                reshuffled = numpy.reshape(matrix, dimensions, order)
                reshuffled = numpy.einsum(P, reshuffled)
                return numpy.reshape(reshuffled, matrix.shape, order)

        :param permutation:
            A sequence of comparable elements with length equal to the number of
            tensor dimensions. The sequence is compared to its ordered version
            and the resulting permutation pattern is used to permute the tensor
            indices. For instance, the string ``"ikjl"`` is compared to its
            sorted version ``"ijkl"`` and denotes that the second and third axis
            should be swapped.
        :type permutation:
            str or tuple or list

        :param dimensions:
            If this is an integer sequence, then it defines the dimensions of
            the tensor. If this is :obj:`None`, then the tensor is assumed to be
            hypercubic and the number of dimensions is inferred from the
            ``permutation`` argument.
        :type dimensions:
            None or tuple or list

        :param str order:
            The indexing order to use for the virtual reshaping. Must be either
            ``"F"`` for Fortran-order (generalization of column-major) or
            ``"C"`` for C-order (generalization of row-major). Note that PICOS
            usually reshapes in Fortran-order while NumPy defaults to C-order.

        :Example:

        >>> from picos import Constant
        >>> A = Constant("A", range(16), (4, 4))
        >>> print(A) #doctest: +NORMALIZE_WHITESPACE
        [ 0.00e+00  4.00e+00  8.00e+00  1.20e+01]
        [ 1.00e+00  5.00e+00  9.00e+00  1.30e+01]
        [ 2.00e+00  6.00e+00  1.00e+01  1.40e+01]
        [ 3.00e+00  7.00e+00  1.10e+01  1.50e+01]
        >>> R = A.reshuffled(); R
        <4×4 Real Constant: shuffled(A,ikjl,C)>
        >>> print(R) #doctest: +NORMALIZE_WHITESPACE
        [ 0.00e+00  4.00e+00  1.00e+00  5.00e+00]
        [ 8.00e+00  1.20e+01  9.00e+00  1.30e+01]
        [ 2.00e+00  6.00e+00  3.00e+00  7.00e+00]
        [ 1.00e+01  1.40e+01  1.10e+01  1.50e+01]
        >>> A.reshuffled("ji").equals(A.T)     # Regular transposition.
        True
        >>> A.reshuffled("3214").equals(A.T0)  # Partial transposition (1).
        True
        >>> A.reshuffled("1432").equals(A.T1)  # Partial transposition (2).
        True
        """
        m, n = self._shape
        mn = m*n

        # Load the permutation.
        ordered = sorted(permutation)
        P = dict(enumerate(ordered.index(element) for element in permutation))

        if len(set(P.values())) < len(P):
            raise ValueError("The sequence defining the permutation appears to "
                "contain duplicate elements.")

        assert not set(P.keys()).symmetric_difference(set(P.values()))

        numDims = len(P)

        # Load the dimensions.
        guessDimensions = dimensions is None

        if guessDimensions:
            dimensions = (int(mn**(1.0 / numDims)),)*numDims
        else:
            if len(dimensions) != numDims:
                raise ValueError("The number of indices does not match the "
                    "number of dimensions.")

        if reduce(int.__mul__, dimensions, 1) != mn:
            raise TypeError("The {} matrix {} cannot be reshaped to a {} "
                "tensor.".format(glyphs.shape(self.shape), self.string,
                "hypercubic order {}".format(numDims) if guessDimensions
                else glyphs.size("", "").join(str(d) for d in dimensions)))

        # Load the indexing order.
        if order not in "FC":
            raise ValueError("Order must be given as 'F' or 'C'.")

        reindex = self._reindex_F if order == "F" else self._reindex_C

        # Nothing to do for the neutral permutation.
        if all(key == val for key, val in P.items()):
            return self

        # Create a sparse mtrix R such that R·vec(A) = vec(reshuffle(A)).
        V, I, J = [1]*mn, [], range(mn)
        for i in range(mn):
            (k, j) = divmod(i, m)  # (j, k) are column-major matrix indices.
            indices = reindex((j, k), (m, n), dimensions)
            newIndices = tuple(indices[P[d]] for d in range(numDims))
            newDimensions = tuple(dimensions[P[d]] for d in range(numDims))
            (j, k) = reindex(newIndices, newDimensions, (m, n))
            I.append(k*m + j)
        R = cvxopt.spmatrix(V, I, J, (mn, mn), self._typecode)

        # Create the string.
        strArgs = [self.string, str(permutation).replace(" ", ""), order]

        if not guessDimensions:
            strArgs.insert(2, str(dimensions).replace(" ", ""))

        string = glyphs.shuffled(",".join(strArgs))

        # Finalize the new expression.
        shape = (m, n)
        coefs = {mtbs: R * coef for mtbs, coef in self._coefs.items()}

        return self._basetype(string, shape, coefs)

    @cached_property
    def sum(self):
        """Sum over all scalar elements of the expression."""
        # NOTE: glyphs.clever_dotp detects this case and uses the sum glyph.
        # NOTE: 1 on the right hand side in case self is complex.
        return (self | 1)

    @cached_property
    def rowsum(self):
        """Sum over the rows of the expression as a row vector."""
        from .algebra import J

        return J(self.shape[0]).T * self

    @cached_property
    def colsum(self):
        """Sum over the columns of the expression as a column vector."""
        from .algebra import J

        return self * J(self.shape[1])

    @cached_property
    def tr(self):
        """Trace of a square expression."""
        if not self.square:
            raise TypeError("Cannot compute the trace of non-square {}."
                .format(self.string))

        # NOTE: glyphs.clever_dotp detects this case and uses the trace glyph.
        # NOTE: "I" on the right hand side in case self is complex.
        return (self | "I")

    def partial_trace(self, subsystems, dimensions=2):
        r"""Return the partial trace over selected subsystems.

        If the expression can be written as
        :math:`A_0 \otimes \cdots \otimes A_{n-1}` for matrices
        :math:`A_0, \ldots, A_{n-1}` with shapes given in ``dimensions``, then
        this returns :math:`B_0 \otimes \cdots \otimes B_{n-1}` with
        :math:`B_i = \operatorname{tr}(A_i)`, if ``i in subsystems`` (with
        :math:`i = -1` read as :math:`n-1`), and :math:`B_i = A_i`, otherwise.

        :param subsystems: A collection of or a single subystem number, indexed
            from zero, corresponding to subsystems that shall be traced over.
            The value :math:`-1` refers to the last subsystem.
        :type subsystems: int or tuple or list

        :param dimensions: Either an integer :math:`d` so that the subsystems
            are assumed to be all of shape :math:`d \times d`, or a sequence of
            subsystem shapes where an integer :math:`d` within the sequence is
            read as :math:`d \times d`. In any case, the elementwise product
            over all subsystem shapes must equal the expression's shape.
        :type dimensions: int or tuple or list

        :raises TypeError: If the subsystems do not match the expression or if
            a non-square subsystem is to be traced over.
        :raises IndexError: If the subsystem selection is invalid in any other
            way.

        :Example:

        >>> from picos import Constant
        >>> A = Constant("A", range(16), (4, 4))
        >>> print(A) #doctest: +NORMALIZE_WHITESPACE
        [ 0.00e+00  4.00e+00  8.00e+00  1.20e+01]
        [ 1.00e+00  5.00e+00  9.00e+00  1.30e+01]
        [ 2.00e+00  6.00e+00  1.00e+01  1.40e+01]
        [ 3.00e+00  7.00e+00  1.10e+01  1.50e+01]
        >>> A0 = A.partial_trace(0); A0
        <2×2 Real Constant: A.{tr([2×2])⊗[2×2]}>
        >>> print(A0) #doctest: +NORMALIZE_WHITESPACE
        [ 1.00e+01  1.80e+01]
        [ 1.20e+01  2.00e+01]
        >>> A1 = A.partial_trace(1); A1
        <2×2 Real Constant: A.{[2×2]⊗tr([2×2])}>
        >>> print(A1) #doctest: +NORMALIZE_WHITESPACE
        [ 5.00e+00  2.10e+01]
        [ 9.00e+00  2.50e+01]
        """
        # Shape of the original matrix.
        m, n = self._shape

        if isinstance(dimensions, int):
            dimensions = self._square_equal_subsystem_dims(dimensions)
        else:
            dimensions = [
                (d, d) if isinstance(d, int) else d for d in dimensions]

        if reduce(
                lambda x, y: (x[0]*y[0], x[1]*y[1]), dimensions) != self._shape:
            raise TypeError("Subsystem dimensions do not match expression.")

        if isinstance(subsystems, int):
            subsystems = (subsystems,)
        elif not subsystems:
            return self

        numSys     = len(dimensions)
        subsystems = set(numSys - 1 if sys == -1 else sys for sys in subsystems)

        for sys in subsystems:
            if not isinstance(sys, int):
                raise IndexError("Subsystem indices must be integer, not {}."
                    .format(type(sys).__name__))
            elif sys < 0:
                raise IndexError("Subsystem indices must be nonnegative.")
            elif sys >= numSys:
                raise IndexError(
                    "Subsystem index {} out of range for {} systems total."
                    .format(sys, numSys))
            elif dimensions[sys][0] != dimensions[sys][1]:
                raise TypeError(
                    "Subsystem index {} refers to a non-square subsystem that "
                    "cannot be traced over.".format(sys))

        # If all subsystems are traced over, this is the regular trace.
        if len(subsystems) == numSys:
            return self.tr

        # Prepare sparse T such that T·vec(A) = vec(partial_trace(A)).
        T = []

        # Compute factors of T, one for each subsystem being traced over.
        ibh, ibw   = m, n
        sysStrings = None
        for sys in range(numSys):
            # Shape of current system.
            p, q = dimensions[sys]
            sysString = glyphs.matrix(glyphs.shape((p, q)))

            # Heigh/width of "outer" blocks whose relative position is
            # maintained but that are reduced independently to the size of an
            # inner block if the current system is to be traced over.
            obh, obw = ibh, ibw

            # Height/width of "inner" blocks that are summed if they are
            # main-diagonal blocks of an outer block.
            ibh, ibw = obh // p, obw // q

            # Only trace over selected subsystems.
            if sys not in subsystems:
                sysStrings = glyphs.kron(sysStrings, sysString) \
                    if sysStrings else sysString
                continue
            else:
                sysStrings = glyphs.kron(sysStrings, glyphs.trace(sysString)) \
                    if sysStrings else glyphs.trace(sysString)

            # Shape of new matrix.
            assert p == q
            mN, nN = m // p, n // p

            # Prepare one factor of T.
            oldLen = m  * n
            newLen = mN * nN
            V, I, J = [1]*(newLen*p), [], []
            shape = (newLen, oldLen)

            # Determine the summands that make up each entry of the new matrix.
            for viN in range(newLen):
                # A column/row pair that represents a scalar in the new matrix
                # and a number of p scalars within different on-diagonal inner
                # blocks but within the same outer block in the old matrix.
                cN, rN = divmod(viN, mN)

                # Index pair (obi, obj) for outer block in question, row/column
                # pair (ibr, ibc) identifies the scalar within each inner block.
                obi, ibr = divmod(rN, ibh)
                obj, ibc = divmod(cN, ibw)

                # Collect summands for the current entry of the new matrix; one
                # scalar per on-diagonal inner block.
                for k in range(p):
                    rO = obi*obh + k*ibh + ibr
                    cO = obj*obw + k*ibw + ibc
                    I.append(viN)
                    J.append(cO*m + rO)

            # Store the operator that performs the current subsystem trace.
            T.insert(0, cvxopt.spmatrix(V, I, J, shape, self._typecode))

            # Make next iteration work on the new matrix.
            m, n = mN, nN

        # Multiply out the linear partial trace operator T.
        # TODO: Fast matrix multiplication dynamic program useful here?
        T = reduce(lambda A, B: A*B, T)

        string = glyphs.ptrace_(self.string, sysStrings)
        shape  = (m, n)
        coefs  = {mtbs: T * coef for mtbs, coef in self._coefs.items()}

        return self._basetype(string, shape, coefs)

    @cached_property
    def tr0(self):
        r"""Expression with the first :math:`2 \times 2` subsystem traced out.

        Only available for a :math:`2^k \times 2^k` matrix with all subsystems
        of shape :math:`2 \times 2`. Use :meth:`partial_trace` otherwise.
        """
        return self.partial_trace(subsystems=0)

    @cached_property
    def tr1(self):
        r"""Expression with the second :math:`2 \times 2` subsystem traced out.

        Only available for a :math:`2^k \times 2^k` matrix with all subsystems
        of shape :math:`2 \times 2`. Use :meth:`partial_trace` otherwise.
        """
        return self.partial_trace(subsystems=1)

    @cached_property
    def tr2(self):
        r"""Expression with the third :math:`2 \times 2` subsystem traced out.

        Only available for a :math:`2^k \times 2^k` matrix with all subsystems
        of shape :math:`2 \times 2`. Use :meth:`partial_trace` otherwise.
        """
        return self.partial_trace(subsystems=2)

    @cached_property
    def tr3(self):
        r"""Expression with the fourth :math:`2 \times 2` subsystem traced out.

        Only available for a :math:`2^k \times 2^k` matrix with all subsystems
        of shape :math:`2 \times 2`. Use :meth:`partial_trace` otherwise.
        """
        return self.partial_trace(subsystems=3)

    @cached_property
    def trl(self):
        r"""Expression with the last :math:`2 \times 2` subsystem traced out.

        Only available for a :math:`2^k \times 2^k` matrix with all subsystems
        of shape :math:`2 \times 2`. Use :meth:`partial_trace` otherwise.
        """
        return self.partial_trace(subsystems=-1)

    @cached_property
    def vec(self):
        """Column-major vectorization of the expression as a column vector.

        .. note::
            Given an expression ``A``, ``A.vec`` and ``A[:]`` produce the same
            result (up to its string description) but ``A.vec`` is faster and
            its result is cached.

        :Example:

        >>> from picos import Constant
        >>> A = Constant("A", [[1, 2], [3, 4]])
        >>> A.vec.equals(A[:])
        True
        >>> A[:] is A[:]
        False
        >>> A.vec is A.vec
        True
        """
        if self._shape[1] == 1:
            return self
        else:
            return self._basetype(
                glyphs.vec(self.string), (len(self), 1), self._coefs)

    def dupvec(self, n):
        """Return a (repeated) column-major vectorization of the expression.

        :param int n: Number of times to duplicate the vectorization.

        :returns: A column vector.

        :Example:

        >>> from picos import Constant
        >>> A = Constant("A", [[1, 2], [3, 4]])
        >>> A.dupvec(1) is A.vec
        True
        >>> A.dupvec(3).equals(A.vec // A.vec // A.vec)
        True
        """
        if not isinstance(n, int):
            raise TypeError("Number of copies must be integer.")

        if n < 1:
            raise ValueError("Number of copies must be positive.")

        if n == 1:
            return self.vec
        else:
            string = glyphs.vec(glyphs.comma(self.string, n))
            shape  = (len(self)*n, 1)
            coefs  = {mtbs: cvxopt_vcat([coef]*n)
                for mtbs, coef in self._coefs.items()}

            return self._basetype(string, shape, coefs)

    @cached_property
    def trilvec(self):
        r"""Column-major vectorization of the lower triangular part.

        :returns:
            A column vector of all elements :math:`A_{ij}` that satisfy
            :math:`i \geq j`.

        .. note::

            If you want a row-major vectorization instead, write ``A.T.triuvec``
            instead of ``A.trilvec``.

        :Example:

        >>> from picos import Constant
        >>> A = Constant("A", [[1, 2], [3, 4], [5, 6]])
        >>> print(A)
        [ 1.00e+00  2.00e+00]
        [ 3.00e+00  4.00e+00]
        [ 5.00e+00  6.00e+00]
        >>> print(A.trilvec)
        [ 1.00e+00]
        [ 3.00e+00]
        [ 5.00e+00]
        [ 4.00e+00]
        [ 6.00e+00]
        """
        m, n = self._shape

        if n == 1:  # Column vector or scalar.
            return self
        elif m == 1:  # Row vector.
            return self[0]

        # Build a transformation D such that D·vec(A) = trilvec(A).
        rows = [j*m + i for j in range(n) for i in range(m) if i >= j]
        d    = len(rows)
        D    = cvxopt.spmatrix([1]*d, range(d), rows, (d, len(self)))

        string = glyphs.trilvec(self.string)
        shape  = (d, 1)
        coefs  = {mtbs: D*coef for mtbs, coef in self._coefs.items()}

        return self._basetype(string, shape, coefs)

    @cached_property
    def triuvec(self):
        r"""Column-major vectorization of the upper triangular part.

        :returns:
            A column vector of all elements :math:`A_{ij}` that satisfy
            :math:`i \leq j`.

        .. note::

            If you want a row-major vectorization instead, write ``A.T.trilvec``
            instead of ``A.triuvec``.

        :Example:

        >>> from picos import Constant
        >>> A = Constant("A", [[1, 2, 3], [4, 5, 6]])
        >>> print(A)
        [ 1.00e+00  2.00e+00  3.00e+00]
        [ 4.00e+00  5.00e+00  6.00e+00]
        >>> print(A.triuvec)
        [ 1.00e+00]
        [ 2.00e+00]
        [ 5.00e+00]
        [ 3.00e+00]
        [ 6.00e+00]
        """
        m, n = self._shape

        if m == 1:  # Row vector or scalar.
            return self
        elif n == 1:  # Column vector.
            return self[0]

        # Build a transformation D such that D·vec(A) = triuvec(A).
        rows = [j*m + i for j in range(n) for i in range(m) if i <= j]
        d    = len(rows)
        D    = cvxopt.spmatrix([1]*d, range(d), rows, (d, len(self)))

        string = glyphs.triuvec(self.string)
        shape  = (d, 1)
        coefs  = {mtbs: D*coef for mtbs, coef in self._coefs.items()}

        return self._basetype(string, shape, coefs)

    @cached_property
    def svec(self):
        """An isometric vectorization of a symmetric or hermitian expression.

        In the real symmetric case

            - the vectorization format is precisely the one define in [svec]_,
            - the vectorization is isometric and isomorphic, and
            - this is the same vectorization as used internally by the
              :class:`~.variables.SymmetricVariable` class.

        In the complex hermitian case

            - the same format is used, now resulting in a complex vector,
            - the vectorization is isometric but **not** isomorphic as there are
              guaranteed zeros in the imaginary part of the vector, and
            - this is **not** the same vectorization as the isomorphic,
              real-valued one used by :class:`~.variables.HermitianVariable`.

        The reverse operation is denoted by :attr:`desvec` in either case.

        :raises TypeError:
            If the expression is not square.

        :raises ValueError:
            If the expression is not hermitian.
        """
        if not self.square:
            raise TypeError("Can only compute svec for a square matrix, not for"
                " the {} expression {}.".format(self._shape, self.string))
        elif not self.hermitian:
            raise ValueError("Cannot compute svec for the non-hermitian "
                "expression {}.".format(self.string))

        vec = SymmetricVectorization(self._shape)
        V = vec._full2special

        string = glyphs.svec(self.string)

        if self.isreal:
            vec = SymmetricVectorization(self._shape)
            V = vec._full2special
            coefs = {mtbs: V*coef for mtbs, coef in self._coefs.items()}
            result = self._basetype(string, (vec.dim, 1), coefs)
        else:
            # NOTE: We need to reproduce svec using triuvec because, for numeric
            #       reasons, SymmetricVectorization averages the elements in the
            #       lower and upper triangular part. For symmetric matrices,
            #       this is equivalent to the formal definition of svec found in
            #       Datorro's book. For hermitian matrices however it is not.
            real_svec = self.real.svec
            imag_svec = 2**0.5 * 1j * self.imag.triuvec

            result = (real_svec + imag_svec).renamed(string)

        with unlocked_cached_properties(result):
            result.desvec = self

        return result

    @cached_property
    def desvec(self):
        r"""The reverse operation of :attr:`svec`.

        :raises TypeError:
            If the expression is not a vector or has a dimension outside the
            permissible set

            .. math::

                \left\{ \frac{n(n + 1)}{2}
                    \mid n \in \mathbb{Z}_{\geq 1} \right\}
                 = \left\{ n \in \mathbb{Z}_{\geq 1}
                    \mid \frac{1}{2} \left( \sqrt{8n + 1} - 1 \right)
                        \in \mathbb{Z}_{\geq 1} \right\}.

        :raises ValueError:
            In the case of a complex vector, If the vector is not in the image
            of :attr:`svec`.
        """
        if 1 not in self.shape:
            raise TypeError(
                "Can only compute desvec for a vector, not for the {} "
                "expression {}.".format(glyphs.shape(self._shape), self.string))

        n = 0.5*((8*len(self) + 1)**0.5 - 1)
        if int(n) != n:
            raise TypeError("Cannot compute desvec for the {}-dimensional "
                "vector {} as this size is not a possible outcome of svec."
                .format(len(self), self.string))
        n = int(n)

        vec = SymmetricVectorization((n, n))
        D = vec._special2full
        string = glyphs.desvec(self.string)

        if self.isreal:
            coefs = {mtbs: D*coef for mtbs, coef in self._coefs.items()}
            result = self._basetype(string, (n, n), coefs)

            assert result.hermitian
        else:
            # NOTE: While :attr:`svec` performs essentially the same operation
            #       for both symmetric and hermitian matrices, we now need to
            #       treat the imaginary separately as the imaginary part of a
            #       hermitian matrix is skew-symmetric instead of symmetric.
            V, I, J = [], D.I, D.J
            for v, i, j in zip(D.V, I, J):
                V.append(1j*v if i % n <= i // n else -1j*v)
            C = cvxopt.spmatrix(V, I, J, D.size, tc="z")

            real_desvec = self.real.desvec
            imag_desvec = self._basetype(string, (n, n), {
                mtbs: C*coef for mtbs, coef in self.imag._coefs.items()})

            result = (real_desvec + imag_desvec).renamed(string)

            if not result.hermitian:
                raise ValueError("The complex vector {} is not in the image of "
                    "svec. Note that svec is not bijective in the complex case."
                    .format(self.string))

        with unlocked_cached_properties(result):
            result.svec = self

        return result

    def dupdiag(self, n):
        """Return a matrix with the (repeated) expression on the diagonal.

        Vectorization is performed in column-major order.

        :param int n: Number of times to duplicate the vectorization.
        """
        from .algebra import I

        if self.scalar:
            return self * I(n)

        # Vectorize and duplicate the expression.
        vec = self.dupvec(n)
        d   = len(vec)

        # Build a transformation D such that D·vec(A) = vec(diag(vec(A))).
        ones = [1]*d
        D    = cvxopt.spdiag(ones)[:]
        D    = cvxopt.spmatrix(ones, D.I, range(d), (D.size[0], d))

        string = glyphs.diag(vec.string)
        shape  = (d, d)
        coefs  = {mtbs: D*coef for mtbs, coef in vec._coefs.items()}

        return self._basetype(string, shape, coefs)

    @cached_property
    def diag(self):
        """Diagonal matrix with the expression on the main diagonal.

        Vectorization is performed in column-major order.
        """
        from .algebra import O, I

        if self.is0:
            return O(len(self), len(self))
        elif self.is1:
            return I(len(self))
        else:
            return self.dupdiag(1)

    @cached_property
    def maindiag(self):
        """The main diagonal of the expression as a column vector."""
        if 1 in self._shape:
            return self[0]

        # Build a transformation D such that D·vec(A) = diag(A).
        step = self._shape[0] + 1
        rows = [i*step for i in range(min(self._shape))]
        d    = len(rows)
        D    = cvxopt.spmatrix([1]*d, range(d), rows, (d, len(self)))

        string = glyphs.maindiag(self.string)
        shape  = (d, 1)
        coefs  = {mtbs: D*coef for mtbs, coef in self._coefs.items()}

        return self._basetype(string, shape, coefs)

    def factor_out(self, mutable):
        r"""Factor out a single mutable from a vector expression.

        If this expression is a column vector :math:`a` that depends on some
        mutable :math:`x` with a trivial internal vectorization format (i.e.
        :class:`~.vectorizations.FullVectorization`), then this method, called
        with :math:`x` as its argument, returns a pair of expressions
        :math:`(a_x, a_0)` such that :math:`a = a_x\operatorname{vec}(x) + a_0`.

        :returns:
            Two refined :class:`BiaffineExpression` instances that do not depend
            on ``mutable``.

        :raises TypeError:
            If the expression is not a column vector or if ``mutable`` is not a
            :class:`~.mutable.Mutable` or does not have a trivial vectorization
            format.

        :raises LookupError:
            If the expression does not depend on ``mutable``.

        :Example:

        >>> from picos import RealVariable
        >>> from picos.uncertain import UnitBallPerturbationSet
        >>> x = RealVariable("x", 3)
        >>> z = UnitBallPerturbationSet("z", 3).parameter
        >>> a = ((2*x + 3)^z) + 4*x + 5; a
        <3×1 Uncertain Affine Expression: (2·x + [3])⊙z + 4·x + [5]>
        >>> sorted(a.mutables, key=lambda mtb: mtb.name)
        [<3×1 Real Variable: x>, <3×1 Perturbation: z>]
        >>> az, a0 = a.factor_out(z)
        >>> az
        <3×3 Real Affine Expression: ((2·x + [3])⊙z + 4·x + [5])_z>
        >>> a0
        <3×1 Real Affine Expression: ((2·x + [3])⊙z + 4·x + [5])_0>
        >>> sorted(az.mutables.union(a0.mutables), key=lambda mtb: mtb.name)
        [<3×1 Real Variable: x>]
        >>> (az*z + a0).equals(a)
        True
        """
        if self._shape[1] != 1:
            raise TypeError(
                "Can only factor out mutables from column vectors but {} has "
                "a shape of {}.".format(self.string, glyphs.shape(self._shape)))

        mtb = mutable

        if not isinstance(mtb, Mutable):
            raise TypeError("Can only factor out mutables, not instances of {}."
                .format(type(mtb).__name__))

        if not isinstance(mtb._vec, FullVectorization):
            raise TypeError("Can only factor out mutables with a trivial "
                "vectorization format but {} uses {}."
                .format(mtb.name, type(mtb._vec).__name__))

        if mtb not in self.mutables:
            raise LookupError("Cannot factor out {} from {} as the latter does "
                "not depend on the former.".format(mtb.name, self.string))

        ax_string = glyphs.index(self.string, mtb.name)
        ax_shape = (self._shape[0], mtb.dim)

        # Recipe in "Robust conic optimization in Python" (Stahlberg 2020).
        ax_coefs = {}
        for mtbs, coef in self._coefs.items():
            if mtb not in set(mtbs):
                continue
            elif len(mtbs) == 1:
                assert mtbs[0] is mtb
                self._update_coefs(ax_coefs, (), coef[:])
            else:
                if mtbs[0] is mtb:
                    other_mtb = mtbs[1]
                    C = coef*cvxopt_K(other_mtb.dim, mtb.dim)
                else:
                    assert mtbs[1] is mtb
                    other_mtb = mtbs[0]
                    C = coef

                d = other_mtb.dim
                C = cvxopt.sparse([C[:, i*d:(i+1)*d] for i in range(mtb.dim)])
                self._update_coefs(ax_coefs, (other_mtb,), C)

        ax = self._basetype(ax_string, ax_shape, ax_coefs)

        a0_string = glyphs.index(self.string, 0)
        a0_shape = self._shape
        a0_coefs = {M: C for M, C in self._coefs.items() if mtb not in set(M)}
        a0 = self._basetype(a0_string, a0_shape, a0_coefs)

        return ax.refined, a0.refined

    # --------------------------------------------------------------------------
    # Backwards compatibility methods.
    # --------------------------------------------------------------------------

    @classmethod
    @deprecated("2.0", useInstead="from_constant")
    def fromScalar(cls, scalar):
        """Create a class instance from a numeric scalar."""
        return cls.from_constant(scalar, (1, 1))

    @classmethod
    @deprecated("2.0", useInstead="from_constant")
    def fromMatrix(cls, matrix, size=None):
        """Create a class instance from a numeric matrix."""
        return cls.from_constant(matrix, size)

    @deprecated("2.0", useInstead="object.__xor__")
    def hadamard(self, fact):
        """Denote the elementwise (or Hadamard) product."""
        return self.__xor__(fact)

    @deprecated("2.0", useInstead="~.expression.Expression.constant")
    def isconstant(self):
        """Whether the expression involves no mutables."""
        return self.constant

    @deprecated("2.0", useInstead="equals")
    def same_as(self, other):
        """Check mathematical equality with another affine expression."""
        return self.equals(other)

    @deprecated("2.0", useInstead="T")
    def transpose(self):
        """Return the matrix transpose."""
        return self.T

    @cached_property
    @deprecated("2.0", useInstead="partial_transpose", decoratorLevel=1)
    def Tx(self):
        """Auto-detect few subsystems of same shape and transpose the last."""
        m, n = self._shape
        dims = None

        for k in range(2, int(math.log(min(m, n), 2)) + 1):
            p, q = int(round(m**(1.0/k))), int(round(n**(1.0/k)))
            if m == p**k and n == q**k:
                dims = ((p, q),)*k
                break

        if dims:
            return self.partial_transpose(subsystems=-1, dimensions=dims)
        else:
            raise RuntimeError("Failed to auto-detect subsystem dimensions for "
                "partial transposition: Only supported for {} matrices, {}."
                .format(glyphs.shape(
                    (glyphs.power("m", "k"), glyphs.power("n", "k"))),
                    glyphs.ge("k", 2)))

    @deprecated("2.0", useInstead="conj")
    def conjugate(self):
        """Return the complex conjugate."""
        return self.conj

    @deprecated("2.0", useInstead="H")
    def Htranspose(self):
        """Return the conjugate (or Hermitian) transpose."""
        return self.H

    @deprecated("2.0", reason="PICOS expressions are now immutable.")
    def copy(self):
        """Return a deep copy of the expression."""
        from copy import copy as cp

        return self._basetype(cp(self._symbStr), self._shape,
            {mtbs: cp(coef) for mtbs, coef in self._coefs.items()})

    @deprecated("2.0", reason="PICOS expressions are now immutable.")
    def soft_copy(self):
        """Return a shallow copy of the expression."""
        return self._basetype(self._symbStr, self._shape, self._coefs)


# --------------------------------------
__all__ = api_end(_API_START, globals())
