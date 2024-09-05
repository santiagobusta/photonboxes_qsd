# ------------------------------------------------------------------------------
# Copyright (C) 2019-2020 Maximilian Stahlberg
# Based in parts on the tools module by Guillaume Sagnol.
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

"""Implements functions that create or modify algebraic expressions."""

import builtins
import functools
import itertools
from collections.abc import Iterator

import cvxopt
import numpy

from .. import glyphs
from ..apidoc import api_end, api_start
from ..formatting import arguments
from ..legacy import deprecated, throw_deprecation_warning
from .cone_expcone import ExponentialCone
from .cone_rsoc import RotatedSecondOrderCone
from .cone_soc import SecondOrderCone
from .data import convert_and_refine_arguments, cvxopt_vcat, load_data
from .exp_affine import AffineExpression, BiaffineExpression, Constant
from .exp_detrootn import DetRootN
from .exp_entropy import NegativeEntropy
from .exp_extremum import MaximumConvex, MinimumConcave
from .exp_geomean import GeometricMean
from .exp_logsumexp import LogSumExp
from .exp_norm import Norm
from .exp_powtrace import PowerTrace
from .exp_sumexp import SumExponentials
from .exp_sumxtr import SumExtremes
from .expression import Expression
from .set_ball import Ball
from .set_simplex import Simplex
from .uncertain.uexp_affine import UncertainAffineExpression
from .uncertain.uexp_rand_pwl import RandomMaximumAffine, RandomMinimumAffine

_API_START = api_start(globals())
# -------------------------------


# ------------------------------------------------------------------------------
# Algebraic functions with a logic of their own.
# ------------------------------------------------------------------------------


@functools.lru_cache()
def O(rows=1, cols=1):  # noqa
    """Create a zero matrix.

    :Example:

    >>> from picos import O
    >>> print(O(2, 3))
    [0 0 0]
    [0 0 0]
    """
    return AffineExpression.zero((rows, cols))


@functools.lru_cache()
def I(size=1):  # noqa
    """Create an identity matrix.

    :Example:

    >>> from picos import I
    >>> print(I(3))
    [ 1.00e+00     0         0    ]
    [    0      1.00e+00     0    ]
    [    0         0      1.00e+00]
    """
    return AffineExpression.from_constant("I", (size, size))


@functools.lru_cache()
def J(rows=1, cols=1):
    """Create a matrix of all ones.

    :Example:

    >>> from picos import J
    >>> print(J(2, 3))
    [ 1.00e+00  1.00e+00  1.00e+00]
    [ 1.00e+00  1.00e+00  1.00e+00]
    """
    return AffineExpression.from_constant(1, (rows, cols))


def sum(lst, axis=None):
    """Sum PICOS expressions and give the result a meaningful description.

    This is a replacement for Python's :func:`sum` that produces sensible string
    representations, and in some cases a speedup, when summing over multiple
    PICOS expressions. Additionally, this can be used to denote the (complete,
    column- or row-) sum over a single matrix expression.

    :param None or int axis:
        If summing over a single matrix expression, this is the axis over which
        the sum is performed: :obj:`None` denotes the sum over all elements,
        ``0`` denotes the sum of the rows as a row vector and ``1`` denotes the
        sum of the columns as a column vector. If summing multiple expressions,
        any value other than :obj:`None` raises a :exc:`ValueError`.

    :Example:

    >>> import builtins, picos, numpy
    >>> x = picos.RealVariable("x", 5)
    >>> e = [x[i]*x[i+1] for i in range(len(x) - 1)]
    >>> builtins.sum(e)
    <Quadratic Expression: x[0]·x[1] + x[1]·x[2] + x[2]·x[3] + x[3]·x[4]>
    >>> picos.sum(e)
    <Quadratic Expression: ∑(x[i]·x[i+1] : i ∈ [0…3])>
    >>> picos.sum(x)  # The same as x.sum or (x|1).
    <1×1 Real Linear Expression: ∑(x)>
    >>> A = picos.Constant("A", range(20), (4, 5))
    >>> picos.sum(A, axis=0)  # The same as A.rowsum
    <1×5 Real Constant: [1]ᵀ·A>
    >>> picos.sum(A, axis=1)  # The same as A.colsum
    <4×1 Real Constant: A·[1]>
    >>> numpy.allclose(  # Same axis convention as NumPy.
    ...     numpy.sum(A.np, axis=0), picos.sum(A, axis=0).np)
    True
    """
    if isinstance(lst, Expression):
        if isinstance(lst, BiaffineExpression):
            if axis is None:
                return lst.sum
            elif not isinstance(axis, (int, numpy.integer)):
                raise TypeError(
                    "Axis must be an integer or None. (To sum multiple "
                    "expressions, provide an iterable as the first argument.)")
            elif axis == 0:
                return lst.rowsum
            elif axis == 1:
                return lst.colsum
            else:
                raise ValueError("Bad axis: {}.".format(axis))
        else:
            raise TypeError(
                "PICOS doesn't know how to sum over a single {}."
                .format(type(lst).__name__))

    if axis is not None:
        raise ValueError(
            "No axis may be given when summing multiple expressions.")

    # Allow passing also an iterator instead of an iterable. (The conversion is
    # necessary as otherwise the check below would use up the iterator.)
    if isinstance(lst, Iterator):
        lst = tuple(lst)

    # Resort to Python's built-in sum when no summand is a PICOS expression.
    if not any(isinstance(expression, Expression) for expression in lst):
        return builtins.sum(lst)

    # If at least one summand is a PICOS expression, attempt to convert others.
    try:
        lst = [Constant(x) if not isinstance(x, Expression) else x for x in lst]
    except Exception as error:
        raise TypeError("Failed to convert some non-expression argument to a "
            "PICOS constant.") from error

    # Handle sums with at most two summands.
    if len(lst) == 0:
        return O()
    elif len(lst) == 1:
        return lst[0]
    elif len(lst) == 2:
        return lst[0] + lst[1]

    # Find a suitable string description.
    string = glyphs.sum(arguments([exp.string for exp in lst]))

    # Handle (large) sums of only affine expressions efficiently.
    if all(isinstance(expression, BiaffineExpression) for expression in lst):
        # Determine resulting shape.
        shapes = set(expression.shape for expression in lst)
        if len(shapes) != 1:
            raise TypeError("The shapes of summands do not match.")
        else:
            shape = shapes.pop()

        # Determine resulting type.
        try:
            basetype = functools.reduce(
                lambda A, B: A._common_basetype(B), lst, AffineExpression)
        except Exception as error:
            raise TypeError("Could not find a common base type for the given "
                "(bi-)affine expressions.") from error

        # Sum all coefficient matrices.
        # NOTE: BiaffineExpression.__init__ will order mutable pairs and merge
        #       their coefficient matrices.
        coefs = {}
        byref = set()
        for expression in lst:
            for mtbs, coef in expression._coefs.items():
                if mtbs in coefs:
                    if mtbs in byref:
                        # Make a copy of the coefficient so we may modify it.
                        coefs[mtbs] = load_data(coefs[mtbs], alwaysCopy=True)[0]
                        byref.remove(mtbs)

                    try:
                        coefs[mtbs] += coef
                    except TypeError:
                        # No in-place addition for sparse and dense types.
                        coefs[mtbs] = coefs[mtbs] + coef
                else:
                    # Store the first coefficient by reference.
                    coefs[mtbs] = coef
                    byref.add(mtbs)

        return basetype(string, shape, coefs)

    theSum = lst[0]
    for expression in lst[1:]:
        theSum += expression

    theSum._symbStr = string

    return theSum


def block(nested, shapes=None, name=None):
    """Create an affine block matrix expression.

    Given a two-level nested iterable container (e.g. a list of lists) of PICOS
    affine expressions or constant data values or a mix thereof, this creates an
    affine block matrix where each inner container represents one block row and
    each expression or constant represents one block.

    Blocks that are given as PICOS expressions are never reshaped or
    broadcasted. Their shapes must already be consistent. Blocks that are given
    as constant data values are reshaped or broadcasted as necessary **to match
    existing PICOS expressions**. This means you can specify blocks as e.g.
    ``"I"`` or ``0`` and PICOS will load them as matrices with the smallest
    shape that is consistent with other blocks given as PICOS expressions.

    Since constant data values are not reshaped or broadcasted with respect to
    each other, the ``shapes`` parameter allows a manual clarification of block
    shapes. It must be consistent with the shapes of blocks given as PICOS
    expressions (they are still not reshaped or broadcasted).

    :param nested:
        The blocks.
    :type nested:
        tuple(tuple) or list(list)

    :param shapes:
        A pair ``(rows, columns)`` where ``rows`` defines the number of rows for
        each block row and ``columns`` defines the number of columns for each
        block column. You can put a ``0`` or :obj:`None` as a wildcard.
    :type shapes:
        tuple(tuple) or list(list)

    :param str name:
        Name or string description of the resulting block matrix. If
        :obj:`None`, a descriptive string will be generated.

    :Example:

    >>> from picos import block, Constant, RealVariable
    >>> C = Constant("C", range(6), (3, 2))
    >>> d = Constant("d", 0.5, 2)
    >>> x = RealVariable("x", 3)
    >>> A = block([[ C,  x ],
    ...            ["I", d ]]); A
    <5×3 Real Affine Expression: [C, x; I, d]>
    >>> x.value = [60, 70, 80]
    >>> print(A)
    [ 0.00e+00  3.00e+00  6.00e+01]
    [ 1.00e+00  4.00e+00  7.00e+01]
    [ 2.00e+00  5.00e+00  8.00e+01]
    [ 1.00e+00  0.00e+00  5.00e-01]
    [ 0.00e+00  1.00e+00  5.00e-01]
    >>> B = block([[ C,  x ],  # With a shape hint.
    ...            ["I", 0 ]], shapes=((3, 2), (2, 1))); B
    <5×3 Real Affine Expression: [C, x; I, 0]>
    >>> print(B)
    [ 0.00e+00  3.00e+00  6.00e+01]
    [ 1.00e+00  4.00e+00  7.00e+01]
    [ 2.00e+00  5.00e+00  8.00e+01]
    [ 1.00e+00  0.00e+00  0.00e+00]
    [ 0.00e+00  1.00e+00  0.00e+00]
    """
    # In a first stage, determine and validate fixed shapes from PICOS
    # expressions, then load the remaining data to be consistent with the fixed
    # shapes. In a second stage, validate also the shapes of the loaded data.
    for stage in range(2):
        R = numpy.array([[  # The row count for each block.
            x.shape[0] if isinstance(x, BiaffineExpression) else 0 for x in row]
            for row in nested], dtype=int)

        M = []  # The row count for each block row.
        for i, Ri in enumerate(R):
            m = set(int(x) for x in Ri[numpy.nonzero(Ri)])

            if shapes and shapes[0][i]:
                m.add(shapes[0][i])

            if len(m) > 1:
                raise TypeError(
                    "Inconsistent number of rows in block row {}: {}."
                    .format(i + 1, m))
            elif len(m) == 1:
                M.append(int(m.pop()))
            else:
                assert stage == 0, "All blocks should have a shape by now."
                M.append(None)

        C = numpy.array([[  # The column count for each block.
            x.shape[1] if isinstance(x, BiaffineExpression) else 0 for x in row]
            for row in nested], dtype=int)

        N = []  # The column count for each block column.
        for j, Cj in enumerate(C.T):
            n = set(int(x) for x in Cj[numpy.nonzero(Cj)])

            if shapes and shapes[1][j]:
                n.add(shapes[1][j])

            if len(n) > 1:
                raise TypeError(
                    "Inconsistent number of columns in block column {}: {}."
                    .format(j + 1, n))
            elif len(n) == 1:
                N.append(n.pop())
            else:
                assert stage == 0, "All blocks should have a shape by now."
                N.append(None)

        if stage == 0:
            nested = [[
                x if isinstance(x, BiaffineExpression)
                else Constant(x, shape=(M[i], N[j]))
                for j, x in enumerate(row)] for i, row in enumerate(nested)]

    # List the blocks in block-row-major order.
    blocks = [block for blockRow in nested for block in blockRow]

    # Find the common base type of all expressions.
    basetype = functools.reduce(BiaffineExpression._common_basetype.__func__,
        (block.__class__ for block in blocks), AffineExpression)
    typecode = basetype._get_typecode()

    # Allow constant time random access to the block dimensions.
    M, N = tuple(M), tuple(N)

    # Compute the row (column) offsets for each block row (block column).
    MOffsets = tuple(int(x) for x in numpy.cumsum((0,) + M))
    NOffsets = tuple(int(x) for x in numpy.cumsum((0,) + N))

    # Compute the full matrix dimensions.
    m = builtins.sum(M)
    n = builtins.sum(N)
    mn = m*n

    # Compute row and column offsets for each block in block-row-major-order.
    MLen = len(N)
    blockIndices = tuple(divmod(k, MLen) for k in range(len(blocks)))
    blockOffsets = tuple(
        (MOffsets[blockIndices[k][0]], NOffsets[blockIndices[k][1]])
        for k in range(len(blocks)))

    # Helper function to compute the matrix T (see below).
    def _I():
        for k, block in enumerate(blocks):
            rows, cols = block.shape
            i, j = blockOffsets[k]
            for c in range(cols):
                columnOffset = (j + c)*m + i
                yield range(columnOffset, columnOffset + rows)

    # Compute a sparse linear operator matrix T that transforms the stacked
    # column-major vectorizations of the blocks in block-row-major order to the
    # column-major vectorization of the full matrix.
    V = tuple(itertools.repeat(1, mn))
    I = tuple(itertools.chain(*_I()))
    J = range(mn)
    T = cvxopt.spmatrix(V, I, J, (mn, mn), typecode)

    # Obtain all coefficient keys.
    keys = set(key for block in blocks for key in block._coefs.keys())

    # Stack all coefficient matrices in block-row-major order and apply T.
    coefs = {}
    for mtbs in keys:
        dim = functools.reduce(lambda x, y:
            (x if isinstance(x, int) else x.dim)*y.dim, mtbs, 1)

        coefs[mtbs] = T*cvxopt_vcat([
            block._coefs[mtbs] if mtbs in block._coefs
            else cvxopt.spmatrix([], [], [], (len(block), dim), typecode)
            for block in blocks])

    # Build the string description.
    if name:
        string = str(name)
    elif len(blocks) > 9:
        string = glyphs.matrix(glyphs.shape((m, n)))
    else:
        string = functools.reduce(
            lambda x, y: glyphs.matrix_cat(x, y, False),
            (functools.reduce(
                lambda x, y: glyphs.matrix_cat(x, y, True),
                (block.string for block in blockRow))
            for blockRow in nested))

    return basetype(string, (m, n), coefs)


def max(lst):
    """Denote the maximum over a collection of convex scalar expressions.

    If instead of a collection of expressions only a single multidimensional
    affine expression is given, this denotes its largest element instead.

    If some individual expressions are uncertain and their uncertainty is not of
    stochastic but of worst-case nature (robust optimization), then the maximum
    implicitly goes over their perturbation parameters as well.

    :param lst:
        A list of convex expressions or a single affine expression.
    :type lst:
        list or tuple or ~picos.expressions.AffineExpression

    :Example:

    >>> from picos import RealVariable, max, sum
    >>> x = RealVariable("x", 5)
    >>> max(x)
    <Largest Element: max(x)>
    >>> max(x) <= 2  # The same as x <= 2.
    <Largest Element Constraint: max(x) ≤ 2>
    >>> max([sum(x), abs(x)])
    <Maximum of Convex Functions: max(∑(x), ‖x‖)>
    >>> max([sum(x), abs(x)]) <= 2  # Both must be <= 2.
    <Maximum of Convex Functions Constraint: max(∑(x), ‖x‖) ≤ 2>
    >>> from picos.uncertain import UnitBallPerturbationSet
    >>> z = UnitBallPerturbationSet("z", 5).parameter
    >>> max([sum(x), x.T*z])  # Also maximize over z.
    <Maximum of Convex Functions: max(∑(x), max_z xᵀ·z)>
    """
    UAE = UncertainAffineExpression

    if isinstance(lst, AffineExpression):
        return SumExtremes(lst, 1, largest=True, eigenvalues=False)
    elif isinstance(lst, Expression):
        raise TypeError("May only denote the maximum of a single affine "
            "expression or of multiple (convex) expressions.")

    try:
        lst = [Constant(x) if not isinstance(x, Expression) else x for x in lst]
    except Exception as error:
        raise TypeError("Failed to convert some non-expression argument to a "
            "PICOS constant.") from error

    if any(isinstance(x, UAE) for x in lst) \
    and all(isinstance(x, (AffineExpression, UAE)) for x in lst) \
    and all(x.certain or x.universe.distributional for x in lst):
        return RandomMaximumAffine(lst)
    else:
        return MaximumConvex(lst)


def min(lst):
    """Denote the minimum over a collection of concave scalar expressions.

    If instead of a collection of expressions only a single multidimensional
    affine expression is given, this denotes its smallest element instead.

    If some individual expressions are uncertain and their uncertainty is not of
    stochastic but of worst-case nature (robust optimization), then the minimum
    implicitly goes over their perturbation parameters as well.

    :param lst:
        A list of concave expressions or a single affine expression.
    :type lst:
        list or tuple or ~picos.expressions.AffineExpression

    :Example:

    >>> from picos import RealVariable, min, sum
    >>> x = RealVariable("x", 5)
    >>> min(x)
    <Smallest Element: min(x)>
    >>> min(x) >= 2  # The same as x >= 2.
    <Smallest Element Constraint: min(x) ≥ 2>
    >>> min([sum(x), -x[0]**2])
    <Minimum of Concave Functions: min(∑(x), -x[0]²)>
    >>> min([sum(x), -x[0]**2]) >= 2  # Both must be >= 2.
    <Minimum of Concave Functions Constraint: min(∑(x), -x[0]²) ≥ 2>
    >>> from picos.uncertain import UnitBallPerturbationSet
    >>> z = UnitBallPerturbationSet("z", 5).parameter
    >>> min([sum(x), x.T*z])  # Also minimize over z.
    <Minimum of Concave Functions: min(∑(x), min_z xᵀ·z)>
    """
    UAE = UncertainAffineExpression

    if isinstance(lst, AffineExpression):
        return SumExtremes(lst, 1, largest=False, eigenvalues=False)
    elif isinstance(lst, Expression):
        raise TypeError("May only denote the minimum of a single affine "
            "expression or of multiple (concave) expressions.")

    try:
        lst = [Constant(x) if not isinstance(x, Expression) else x for x in lst]
    except Exception as error:
        raise TypeError("Failed to convert some non-expression argument to a "
            "PICOS constant.") from error

    if any(isinstance(x, UAE) for x in lst) \
    and all(isinstance(x, (AffineExpression, UAE)) for x in lst) \
    and all(x.certain or x.universe.distributional for x in lst):
        return RandomMinimumAffine(lst)
    else:
        return MinimumConcave(lst)


# ------------------------------------------------------------------------------
# Functions that call expression methods.
# ------------------------------------------------------------------------------


def _error_on_none(func):
    """Raise a :exc:`TypeError` if the function returns :obj:`None`."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        if result is None:
            raise TypeError("PICOS does not have a representation for {}({})."
                .format(func.__qualname__ if hasattr(func, "__qualname__") else
                func.__name__, ", ".join([type(x).__name__ for x in args] +
                ["{}={}".format(k, type(x).__name__) for k, x in kwargs.items()]
                )))

        return result
    return wrapper


@_error_on_none
@convert_and_refine_arguments("x")
def exp(x):
    """Denote the exponential."""
    if hasattr(x, "exp"):
        return x.exp


@_error_on_none
@convert_and_refine_arguments("x")
def log(x):
    """Denote the natural logarithm."""
    if hasattr(x, "log"):
        return x.log


@deprecated("2.2", "Ensure that one operand is a PICOS expression and use infix"
    " @ instead.")
@_error_on_none
@convert_and_refine_arguments("x", "y")
def kron(x, y):
    """Denote the kronecker product."""
    if hasattr(x, "kron"):
        return x @ y


@_error_on_none
@convert_and_refine_arguments("x")
def diag(x, n=1):
    r"""Form a diagonal matrix from the column-major vectorization of :math:`x`.

    If :math:`n \neq 1`, then the vectorization is repeated :math:`n` times.
    """
    if hasattr(x, "diag") and n == 1:
        return x.diag
    elif hasattr(x, "dupdiag"):
        return x.dupdiag(n)


@_error_on_none
@convert_and_refine_arguments("x")
def maindiag(x):
    """Extract the diagonal of :math:`x` as a column vector."""
    if hasattr(x, "maindiag"):
        return x.maindiag


@_error_on_none
@convert_and_refine_arguments("x")
def trace(x):
    """Denote the trace of a square matrix."""
    if hasattr(x, "tr"):
        return x.tr


@_error_on_none
@convert_and_refine_arguments("x")
def partial_trace(x, subsystems=0, dimensions=2, k=None, dim=None):
    """See :meth:`.exp_biaffine.BiaffineExpression.partial_trace`.

    The parameters `k` and `dim` are for backwards compatibility.
    """
    if k is not None:
        throw_deprecation_warning("Argument 'k' to partial_trace is "
            "deprecated: Use 'subsystems' instead.", decoratorLevel=2)
        subsystems = k

    if dim is not None:
        throw_deprecation_warning("Argument 'dim' to partial_trace is "
            "deprecated: Use 'dimensions' instead.", decoratorLevel=2)
        dimensions = dim

    if isinstance(x, BiaffineExpression):
        return x.partial_trace(subsystems, dimensions)


@_error_on_none
@convert_and_refine_arguments("x")
def partial_transpose(x, subsystems=0, dimensions=2, k=None, dim=None):
    """See :meth:`.exp_biaffine.BiaffineExpression.partial_transpose`.

    The parameters `k` and `dim` are for backwards compatibility.
    """
    if k is not None:
        throw_deprecation_warning("Argument 'k' to partial_transpose is "
            "deprecated: Use 'subsystems' instead.", decoratorLevel=2)
        subsystems = k

    if dim is not None:
        throw_deprecation_warning("Argument 'dim' to partial_transpose is "
            "deprecated: Use 'dimensions' instead.", decoratorLevel=2)
        dimensions = dim

    if isinstance(x, BiaffineExpression):
        return x.partial_transpose(subsystems, dimensions)


# ------------------------------------------------------------------------------
# Alias functions for expression classes meant to be instanciated by the user.
# ------------------------------------------------------------------------------


def _shorthand(name, cls):
    def shorthand(*args, **kwargs):
        return cls(*args, **kwargs)

    shorthand.__doc__ = "Shorthand for :class:`{1} <{0}.{1}>`.".format(
        cls.__module__, cls.__qualname__)
    shorthand.__name__ = name
    shorthand.__qualname__ = name

    return shorthand


expcone = _shorthand("expcone", ExponentialCone)
geomean = _shorthand("geomean", GeometricMean)
kldiv   = _shorthand("kldiv", NegativeEntropy)
lse     = _shorthand("lse", LogSumExp)
rsoc    = _shorthand("rsoc", RotatedSecondOrderCone)
soc     = _shorthand("soc", SecondOrderCone)
sumexp  = _shorthand("sumexp", SumExponentials)


def sum_k_largest(x, k):
    """Wrapper for :class:`~picos.SumExtremes`.

    Sets ``largest = True`` and ``eigenvalues = False``.

    :Example:

    >>> from picos import RealVariable, sum_k_largest
    >>> x = RealVariable("x", 5)
    >>> sum_k_largest(x, 2)
    <Sum of Largest Elements: sum_2_largest(x)>
    >>> sum_k_largest(x, 2) <= 2
    <Sum of Largest Elements Constraint: sum_2_largest(x) ≤ 2>
    """
    return SumExtremes(x, k, largest=True, eigenvalues=False)


def sum_k_smallest(x, k):
    """Wrapper for :class:`~picos.SumExtremes`.

    Sets ``largest = False`` and ``eigenvalues = False``.

    :Example:

    >>> from picos import RealVariable, sum_k_smallest
    >>> x = RealVariable("x", 5)
    >>> sum_k_smallest(x, 2)
    <Sum of Smallest Elements: sum_2_smallest(x)>
    >>> sum_k_smallest(x, 2) >= 2
    <Sum of Smallest Elements Constraint: sum_2_smallest(x) ≥ 2>
    """
    return SumExtremes(x, k, largest=False, eigenvalues=False)


def lambda_max(x):
    """Wrapper for :class:`~picos.SumExtremes`.

    Sets ``k = 1``, ``largest = True`` and ``eigenvalues = True``.

    :Example:

    >>> from picos import SymmetricVariable, lambda_max
    >>> X = SymmetricVariable("X", 5)
    >>> lambda_max(X)
    <Largest Eigenvalue: λ_max(X)>
    >>> lambda_max(X) <= 2
    <Largest Eigenvalue Constraint: λ_max(X) ≤ 2>
    """
    return SumExtremes(x, 1, largest=True, eigenvalues=True)


def lambda_min(x):
    """Wrapper for :class:`~picos.SumExtremes`.

    Sets ``k = 1``, ``largest = False`` and ``eigenvalues = True``.

    :Example:

    >>> from picos import SymmetricVariable, lambda_min
    >>> X = SymmetricVariable("X", 5)
    >>> lambda_min(X)
    <Smallest Eigenvalue: λ_min(X)>
    >>> lambda_min(X) >= 2
    <Smallest Eigenvalue Constraint: λ_min(X) ≥ 2>
    """
    return SumExtremes(x, 1, largest=False, eigenvalues=True)


def sum_k_largest_lambda(x, k):
    """Wrapper for :class:`~picos.SumExtremes`.

    Sets ``largest = True`` and ``eigenvalues = True``.

    :Example:

    >>> from picos import SymmetricVariable, sum_k_largest_lambda
    >>> X = SymmetricVariable("X", 5)
    >>> sum_k_largest_lambda(X, 2)
    <Sum of Largest Eigenvalues: sum_2_largest_λ(X)>
    >>> sum_k_largest_lambda(X, 2) <= 2
    <Sum of Largest Eigenvalues Constraint: sum_2_largest_λ(X) ≤ 2>
    """
    return SumExtremes(x, k, largest=True, eigenvalues=True)


def sum_k_smallest_lambda(x, k):
    """Wrapper for :class:`~picos.SumExtremes`.

    Sets ``largest = False`` and ``eigenvalues = True``.

    :Example:

    >>> from picos import SymmetricVariable, sum_k_smallest_lambda
    >>> X = SymmetricVariable("X", 5)
    >>> sum_k_smallest_lambda(X, 2)
    <Sum of Smallest Eigenvalues: sum_2_smallest_λ(X)>
    >>> sum_k_smallest_lambda(X, 2) >= 2
    <Sum of Smallest Eigenvalues Constraint: sum_2_smallest_λ(X) ≥ 2>
    """
    return SumExtremes(x, k, largest=False, eigenvalues=True)


# ------------------------------------------------------------------------------
# Legacy algebraic functions for backwards compatibility.
# ------------------------------------------------------------------------------


def _deprecated_shorthand(name, cls, new_shorthand=None):
    uiRef = "picos.{}".format(new_shorthand if new_shorthand else cls.__name__)

    # FIXME: Warning doesn't show the name of the deprecated shorthand function.
    @deprecated("2.0", useInstead=uiRef)
    def shorthand(*args, **kwargs):
        """|PLACEHOLDER|"""  # noqa
        return cls(*args, **kwargs)

    shorthand.__doc__ = shorthand.__doc__.replace("|PLACEHOLDER|",
        "Legacy shorthand for :class:`{1} <{0}.{1}>`.".format(
        cls.__module__, cls.__qualname__))
    shorthand.__name__ = name
    shorthand.__qualname__ = name

    return shorthand


ball             = _deprecated_shorthand("ball", Ball)
detrootn         = _deprecated_shorthand("detrootn", DetRootN)
kullback_leibler = _deprecated_shorthand(
    "kullback_leibler", NegativeEntropy, "kldiv")
logsumexp        = _deprecated_shorthand("logsumexp", LogSumExp, "lse")
norm             = _deprecated_shorthand("norm", Norm)


@deprecated("2.0", useInstead="picos.PowerTrace")
def tracepow(exp, num=1, denom=1, coef=None):
    """Legacy shorthand for :class:`~picos.PowerTrace`."""
    return PowerTrace(exp, num / denom, coef)


@deprecated("2.0", useInstead="picos.Constant")
def new_param(name, value):
    """Create a constant or a list or dict or tuple thereof."""
    if isinstance(value, list):
        # Handle a vector.
        try:
            for x in value:
                complex(x)
        except Exception:
            pass
        else:
            return Constant(name, value)

        # Handle a matrix.
        if all(isinstance(x, list) for x in value) \
        and all(len(x) == len(value[0]) for x in value):
            try:
                for x in value:
                    for y in x:
                        complex(y)
            except Exception:
                pass
            else:
                return Constant(name, value)

        # Return a list of constants.
        return [Constant(glyphs.slice(name, i), x) for i, x in enumerate(value)]
    elif isinstance(value, tuple):
        # Return a list of constants.
        # NOTE: This is very inconsistent, but legacy behavior.
        return [Constant(glyphs.slice(name, i), x) for i, x in enumerate(value)]
    elif isinstance(value, dict):
        return {k: Constant(glyphs.slice(name, k), x) for k, x in value.items()}
    else:
        return Constant(name, value)


@deprecated("2.0", useInstead="picos.FlowConstraint")
def flow_Constraint(*args, **kwargs):
    """Legacy shorthand for :class:`~picos.FlowConstraint`."""
    from ..constraints.con_flow import FlowConstraint
    return FlowConstraint(*args, **kwargs)


@deprecated("2.0", useInstead="picos.maindiag")
def diag_vect(x):
    """Extract the diagonal of :math:`x` as a column vector."""
    return maindiag(x)


@deprecated("2.0", useInstead="picos.Simplex")
def simplex(gamma):
    r"""Create a standard simplex of radius :math:`\gamma`."""
    return Simplex(gamma, truncated=False, symmetrized=False)


@deprecated("2.0", useInstead="picos.Simplex")
def truncated_simplex(gamma, sym=False):
    r"""Create a truncated simplex of radius :math:`\gamma`."""
    return Simplex(gamma, truncated=True, symmetrized=sym)


# --------------------------------------
__all__ = api_end(_API_START, globals())
