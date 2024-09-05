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

"""Implements :class:`QuadraticExpression`."""

import operator
import sys
from collections import OrderedDict as odict
from collections import namedtuple
from types import MappingProxyType

import cvxopt
import cvxopt.cholmod
import numpy

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import (borrow_cache, cached_property,
                       cached_selfinverse_unary_operator,
                       cached_unary_operator)
from ..constraints import (ConicQuadraticConstraint, Constraint,
                           ConvexQuadraticConstraint,
                           NonconvexQuadraticConstraint)
from ..legacy import deprecated
from .data import convert_operands, cvx2np, should_be_sparse
from .exp_affine import AffineExpression, ComplexAffineExpression, Constant
from .expression import Expression, refine_operands, validate_prediction
from .variables import BaseVariable

_API_START = api_start(globals())
# -------------------------------


PSD_PERTURBATIONS = 3
r"""Maximum number of singular quadratic form perturbations.

PICOS uses NumPy either or CHOLMOD to compute a Cholesky decomposition of a
positive semidefinite quadratic form :math:`Q`. Both libraries require
:math:`Q` to be nonsingular, which is not a requirement on PICOS' end.
If either library rejects :math:`Q`, then PICOS provides a sequence of
:data:`PSD_PERTURBATIONS` perturbed quadratic forms :math:`Q + \epsilon I` for
increasing :math:`\epsilon` until the perturbed matrix is found positive
definite or until the largest :math:`\epsilon` was tested unsuccessfully.

If this is zero, PICOS will only decompose quadratic forms that are nonsingular.
If this is one, then only the largest epsilon is tested.
"""


class QuadraticExpression(Expression):
    """A scalar quadratic expression of the form :math:`x^TQx + a^Tx + b`."""

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    def __init__(
            self, string, quadraticPart={}, affinePart=AffineExpression.zero(),
            scalarFactors=None, copyDecomposition=None):
        r"""Initialize a scalar quadratic expression.

        This constructor is meant for internal use. As a user, you will most
        likely want to build expressions starting from
        :mod:`~picos.expressions.variables` or a :func:`~picos.Constant`.

        :param str string: A symbolic string description.
        :param quadraticPart: A :class:`dict` mapping PICOS variable pairs to
            CVXOPT matrices. Each entry :math:`(x, y) \mapsto A_{xy}` represents
            a quadratic form
            :math:`\operatorname{vec_x}(x)^T A_{xy} \operatorname{vec_y}(y)`
            where :math:`\operatorname{vec_x}` and :math:`\operatorname{vec_y}`
            refer to the isometric real
            :mod:`~.picos.expressions.vectorizations` that are used by PICOS
            internally to store the variables :math:`x` and :math:`y`,
            respectively. The quadratic part :math:`Q` of the expression is then
            given as :math:`Q = \sum_{x, y} A_{xy}`.
        :param affinePart: The affine part :math:`a^Tx + b` of the expression.
        :type affinePart: ~picos.expressions.AffineExpression
        :param scalarFactors: A pair :math:`(u, v)` with both :math:`u` and
            :math:`v` scalar real affine expressions representing a known
            :math:`x^TQx + a^Tx + b = uv` factorization of the expression.
        :type factorization:
            tuple(~picos.expressions.AffineExpression)
        :param copyDecomposition: Another quadratic expression with equal
            quadratic part whose quadratic part decomposition shall be copied.
        :type copyDecomposition:
            ~picos.expressions.QuadraticExpression
        """
        # Ensure correct usage within PICOS.
        assert isinstance(quadraticPart, dict) and all(
            isinstance(xy, (tuple, list)) and len(xy) == 2
            and isinstance(xy[0], BaseVariable)
            and isinstance(xy[1], BaseVariable)
            and isinstance(A, (cvxopt.matrix, cvxopt.spmatrix))
            and A.size == (xy[0].dim, xy[1].dim)
            for xy, A in quadraticPart.items())
        assert isinstance(affinePart, ComplexAffineExpression) \
            and len(affinePart) == 1
        assert not copyDecomposition \
            or isinstance(copyDecomposition, QuadraticExpression)
        assert not scalarFactors or (
            isinstance(scalarFactors, (tuple, list)) and len(scalarFactors) == 2
            and all(isinstance(x, AffineExpression) for x in scalarFactors)
            and all(x.scalar for x in scalarFactors))

        Expression.__init__(self, "Quadratic Expression", string)

        # Check affine part.
        affinePart = affinePart.refined
        if not isinstance(affinePart, AffineExpression):
            raise NotImplementedError(
                "PICOS does not support complex-valued quadratic expressions "
                "and the affine part of {} is not real.".format(self._symbStr))
        elif len(affinePart) != 1:
            raise TypeError("Affine part of {} is not scalar."
                .format(self._symbStr))

        # Store quadratic part.
        self._quads = {}
        for xy, A in quadraticPart.items():
            # Ignore coefficients that are already zero.
            if not A:
                continue

            if xy[0] is xy[1]:
                # Store the unique symmetric form for two reasons:
                # - Hermitian forms concerning the original variables are
                #   supplied as quadratic forms concerning the isometric real
                #   vectorizations of the variables. These matrices are in
                #   general still complex, but their symmetric form is not.
                # - Coefficients that cancel out are set to numeric zeros and
                #   can be converted to structural zeros by _sparse_quads.
                A = 0.5*(A + A.T)
            if hash(xy[1]) < hash(xy[0]):
                # Store only one coefficient per unordered pair.
                xy, A = (xy[1], xy[0]), A.T

            if xy in self._quads:
                self._quads[xy] = self._quads[xy] + A
            else:
                self._quads[xy] = A

        # Remove coefficients of zero.
        self._quads = {xy: A for xy, A in self._quads.items() if A}

        # Check if coefficients are real.
        for (x, y), A in self._quads.items():
            if A.typecode == "z":
                if any(A.imag()):
                    raise NotImplementedError(
                        "PICOS does not support complex-valued quadratic "
                        "expressions and the quadratic part of {} for variables"
                        " {} and {} is not real or equivalent to a real form."
                        .format(self._symbStr, x.string, y.string))

                self._quads[xy] = A.real()

        # Store affine part.
        self._aff = affinePart

        # Store a known factorization into real scalars.
        if scalarFactors:
            a, b = scalarFactors
            if a.equals(b):
                self._sf = (a, a)  # Speedup future equality checks.
            else:
                self._sf = (a, b)
        else:
            self._sf = None

        # Copy a decomposition from another quadratic expression.
        if copyDecomposition is not None:
            borrow_cache(self, copyDecomposition, ("_Q_and_x", "L", "quadroot"))

    # --------------------------------------------------------------------------
    # Abstract method implementations and method overridings, except _predict.
    # --------------------------------------------------------------------------

    @cached_unary_operator
    def _get_refined(self):
        if not self._quads:
            return self._aff.refined.renamed(self._symbStr)
        else:
            return self

    Subtype = namedtuple("Subtype", (
        "convex", "concave", "qterms", "rank", "haslin", "factors"))

    @cached_unary_operator
    def _get_subtype(self):
        convex  = self.convex
        concave = False if convex and self._quads else self.concave
        qterms  = self.num_quad_terms
        haslin  = not self._aff.constant

        try:
            if convex:
                rank = self.rank
            elif concave:
                rank = (-self).rank
            else:
                rank = None
        except ValueError:  # No quadratic part.
            rank = 0

        factors = len(set(self._sf)) if self._sf else 0

        return self.Subtype(convex, concave, qterms, rank, haslin, factors)

    def _get_value(self):
        value = self._aff._get_value()

        for xy, A in self._quads.items():
            x, y = xy
            xVal = x._get_internal_value()
            yVal = y._get_internal_value()
            value = value + xVal.T * A * yVal

        return value

    @cached_unary_operator
    def _get_mutables(self):
        return self._aff._get_mutables().union(
            var for vars in self._quads for var in vars)

    def _is_convex(self):
        try:
            self.L
        except ValueError:
            return True  # Expression is affine.
        except ArithmeticError:
            return False
        else:
            return True

    def _is_concave(self):
        try:
            (-self).L
        except ValueError:
            return True  # Expression is affine.
        except ArithmeticError:
            return False
        else:
            return True

    def _replace_mutables(self, mapping):
        name_map = {old.name: new.name for old, new in mapping.items()}

        string = self.string
        if isinstance(string, glyphs.GlStr):
            string = string.reglyphed(name_map)

        quads = {(mapping[var1], mapping[var2]): coef
            for (var1, var2), coef in self._quads.items()}

        if not self._sf:
            sf = None
        elif self._sf[0] is self._sf[1]:
            sf = (self._sf[0]._replace_mutables(mapping),)*2
        else:
            sf = tuple(f._replace_mutables(mapping) for f in self._sf)

        return QuadraticExpression(
            string, quads, self.aff._replace_mutables(mapping), sf)

    def _freeze_mutables(self, freeze):
        # TODO: Allow freezing of quadratic expressions.
        raise NotImplementedError(
            "Partially freezing quadratic expressions is not yet supported.")

    # --------------------------------------------------------------------------
    # Python special method implementations, except constraint-creating ones.
    # --------------------------------------------------------------------------

    @classmethod
    def _add_sub(cls, self, other, add, forward):
        def affine_part_and_string(other_aff):
            if add:
                if forward:
                    affine = self._aff + other_aff
                    string = glyphs.clever_add(self.string, other.string)
                else:
                    affine = other_aff + self._aff
                    string = glyphs.clever_add(other.string, self.string)
            else:
                if forward:
                    affine = self._aff - other_aff
                    string = glyphs.clever_sub(self.string, other.string)
                else:
                    affine = other_aff - self._aff
                    string = glyphs.clever_sub(other.string, self.string)

            return affine, string

        if isinstance(other, cls):
            affine, string = affine_part_and_string(other._aff)

            quads = {}
            varPairs = set(self._quads.keys()).union(other._quads.keys())
            for vars in varPairs:
                if vars in self._quads:
                    if vars in other._quads:
                        if add:
                            quads[vars] = self._quads[vars] + other._quads[vars]
                        elif forward:
                            quads[vars] = self._quads[vars] - other._quads[vars]
                        else:
                            quads[vars] = other._quads[vars] - self._quads[vars]
                    elif not add and not forward:
                        quads[vars] = -self._quads[vars]
                    else:
                        quads[vars] = self._quads[vars]
                elif not add and forward:
                    quads[vars] = -other._quads[vars]
                else:
                    quads[vars] = other._quads[vars]

            return cls(string, quads, affine)
        elif isinstance(other, AffineExpression):
            affine, string = affine_part_and_string(other)

            if add or forward:
                quads = self._quads
                copy_ = self
            else:
                quads = {xy: -A for xy, A in self._quads.items()}
                copy_ = None

            return cls(string, quads, affine, copyDecomposition=copy_)

        if add:
            if forward:
                return Expression.__add__(self, other)
            else:
                return Expression.__radd__(self, other)
        else:
            if forward:
                return Expression.__sub__(self, other)
            else:
                return Expression.__rsub__(self, other)

    @convert_operands(sameShape=True)
    def __add__(self, other):
        """Denote addition from the right hand side."""
        return QuadraticExpression._add_sub(self, other, True, True)

    @convert_operands(sameShape=True)
    def __radd__(self, other):
        """Denote addition from the left hand side."""
        return QuadraticExpression._add_sub(self, other, True, False)

    @convert_operands(sameShape=True)
    def __sub__(self, other):
        """Denote substraction from the right hand side."""
        return QuadraticExpression._add_sub(self, other, False, True)

    @convert_operands(sameShape=True)
    def __rsub__(self, other):
        """Denote substraction with self on the right hand side."""
        return QuadraticExpression._add_sub(self, other, False, False)

    @cached_selfinverse_unary_operator
    def __neg__(self):
        string  = glyphs.clever_neg(self.string)
        quads   = {xy: -A for xy, A in self._quads.items()}
        affine  = -self._aff
        factors = (self._sf[0], -self._sf[1]) if self._sf else None

        return QuadraticExpression(string, quads, affine, factors)

    @classmethod
    def _mul_div(cls, self, other, div, forward):
        assert not div or forward

        if isinstance(other, AffineExpression):
            if not other.constant:
                if div:
                    # NIE as this would makes sense if dividing by a factor.
                    raise NotImplementedError("You may only divide a quadratic "
                        "expression by a constant term.")
                else:
                    raise TypeError("You may only multiply a quadratic "
                        "expression with a constant term.")

            factor = other.safe_value

            if not factor:
                if div:
                    raise ZeroDivisionError(
                        "Cannot divide {} by zero.".format(self.string))
                else:
                    return AffineExpression.zero()
            elif factor == 1:
                return self
            elif factor == -1:
                return -self

            if div:
                factor = 1 / factor
                affine = self._aff / other
                string = glyphs.div(self.string, other.string)
            elif forward:
                affine = self._aff * other
                string = glyphs.clever_mul(self.string, other.string)
            else:
                affine = other * self._aff
                string = glyphs.clever_mul(other.string, self.string)

            quads = {xy: factor*A for xy, A in self._quads.items()}

            if self._sf:
                r = factor**0.5
                if r.imag:
                    factors = (r.imag*self._sf[0], -r.imag*self._sf[1])
                elif self._sf[0] is self._sf[1]:
                    factors = (r*self._sf[0],)*2
                else:
                    factors = (r*self._sf[0], r*self._sf[1])
            else:
                factors = None

            return cls(string, quads, affine, factors)

        if div:
            return Expression.__truediv__(self, other)
        elif forward:
            return Expression.__mul__(self, other)
        else:
            return Expression.__rmul__(self, other)

    @convert_operands(scalarRHS=True)
    def __mul__(self, other):
        """Denote scaling from the right hand side."""
        return QuadraticExpression._mul_div(self, other, False, True)

    @convert_operands(scalarRHS=True)
    def __rmul__(self, other):
        """Denote scaling from the left hand side."""
        return QuadraticExpression._mul_div(self, other, False, False)

    @convert_operands(scalarRHS=True)
    def __truediv__(self, other):
        """Denote division by a constant scalar."""
        return QuadraticExpression._mul_div(self, other, True, True)

    # --------------------------------------------------------------------------
    # Decomposition related methods.
    # --------------------------------------------------------------------------

    @cached_property
    def _sparse_quads(self):
        """Quadratic coefficients (re-)cast to sparse matrices."""
        return {xy: cvxopt.sparse(M) for xy, M in self._quads.items()}

    def _Q_and_x_or_A_and_y(self, augmentedMatrix):
        """Either :attr:`_Q_and_x` or :attr:`_A_and_y`."""
        # Find occurences of the scalar entries of the variable's isometric real
        # vectorization in the quadratic part.
        nonzero = {}
        for (x, y), A in self._sparse_quads.items():
            nonzero.setdefault(x, set())
            nonzero.setdefault(y, set())

            nonzero[x].update(A.I)
            nonzero[y].update(A.J)

        # For the augmented matrix, find additional linear occurences.
        if augmentedMatrix:
            for x, a in self._aff._sparse_linear_coefs.items():
                nonzero.setdefault(x, set())
                nonzero[x].update(a.I)

        nonzero = odict((var, sorted(nonzero[var]))
            for var in sorted(nonzero, key=lambda v: v.id))

        # Compute each (multidimensional) variable's offset in Q/A.
        offset, offsets = 0, {}
        for var, nz in nonzero.items():
            offsets[var] = offset
            offset += len(nz)

        # Compute Q.
        V, I, J = [], [], []
        for (x, y), A in self._sparse_quads.items():
            B = A[nonzero[x], nonzero[y]]

            V.extend(B.V)
            I.extend(offsets[x] + i for i in B.I)
            J.extend(offsets[y] + j for j in B.J)

        # Turn Q into A if requested.
        if augmentedMatrix:
            for x, a in self._aff._sparse_linear_coefs.items():
                b = a[nonzero[x]]

                V.extend(b.V)
                I.extend(offsets[x] + i for i in b.I)
                J.extend([offset]*len(b.J))

            V.append(self._aff._constant_coef[0])
            I.append(offset)
            J.append(offset)

            offset += 1

        # Finalize Q/A.
        Q = cvxopt.spmatrix(V, I, J, (offset,)*2, tc="d")
        Q = 0.5*(Q + Q.T)

        if not Q:
            if augmentedMatrix:
                raise ValueError(
                    "The expression {} is zero.".format(self._symbStr))
            else:
                raise ValueError(
                    "The quadratic part of {} is zero.".format(self._symbStr))

        # Compute x.
        x = None
        for var, nz in nonzero.items():
            n, r = len(nz), var.dim
            F = cvxopt.spmatrix([1]*n, range(n), nz, (n, r))
            y = AffineExpression(
                glyphs.Fn("F")(var.string), n, coefficients={var: F})
            x = y if x is None else x // y

        # Turn x into y if requested.
        if augmentedMatrix:
            x = AffineExpression.from_constant(1) if x is None else x // 1

        return Q, x

    @cached_property
    def _Q_and_x(self):
        """Pair containing :attr:`Q` and :attr:`x`."""
        return self._Q_and_x_or_A_and_y(augmentedMatrix=False)

    @cached_property
    def _A_and_y(self):
        """Pair containing :attr:`R` and :attr:`y`."""
        return self._Q_and_x_or_A_and_y(augmentedMatrix=True)

    @property
    def Q(self):
        """The coefficient matrix :math:`Q` of the expression, condensed.

        This equals the :math:`Q` of the quadratic expression
        :math:`x^TQx + a^Tx + b` but with zero rows and columns removed.

        The vector :attr:`x` is a condensed version of :math:`x` that refers to
        this matrix, so that ``q.x.T*q.Q*q.x`` equals the quadratic part
        :math:`x^TQx` of the expression ``q``.

        :raises ValueError: When the quadratic part is zero.
        """
        return self._Q_and_x[0]

    @property
    def x(self):
        """The stacked variable vector :math:`x` of the expression, condensed.

        This equals the :math:`x` of the quadratic expression
        :math:`x^TQx + a^Tx + b` but entries corresponding to zero rows and
        columns in :attr:`Q` are removed, so that ``q.x.T*q.Q*q.x`` equals the
        :math:`x^TQx` part of the expression ``q``.

        :raises ValueError: When the quadratic part is zero.
        """
        return self._Q_and_x[1]

    @property
    def A(self):
        r"""An affine-augmented quadratic coefficient matrix, condensed.

        For a quadratic expression :math:`x^TQx + a^Tx + b`, this is
        :math:`A = \begin{bmatrix}Q&\frac{a}{2}\\\frac{a^T}{2}&b\end{bmatrix}`
        but with zero rows and columns removed.

        The vector :attr:`y` is a condensed version of :math:`x` that refers to
        this matrix, so that ``q.y.T*q.A*q.y`` equals the expression ``q``.

        :raises ValueError: When the expression is zero.
        """
        return self._A_and_y[0]

    @property
    def y(self):
        """See :attr:`A`.

        :raises ValueError: When the expression is zero.
        """
        return self._A_and_y[1]

    def _L_or_M(self, augmentedMatrix):
        """Either :attr:`L` or :attr:`M`."""
        Q = self.A if augmentedMatrix else self.Q
        n = Q.size[0]

        # Define a sequence of small numbers for perturbation.
        # TODO: These numbers are empirical; find worst case values.
        spread   = [abs(v) for v in Q.V if v]
        minEps   = 1 * min(spread) * sys.float_info.epsilon
        maxEps   = n * max(spread) * sys.float_info.epsilon
        epsilons = set([0.0])
        for i in range(PSD_PERTURBATIONS):
            epsilons.add(minEps + i * (maxEps - minEps) / PSD_PERTURBATIONS)
        epsilons = sorted(epsilons)

        # Check if Q is really sparse.
        sparse = should_be_sparse(Q.size, len(Q))

        # Attempt to find L.
        L = None
        if sparse:  # Use CHOLMOD.
            F = None
            for eps in epsilons:
                P = Q + cvxopt.spdiag([eps]*n) if eps else Q  # Perturbed Q.

                if F is None:
                    F = cvxopt.cholmod.symbolic(P)

                try:
                    cvxopt.cholmod.numeric(P, F)
                except ArithmeticError:
                    pass
                else:
                    L = cvxopt.cholmod.getfactor(F)
                    break
        else:  # Use NumPy.
            Q = cvx2np(Q)
            for eps in epsilons:
                P = Q + numpy.diag([eps]*n) if eps else Q  # Perturbed Q.

                try:
                    L = numpy.linalg.cholesky(P)
                except numpy.linalg.LinAlgError:
                    pass
                else:
                    break

        if L is None:
            if not PSD_PERTURBATIONS:
                raise ArithmeticError("The expression {} has an (augmented) "
                    "quadratic form that is either not positive semidefinite "
                    "or singular. Try enabling PSD_PERTURBATIONS.")
            elif augmentedMatrix:
                raise ArithmeticError("The expression {} is not a squared norm "
                    "as its {} representation is not positive semidefinite."
                    .format(self._symbStr, glyphs.matrix(glyphs.vertcat(
                        glyphs.horicat("Q", glyphs.div("a", 2)),
                        glyphs.horicat(glyphs.div(glyphs.transp("a"), 2), "b")
                    ))))
            else:
                raise ArithmeticError("The expression {} is not convex as its "
                    "quadratic part is not positive semidefinite."
                    .format(self._symbStr))

        # Remove near-zeros in the order of n·sqrt(maxEps).
        cutoff = 2 * n * maxEps**0.5
        if sparse:
            VIJ = list(t for t in zip(L.V, L.I, L.J) if abs(t[0]) > cutoff)
            if VIJ:  # Don't zero the matrix.
                V, I, J = zip(*VIJ)
                L = cvxopt.spmatrix(V, I, J, L.size)
        else:
            mask = abs(L) <= cutoff
            if not numpy.all(mask):  # Don't zero the matrix.
                L[mask] = 0
            L = cvxopt.sparse(cvxopt.matrix(L))

        return L

    @cached_property
    def L(self):
        """The :math:`L` of an :math:`LL^T` Cholesky decomposition of :attr:`Q`.

        :returns: A CVXOPT lower triangular sparse matrix.

        :raises ValueError: When the quadratic part is zero.
        :raises ArithmeticError: When the expression is not convex, that is when
            the matrix :attr:`Q` is not numerically positive semidefinite.
        """
        return self._L_or_M(augmentedMatrix=False)

    @cached_property
    def M(self):
        """The :math:`M` of an :math:`MM^T` Cholesky decomposition of :attr:`A`.

        :returns: A CVXOPT lower triangular sparse matrix.

        :raises ValueError: When the expression is zero.
        :raises ArithmeticError: When the expression can not be written as a
            squared norm, that is when the extended matrix :attr:`A` is not
            numerically positive semidefinite.
        """
        return self._L_or_M(augmentedMatrix=True)

    @cached_property
    def quadroot(self):
        r"""Affine expression whose squared norm equals :math:`x^TQx`.

        For a convex quadratic expression ``q``, this is equal to the vector
        ``q.L.T*q.x`` with zero rows removed.

        :Construction:

        Let :math:`x^TQx` be the quadratic part of the expression with :math:`Q`
        positive semidefinite and :math:`Q = LL^T` a Cholesky decomposition.
        Then,

        .. math::
            x^TQx
            &= x^TLL^Tx \\
            &= (L^Tx)^TL^Tx \\
            &= \langle L^Tx, L^Tx \rangle \\
            &= \lVert L^Tx \rVert^2.

        Note that removing zero rows from :math:`L^Tx` does not affect the norm.

        :raises ValueError: When the quadratic part is zero.
        :raises ArithmeticError: When the expression is not convex, that is when
            the quadratic part is not numerically positive semidefinite.
        """
        L   = self.L
        n   = L.size[0]
        nz  = sorted(set(L.J))
        nnz = len(nz)

        # F removes rows corresponding to zero columns in L.
        F = cvxopt.spmatrix([1.0]*nnz, range(nnz), nz, (nnz, n))

        result = (F*L.T)*self.x
        # NOTE: AffineExpression.__mul__ always creates a fresh expression.
        result._symbStr = glyphs.Fn("quadroot({})")(self.string)
        return result

    @cached_property
    def fullroot(self):
        r"""Affine expression whose squared norm equals the expression.

        For a convex quadratic expression ``q``, this is equal to the vector
        ``q.M.T*q.y`` with zero rows removed.

        :Construction:

        For a quadratic expression :math:`x^TQx + a^Tx + b` with
        :math:`A = \begin{bmatrix}Q&\frac{a}{2}\\\frac{a^T}{2}&b\end{bmatrix}`
        positive semidefinite, let :math:`A = MM^T` be a Cholesky decomposition.
        Let further :math:`y = \begin{bmatrix}x^T & 1\end{bmatrix}^T`. Then,

        .. math::
            x^TQx + a^Tx + b
            &= y^TAy \\
            &= y^TMM^Ty \\
            &= (M^Ty)^TM^Ty \\
            &= \langle M^Ty, M^Ty \rangle \\
            &= \lVert M^Ty \rVert^2.

        Note that removing zero rows from :math:`M^Ty` does not affect the norm.

        :raises ValueError: When the expression is zero.
        :raises ArithmeticError: When the expression is not convex, that is when
            the quadratic part is not numerically positive semidefinite.
        """
        M   = self.M
        n   = M.size[0]
        nz  = sorted(set(M.J))
        nnz = len(nz)

        # F removes rows corresponding to zero columns in M.
        F = cvxopt.spmatrix([1.0]*nnz, range(nnz), nz, (nnz, n))

        result = (F*M.T)*self.y
        # NOTE: AffineExpression.__mul__ always creates a fresh expression.
        result._symbStr = glyphs.Fn("fullroot({})")(self.string)
        return result

    @property
    def rank(self):
        """The length of the vector :attr:`quadroot`.

        Up to numerical considerations, this is the rank of the (convex)
        quadratic coefficient matrix :math:`Q` of the expression.

        :raises ArithmeticError: When the expression is not convex, that is when
            the quadratic part is not numerically positive semidefinite.
        """
        try:
            return len(self.quadroot)
        except ValueError:
            return 0  # No quadratic part.

    @cached_property
    def num_quad_terms(self):
        """The number of terms in the simplified quadratic form."""
        num_terms = 0
        for A in self._sparse_quads.values():
            for i, j in zip(A.I, A.J):
                if i >= j:
                    num_terms += 1
        return num_terms

    @property
    def is_squared_norm(self):
        r"""Whether the expression can be written as a squared norm.

        If this is :obj:`True`, then the there is a coefficient vector
        :math:`c` such that

        .. math::
            \lVert c^T \begin{bmatrix} x \\ 1 \end{bmatrix} \rVert^2
            = x^TQx + a^Tx + b.

        If the expression is also nonzero, then :attr:`fullroot` is
        :math:`c^T \begin{bmatrix} x^T & 1 \end{bmatrix}^T` with zero entries
        removed.
        """
        try:
            self.M
        except ValueError:
            assert self.is0
            return True
        except ArithmeticError:
            return False
        else:
            return True

    # --------------------------------------------------------------------------
    # Properties and functions that describe the expression.
    # --------------------------------------------------------------------------

    @property
    @deprecated("2.2", "This property will be removed in a future release.")
    def quadratic_forms(self):
        """The quadratic forms as a map from variable pairs to sparse matrices.

        .. warning::

            Do not modify the returned matrices.
        """
        return MappingProxyType(self._sparse_quads)

    @property
    def is0(self):
        """Whether the quadratic expression is zero."""
        return not self._quads and self._aff.is0

    @property
    def scalar_factors(self):
        r"""Decomposition into scalar real affine expressions.

        If the expression is known to be equal to :math:`ab` for scalar real
        affine expressions :math:`a` and :math:`b`, this is the pair ``(a, b)``.
        Otherwise, this is :obj:`None`.

        Note that if :math:`a = b`, then also ``a is b`` in the returned tuple.
        """
        return self._sf

    # --------------------------------------------------------------------------
    # Methods and properties that return modified copies.
    # --------------------------------------------------------------------------

    @cached_property
    def quad(self):
        """Quadratic part :math:`x^TQx` of the quadratic expression."""
        factors = self._sf if self._sf and self._aff.is0 else None

        return QuadraticExpression(glyphs.quadpart(self._symbStr), self._quads,
            scalarFactors=factors, copyDecomposition=self)

    @cached_property
    def aff(self):
        """Affine part :math:`a^Tx + b` of the quadratic expression."""
        return self._aff.renamed(glyphs.affpart(self._symbStr))

    @cached_property
    def lin(self):
        """Linear part :math:`a^Tx` of the quadratic expression."""
        return self._aff.lin.renamed(glyphs.linpart(self._symbStr))

    @cached_property
    def cst(self):
        """Constant part :math:`b` of the quadratic expression."""
        return self._aff.cst.renamed(glyphs.cstpart(self._symbStr))

    # --------------------------------------------------------------------------
    # Constraint-creating operators, and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        if issubclass(other.clstype, QuadraticExpression):
            raise NotImplementedError("Constraint outcome prediction not "
                "supported when comparing two quadratic expressions.")

        if relation == operator.__le__:
            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                if subtype.convex:
                    return ConvexQuadraticConstraint.make_type(
                        qterms=subtype.qterms, rank=subtype.rank,
                        haslin=(subtype.haslin or not other.subtype.constant))
                else:
                    return NonconvexQuadraticConstraint.make_type(
                        qterms=subtype.qterms)
        elif relation == operator.__ge__:
            if issubclass(other.clstype, AffineExpression) \
            and other.subtype.dim == 1:
                if subtype.concave:
                    return ConvexQuadraticConstraint.make_type(
                        qterms=subtype.qterms, rank=subtype.rank,
                        haslin=(subtype.haslin or not other.subtype.constant))
                elif subtype.factors and other.subtype.constant \
                and other.subtype.nonneg:
                    # See __ge__ for the variants below.

                    # Variant 1:
                    return ConicQuadraticConstraint.make_type(
                        qterms=subtype.qterms, conic_argdim=1,
                        rotated=(subtype.factors == 2))

                    # Variant 2:
                    # return RSOCConstraint.make_type(argdim=1)

                    # Variant 3:
                    # if subtype.factors == 1:
                    #     return AffineConstraint.make_type(dim=1, eq=False)
                    # else:
                    #     return RSOCConstraint.make_type(argdim=1)
                else:
                    return NonconvexQuadraticConstraint.make_type(
                        qterms=subtype.qterms)

        return NotImplemented

    @convert_operands(sameShape=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        LE = Constraint.LE

        if isinstance(other, QuadraticExpression):
            le0 = self - other  # Unpredictable outcome.
            if le0.convex:
                return ConvexQuadraticConstraint(self, LE, other)
            elif self.is_squared_norm and other.scalar_factors:
                return ConicQuadraticConstraint(self, LE, other)
            else:
                return NonconvexQuadraticConstraint(self, LE, other)
        elif isinstance(other, AffineExpression):
            if self.convex:
                return ConvexQuadraticConstraint(self, LE, other)
            else:
                return NonconvexQuadraticConstraint(self, LE, other)
        else:
            return NotImplemented

    @convert_operands(sameShape=True)
    @validate_prediction
    @refine_operands()
    def __ge__(self, other):
        GE = Constraint.GE

        if isinstance(other, QuadraticExpression):
            le0 = other - self  # Unpredictable outcome.
            if le0.convex:
                return ConvexQuadraticConstraint(self, GE, other)
            elif other.is_squared_norm and self.scalar_factors:
                return ConicQuadraticConstraint(self, GE, other)
            else:
                return NonconvexQuadraticConstraint(self, GE, other)
        elif isinstance(other, AffineExpression):
            if self.concave:
                return ConvexQuadraticConstraint(self, GE, other)
            elif self._sf and other.constant and other.value >= 0:
                # Handle the case of c <= x*x and c <= x*y with c pos. constant.
                root = Constant(glyphs.sqrt(other.string), other.value**0.5)

                # Variant 1: Return a ConicQuadraticConstraint that can be
                #     reformulated to an (R)SOCConstraint. This is consistent
                #     with all other quadratic expression comparison outcomes.
                other = QuadraticExpression(other.string, affinePart=other,
                    scalarFactors=(root, root))  # Unrefined QE.
                return ConicQuadraticConstraint(self, GE, other)

                # Variant 2: Return an RSOCConstraint. This is somewhat
                #     consistent with Norm returning an SOCConstraint and it is
                #     also legacy behavior. However this may not be what the
                #     user wants, because it silently forces x, y >= 0!
                # return RSOCConstraint(root, self._sf[0], self._sf[1],
                #     customString = glyphs.le(other.string, self.string))

                # Variant 3: Return either an RSOCConstraint or an affine
                #     constraint that equals an SOCConstraint. This is faster
                #     than Variant 2 but even more confusing, because the
                #     constraint would transform from e.g. 1 <= x*x to
                #     sqrt(1) <= x without any mention of cones.
                # if self._sf[0] is self._sf[1]:
                #     # NOTE: The following is not allowed by SOCConstraint
                #     #       since this is effectively an affine constraint.
                #     # return SOCConstraint(root, self._sf[0])
                #
                #     return root <= self._sf[0]
                # else:
                #     return RSOCConstraint(root, *self._sf)
            else:
                return NonconvexQuadraticConstraint(self, GE, other)
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
