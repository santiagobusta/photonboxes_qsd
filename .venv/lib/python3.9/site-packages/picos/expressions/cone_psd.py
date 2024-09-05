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

"""Implements the positive semidefinite cone."""

import operator
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..constraints.uncertain import ScenarioUncertainConicConstraint
from .cone import Cone
from .exp_affine import AffineExpression, ComplexAffineExpression
from .expression import ExpressionType
from .uncertain import ScenarioPerturbationSet, UncertainAffineExpression

_API_START = api_start(globals())
# -------------------------------


class PositiveSemidefiniteCone(Cone):
    r"""The positive semidefinite cone.

    Unlike other :class:`cones <.cone.Cone>` which are defined only on
    :math:`\mathbb{K}^n`, this cone accepts both symmetric and hermitian
    matrices as well as their
    :attr:`special vectorization <.exp_biaffine.BiaffineExpression.svec>`
    as members.

    :Example:

    >>> from picos import Constant, PositiveSemidefiniteCone
    >>> R = Constant("R", range(16), (4, 4))
    >>> S = R + R.T
    >>> S.shape
    (4, 4)
    >>> S.svec.shape
    (10, 1)
    >>> S.svec << PositiveSemidefiniteCone()  # Constrain the matrix via svec().
    <4×4 LMI Constraint: R + Rᵀ ≽ 0>
    >>> C = S << PositiveSemidefiniteCone(); C  # Constrain the matrix directly.
    <4×4 LMI Constraint: R + Rᵀ ≽ 0>
    >>> C.conic_membership_form[0]      # The conic form still refers to svec().
    <10×1 Real Constant: svec(R + Rᵀ)>
    >>> C.conic_membership_form[1]
    <10-dim. Positive Semidefinite Cone: {svec(A) : xᵀ·A·x ≥ 0 ∀ x}>
    """

    def __init__(self, dim=None):
        r"""Construct a :class:`PositiveSemidefiniteCone`.

        If a fixed dimensionality is given, this must be the dimensiona of the
        special vectorization. For a :math:`n \times n` matrix, this is
        :math:`\frac{n(n + 1)}{2}`.
        """
        Cone.__init__(self, dim, "Positive Semidefinite Cone",
            glyphs.set(glyphs.sep(glyphs.svec("A"), glyphs.forall(glyphs.ge(
                glyphs.mul(glyphs.mul(glyphs.transp("x"), "A"), "x"),
                glyphs.scalar(0)), "x"))))

    def _get_mutables(self):
        return frozenset()

    def _replace_mutables(self):
        return self

    Subtype = namedtuple("Subtype", ("dim",))

    def _get_subtype(self):
        return self.Subtype(self.dim)

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        if relation == operator.__rshift__:
            if issubclass(other.clstype, (
                    ComplexAffineExpression, UncertainAffineExpression)):
                m, n = other.subtype.shape

                if m == n:
                    svec_length = int(0.5*n*(n + 1))

                    if subtype.dim and subtype.dim != svec_length:
                        return NotImplemented

                    if issubclass(other.clstype, ComplexAffineExpression):
                        # Other is already a square matrix.
                        matrix = other
                    else:
                        # Predict the vector svec(other).
                        vector = other.clstype.make_type(
                            shape=(svec_length, 1),
                            universe_type=other.subtype.universe_type)
                elif 1 in other.subtype.shape:
                    if subtype.dim and subtype.dim != other.subtype.dim:
                        return NotImplemented

                    if issubclass(other.clstype, ComplexAffineExpression):
                        # Predict the square matrix desvec(other).
                        n = 0.5*((8*other.subtype.dim + 1)**0.5 - 1)
                        if int(n) != n:
                            return NotImplemented
                        n = int(n)
                        matrix = other.clstype.make_type(
                            shape=(n, n), constant=other.subtype.constant,
                            nonneg=other.subtype.nonneg)
                    else:
                        # Other is already a vector.
                        vector = other
                else:
                    return NotImplemented

                if issubclass(other.clstype, ComplexAffineExpression):
                    zero = AffineExpression.make_type(
                        shape=matrix.subtype.shape, constant=True, nonneg=True)

                    return matrix.clstype._predict(
                        matrix.subtype, operator.__rshift__, zero)
                elif issubclass(
                        other.subtype.universe_type.clstype,
                        ScenarioPerturbationSet):
                    dim = vector.subtype.dim
                    count = other.subtype.universe_type.subtype.scenario_count
                    cone = ExpressionType(cls, subtype)

                    return ScenarioUncertainConicConstraint.make_type(
                        dim=dim, scenario_count=count, cone_type=cone)
                else:
                    return NotImplemented

        return NotImplemented

    def _rshift_implementation(self, element):
        if isinstance(element, (
                ComplexAffineExpression, UncertainAffineExpression)):
            if element.square:
                # Mimic _check_dimension.
                n = element.shape[0]
                d = int(0.5*n*(n + 1))
                if self.dim and self.dim != d:
                    raise TypeError(
                        "The shape {} of {} implies a {}-dimensional "
                        "svec-representation which does not match the fixed "
                        "dimensionality {} of the cone {}.".format(
                            glyphs.shape(element.shape), element.string, d,
                            self.dim, self.string))

                if isinstance(element, ComplexAffineExpression):
                    return element >> 0
                elif isinstance(element.universe, ScenarioPerturbationSet):
                    return ScenarioUncertainConicConstraint(element.svec, self)
                else:
                    raise TypeError("LMIs with uncertainty parameterized "
                        "through a {} are not supported.".format(
                        element.universe.__class__.__name__))
            elif 1 in element.shape:
                self._check_dimension(element)

                if isinstance(element, ComplexAffineExpression):
                    return element.desvec >> 0
                elif isinstance(element.universe, ScenarioPerturbationSet):
                    return ScenarioUncertainConicConstraint(element, self)
                else:
                    raise TypeError("LMIs with uncertainty parameterized "
                        "through a {} are not supported.".format(
                        element.universe.__class__.__name__))
            else:
                raise TypeError("The {} expression {} is neither square nor a "
                    "vector so it cannot be constrained to be in the positive "
                    "semidefinite cone.".format(element.shape, element.string))

        return NotImplemented

    @property
    def dual_cone(self):
        """Implement :attr:`.cone.Cone.dual_cone`."""
        return self


# --------------------------------------
__all__ = api_end(_API_START, globals())
