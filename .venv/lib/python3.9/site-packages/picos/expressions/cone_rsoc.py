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

"""Implements :class:`RotatedSecondOrderCone`."""

import operator
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from ..constraints import RSOCConstraint
from .cone import Cone
from .exp_affine import AffineExpression

_API_START = api_start(globals())
# -------------------------------


class RotatedSecondOrderCone(Cone):
    r"""The (narrowed or widened) rotated second order cone.

    .. _rotatedcone:

    For :math:`n \in \mathbb{Z}_{\geq 3}` and :math:`p \in \mathbb{R}_{> 0}`,
    represents the convex cone

    .. math::

        \mathcal{R}_{p}^n = \left\{
            x \in \mathbb{R}^n
        ~\middle|~
            p x_1 x_2 \geq \sum_{i = 3}^n x_i^2 \land x_1, x_2 \geq 0
        \right\}.

    For :math:`p = 2`, this is the standard rotated second order cone
    :math:`\mathcal{R}^n` obtained by rotating the
    :class:`second order cone <picos.SecondOrderCone>` :math:`\mathcal{Q}^n`
    by :math:`\frac{\pi}{4}` in the :math:`(x_1, x_2)` plane.

    The default instance of this class has :math:`p = 1`, which can be
    understood as a narrowed version of the standard cone. This is more
    convenient for defining the primal problem but it should be noted that
    :math:`\mathcal{R}_{1}^n` is not self-dual, so working with
    :math:`p = 2` may seem more natural when the dual problem is of interest.

    :Dual cone:

    The dual cone is

    .. math::

        \left(\mathcal{R}_{p}^n\right)^* = \left\{
            x \in \mathbb{R}^n
        ~\middle|~
            \frac{4}{p} x_1 x_2 \geq \sum_{i = 2}^n x_i^2 \land x_1, x_2 \geq 0
        \right\}=\mathcal{R}_{4/p}^n.

    The cone is thus self-dual for :math:`p = 2`.
    """

    def __init__(self, p=1, dim=None):
        """Construct a rotated second order cone.

        :param float p:
            The positive factor :math:`p` in the definition.
        """
        try:
            p = float(p)
        except Exception as error:
            raise TypeError("Failed to load the parameter 'p' as a float.") \
                from error

        if p <= 0:
            raise ValueError("The parameter 'p' must be positive.")

        self._p = p

        if dim and dim < 3:
            raise ValueError("The minimal dimension for {} is {}."
                .format(self.__class__.__name__, 3))

        typeStr = "Rotated Second Order Cone"
        if p < 2:
            typeStr = "Narrowed " + typeStr
        elif p > 2:
            typeStr = "Widened " + typeStr

        symbStr = glyphs.set(glyphs.sep(
            glyphs.col_vectorize("u", "v", "x"), glyphs.and_(
            glyphs.le(
                glyphs.squared(glyphs.norm("x")),
                glyphs.clever_mul(glyphs.scalar(p), glyphs.mul("u", "v"))),
            glyphs.ge("u", 0))))

        Cone.__init__(self, dim, typeStr, symbStr)

    @property
    def p(self):
        """A narrowing (:math:`p < 2`) or widening (:math:`p > 2`) factor."""
        return self._p

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
            if issubclass(other.clstype, AffineExpression) \
            and not subtype.dim or subtype.dim == other.subtype.dim \
            and other.subtype.dim >= 3:
                return RSOCConstraint.make_type(other.subtype.dim - 2)

        return Cone._predict_base(cls, subtype, relation, other)

    def _rshift_implementation(self, element):
        if isinstance(element, AffineExpression):
            self._check_dimension(element)

            if len(element) < 3:
                raise TypeError("Elements of the rotated second order cone must"
                    " be at least three-dimensional.")

            element = element.vec

            return RSOCConstraint(element[2:], self.p * element[0], element[1])

        # Handle scenario uncertainty for all cones.
        return Cone._rshift_base(self, element)

    @cached_property
    def dual_cone(self):
        """Implement :attr:`.cone.Cone.dual_cone`."""
        return self.__class__(p=(4.0/self._p), dim=self.dim)


# --------------------------------------
__all__ = api_end(_API_START, globals())
