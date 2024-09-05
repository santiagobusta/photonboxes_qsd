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

"""Implements :class:`SecondOrderCone`."""

import operator
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..constraints import SOCConstraint
from .cone import Cone
from .exp_affine import AffineExpression

_API_START = api_start(globals())
# -------------------------------


class SecondOrderCone(Cone):
    r"""The second order cone.

    .. _lorentz:

    Also known as the quadratic, :math:`2`-norm, Lorentz, or ice cream cone.

    For :math:`n \in \mathbb{Z}_{\geq 2}`, represents the convex cone

    .. math::

        \mathcal{Q}^n = \left\{
            x \in \mathbb{R}^n
        ~\middle|~
            x_1 \geq \sqrt{\sum_{i = 2}^n x_i^2}
        \right\}.

    :Dual cone:

    The second order cone as defined above is self-dual.
    """

    def __init__(self, dim=None):
        """Construct a second order cone."""
        if dim and dim < 2:
            raise ValueError("The minimal dimension for {} is {}."
                .format(self.__class__.__name__, 2))

        typeStr = "Second Order Cone"
        symbStr = glyphs.set(glyphs.sep(
            glyphs.col_vectorize("t", "x"), glyphs.le(glyphs.norm("x"), "t")))

        Cone.__init__(self, dim, typeStr, symbStr)

    def _get_mutables(self):
        return frozenset()

    def _replace_mutables(self):
        return self

    Subtype = namedtuple("Subtype", ("dim",))

    def _get_subtype(self):
        return self.Subtype(self.dim)

    @classmethod
    def _predict(cls, subtype, relation, other):
        from .uncertain import UncertainAffineExpression

        assert isinstance(subtype, cls.Subtype)

        if relation == operator.__rshift__:
            if issubclass(other.clstype,
                (AffineExpression, UncertainAffineExpression)) \
            and not subtype.dim or subtype.dim == other.subtype.dim \
            and other.subtype.dim >= 2:
                if issubclass(other.clstype, UncertainAffineExpression):
                    raise NotImplementedError("Cannot predict the outcome "
                        "of constraining an uncertain affine expression to the "
                        "second order cone.")

                return SOCConstraint.make_type(other.subtype.dim - 1)

        return Cone._predict_base(cls, subtype, relation, other)

    def _rshift_implementation(self, element):
        from .uncertain import ConicPerturbationSet, UncertainAffineExpression

        if isinstance(element, (AffineExpression, UncertainAffineExpression)):
            self._check_dimension(element)

            if len(element) < 2:
                raise TypeError("Elements of the second order cone must be "
                    "at least two-dimensional.")

            element = element.vec

            if isinstance(element, AffineExpression):
                return SOCConstraint(element[1:], element[0])
            else:
                if isinstance(element.universe, ConicPerturbationSet):
                    # Unpredictable case: Outcome depends on whether slices of
                    # the element remain uncertain.
                    return abs(element[1:]) <= element[0]

        # Handle scenario uncertainty for all cones.
        return Cone._rshift_base(self, element)

    @property
    def dual_cone(self):
        """Implement :attr:`.cone.Cone.dual_cone`."""
        return self


# --------------------------------------
__all__ = api_end(_API_START, globals())
