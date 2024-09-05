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

"""Implements :class:`ExponentialCone`."""

import operator
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..constraints import ExpConeConstraint
from .cone import Cone
from .exp_affine import AffineExpression

_API_START = api_start(globals())
# -------------------------------


class ExponentialCone(Cone):
    r"""The exponential cone.

    Represents the convex cone
    :math:`\operatorname{cl}\{(x,y,z): y \exp(\frac{z}{y}) \leq x, x,y > 0\}`.
    """

    def __init__(self):
        """Construct an exponential cone."""
        typeStr = "Exponential Cone"
        symbStr = glyphs.closure(glyphs.set(glyphs.sep(
            glyphs.col_vectorize("x", "y", "z"), ", ".join([
                glyphs.le(
                    glyphs.mul("y", glyphs.exp(glyphs.div("z", "y"))), "x"),
                glyphs.gt("x", 0),
                glyphs.gt("y", 0)
            ]))))

        Cone.__init__(self, 3, typeStr, symbStr)

    def _get_mutables(self):
        return frozenset()

    def _replace_mutables(self):
        return self

    Subtype = namedtuple("Subtype", ())

    def _get_subtype(self):
        return self.Subtype()

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        if relation == operator.__rshift__:
            if issubclass(other.clstype, AffineExpression):
                if other.subtype.dim == 3:
                    return ExpConeConstraint.make_type()

        return Cone._predict_base(cls, subtype, relation, other)

    def _rshift_implementation(self, element):
        if isinstance(element, AffineExpression):
            if len(element) != 3:
                raise TypeError("Elements of the exponential cone must be "
                    "three-dimensional.")

            return ExpConeConstraint(element)

        # Handle scenario uncertainty for all cones.
        return Cone._rshift_base(self, element)

    @property
    def dual_cone(self):
        """Implement :attr:`.cone.Cone.dual_cone`."""
        raise NotImplementedError(
            "PICOS does not have an explicit representation for the dual of "
            "the exponential cone yet.")


# --------------------------------------
__all__ = api_end(_API_START, globals())
