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

"""Implements the nonnegative orthant cone."""

import operator
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..constraints import AffineConstraint
from ..constraints.uncertain import ConicallyUncertainAffineConstraint
from .cone import Cone
from .exp_affine import AffineExpression
from .uncertain.pert_conic import ConicPerturbationSet
from .uncertain.uexp_affine import UncertainAffineExpression

_API_START = api_start(globals())
# -------------------------------


class NonnegativeOrthant(Cone):
    """The nonnegative orthant."""

    def __init__(self, dim=None):
        """Construct a :class:`NonnegativeOrthant`."""
        Cone.__init__(self, dim, "Nonnegative Orthant",
            glyphs.set(glyphs.ge("x", glyphs.scalar(0))))

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
            and not subtype.dim or subtype.dim == other.subtype.dim:
                return AffineConstraint.make_type(
                    dim=other.subtype.dim, eq=False)
            elif issubclass(other.clstype, UncertainAffineExpression) \
            and not subtype.dim or subtype.dim == other.subtype.dim:
                universe = other.subtype.universe_type

                if issubclass(universe.clstype, ConicPerturbationSet):
                    return ConicallyUncertainAffineConstraint.make_type(
                        dim=other.subtype.dim,
                        universe_subtype=universe.subtype)

        return Cone._predict_base(cls, subtype, relation, other)

    def _rshift_implementation(self, element):
        if isinstance(element, AffineExpression):
            self._check_dimension(element)

            return element >= 0
        elif isinstance(element, UncertainAffineExpression):
            self._check_dimension(element)

            if isinstance(element.universe, ConicPerturbationSet):
                return ConicallyUncertainAffineConstraint(-element)

        # Handle scenario uncertainty for all cones.
        return Cone._rshift_base(self, element)

    @property
    def dual_cone(self):
        """Implement :attr:`.cone.Cone.dual_cone`."""
        return self


# --------------------------------------
__all__ = api_end(_API_START, globals())
