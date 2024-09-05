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

"""Implements trivial cones."""

import operator
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from ..constraints import (AffineConstraint, ComplexAffineConstraint,
                           DummyConstraint)
from .cone import Cone
from .exp_affine import AffineExpression, ComplexAffineExpression

_API_START = api_start(globals())
# -------------------------------


class ZeroSpace(Cone):
    r"""The set containing zero."""

    def __init__(self, dim=None):
        """Construct a :class:`ZeroSpace`."""
        Cone.__init__(self, dim, "Zero Space", glyphs.set(glyphs.scalar(0)))

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
                    dim=other.subtype.dim, eq=True)
            elif issubclass(other.clstype, ComplexAffineExpression) \
            and not subtype.dim or subtype.dim == other.subtype.dim:
                return ComplexAffineConstraint.make_type(
                    dim=other.subtype.dim)

        return Cone._predict_base(cls, subtype, relation, other)

    def _rshift_implementation(self, element):
        if isinstance(element, ComplexAffineExpression):
            self._check_dimension(element)

            return element == 0

        # Handle scenario uncertainty for all cones.
        return Cone._rshift_base(self, element)

    @cached_property
    def dual_cone(self):
        """Implement :attr:`.cone.Cone.dual_cone`."""
        return TheField(dim=self.dim)


class TheField(Cone):
    r"""The real or complex field."""

    def __init__(self, dim=None):
        """Construct a :class:`TheField`."""
        Cone.__init__(self, dim, "The Field", "F")

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
            if issubclass(other.clstype, ComplexAffineExpression) \
            and not subtype.dim or subtype.dim == other.subtype.dim:
                return DummyConstraint.make_type()

        return Cone._predict_base(cls, subtype, relation, other)

    def _rshift_implementation(self, element):
        if isinstance(element, ComplexAffineExpression):
            self._check_dimension(element)

            return DummyConstraint(element)

        # Handle scenario uncertainty for all cones.
        return Cone._rshift_base(self, element)

    @cached_property
    def dual_cone(self):
        """Implement :attr:`.cone.Cone.dual_cone`."""
        return ZeroSpace(dim=self.dim)


# --------------------------------------
__all__ = api_end(_API_START, globals())
