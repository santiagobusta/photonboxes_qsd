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

"""Implements a Cartesian product cone."""

import operator
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from ..constraints import ProductConeConstraint
from .cone import Cone
from .exp_affine import AffineExpression

_API_START = api_start(globals())
# -------------------------------


class ProductCone(Cone):
    """A real Cartesian product cone."""

    @classmethod
    def _unpack(cls, nested_cones):
        flattened_cones = []
        for inner_cone in nested_cones:
            if isinstance(inner_cone, ProductCone):
                flattened_cones.extend(cls._unpack(inner_cone.cones))
            else:
                flattened_cones.append(inner_cone)
        return flattened_cones

    def __init__(self, *cones):
        """Construct a :class:`ProductCone`.

        :param list(picos.expressions.Cone) cones:
            A sequence of cones to build the product cone from. May include
            other product cones that will be "unpacked" first.
        """
        if not cones or not all(isinstance(cone, Cone) for cone in cones):
            raise TypeError("Must initialize product cones with a nonempty "
                "sequence of cone instances.")

        if not all(cone.dim for cone in cones):
            raise TypeError("Product cones must be built from cones with a "
                "fixed dimensionality.")

        if any(cone.mutables for cone in cones):
            raise NotImplementedError("Product cones may not include cones "
                "whose definition depends on mutables.")

        # Unpack nested product cones.
        cones = self._unpack(cones)

        dim = sum(cone.dim for cone in cones)

        Cone.__init__(self, dim, "Product Cone", glyphs.prod(glyphs.sep(
            "Ci", glyphs.element("i", glyphs.interval(len(cones))))))

        self._cones = tuple(cones)

    @property
    def cones(self):
        """The cones that make up the product cone as a tuple."""
        return self._cones

    def _get_mutables(self):
        return frozenset()  # See NotImplementedError in __init__.

    def _replace_mutables(self):
        return self  # See NotImplementedError in __init__.

    Subtype = namedtuple("Subtype", ("dim", "cones"))

    def _get_subtype(self):
        # NOTE: Storing dim is redundant but simplifies _predict.
        return self.Subtype(
            self.dim, cones=tuple(cone.type for cone in self._cones))

    @property
    def refined(self):
        """Overwrite :attr:`~.set.Set.refined`."""
        if len(self._cones) == 1:
            return self._cones[0]
        else:
            return self

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        if relation == operator.__rshift__:
            if issubclass(other.clstype, AffineExpression) \
            and subtype.dim == other.subtype.dim:
                if len(subtype.cones) == 1:
                    return subtype.cones[0].predict(operator.__rshift__, other)

                return ProductConeConstraint.make_type(
                    dim=subtype.dim, cones=subtype.cones)

        return Cone._predict_base(cls, subtype, relation, other)

    def _rshift_implementation(self, element):
        if isinstance(element, AffineExpression):
            self._check_dimension(element)

            # HACK: Mimic refinement: Do not produce a ProductConeConstraint
            #       to represent a basic conic inequality.
            # TODO: Add a common base class for Expression and Set that allows
            #       proper refinement also for instances of the latter.
            if len(self.cones) == 1:
                return self.cones[0] >> element

            return ProductConeConstraint(element, self)

        # Handle scenario uncertainty for all cones.
        return Cone._rshift_base(self, element)

    @cached_property
    def dual_cone(self):
        """Implement :attr:`.cone.Cone.dual_cone`."""
        return self.__class__(*(cone.dual_cone for cone in self._cones))


# --------------------------------------
__all__ = api_end(_API_START, globals())
