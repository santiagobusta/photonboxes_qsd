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

"""Implementation of :class:`ProductConeConstraint`."""

import operator
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from .constraint import ConicConstraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class ProductConeConstraint(ConicConstraint):
    """Confines an element inside a real Cartesian product cone."""

    class Conversion(ConstraintConversion):
        """Cartesian product cone membership conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ..expressions import AffineExpression

            for cone_type in subtype.cones:
                shape = (cone_type.subtype.dim, 1)

                # HACK: Assume that every relevant slice of the product cone
                #       member is nonconstant.
                # NOTE: This works as long as the detailed type of any
                #       constraint created via the membership operator << on a
                #       cone depends only on the member's dimensionality.
                member_type = AffineExpression.make_type(
                    shape=shape, constant=False, nonneg=False)

                constraint = member_type.predict(operator.__lshift__, cone_type)

                yield ("con", constraint, 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            # NOTE: Clone problem for extra safety. This is probably not needed.
            return con._conversion.clone()

        @classmethod
        def dual(cls, auxVarPrimals, auxConDuals, options):
            """Implement :meth:`~.constraint.ConstraintConversion.dual`."""
            from ..expressions.data import cvxopt_vcat

            if None in auxConDuals:
                return None

            return cvxopt_vcat(auxConDuals)

    def __init__(self, element, cone):
        """Construct a :class:`ProductConeConstraint`.

        :param ~picos.expressions.AffineExpression element:
            The element confined in the product cone.
        :param ~picos.expressions.ProductCone cone:
            The product cone.
        """
        from ..expressions import AffineExpression, ProductCone

        assert isinstance(element, AffineExpression)
        assert isinstance(cone, ProductCone)

        self.element = element.vec
        self.cone = cone

        super(ProductConeConstraint, self).__init__("Product Cone")

    @cached_property
    def conic_membership_form(self):
        """Implement for :class:`~.constraint.ConicConstraint`."""
        return self.element, self.cone

    Subtype = namedtuple("Subtype", ("dim", "cones"))  # Same as ProductCone.

    def _subtype(self):
        return self.Subtype(*self.cone.subtype)

    @classmethod
    def _cost(cls, subtype):
        return subtype.dim

    def _expression_names(self):
        yield "element"
        yield "cone"

    def _str(self):
        return glyphs.element(self.element.string, self.cone.string)

    def _get_size(self):
        return (self.cone.dim, 1)

    def _get_slack(self):
        return min(con.slack for con in self._conversion.constraints.values())

    @cached_property
    def _conversion(self):
        """Cached version of Conversion.convert as _get_slack also needs it."""
        from ..expressions import ProductCone
        from ..modeling import Problem

        x = self.element
        C = self.cone

        P = Problem()

        offset = 0
        for Ci in C.cones:
            assert not isinstance(Ci, ProductCone), \
                "Product cones are supposed to not contain other product cones."

            xi = x[offset:offset + Ci.dim]
            offset += Ci.dim

            P.add_constraint(xi << Ci)

        return P


# --------------------------------------
__all__ = api_end(_API_START, globals())
