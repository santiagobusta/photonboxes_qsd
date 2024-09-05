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

"""Implementation of :class:`DummyConstraint`."""

from collections import namedtuple

from ..apidoc import api_end, api_start
from ..caching import cached_property
from .constraint import ConicConstraint

_API_START = api_start(globals())
# -------------------------------


class DummyConstraint(ConicConstraint):
    """An explicit way to *not* put a bound on an affine expression.

    This is produced when declaring an expression a member of the trivial
    :class:`~.expressions.TheField` cone.

    A constraint of this type can be used to pass a variable to a solver that
    does not otherwise appear in the problem.
    """

    def __init__(self, x):
        """Construct a :class:`DummyConstraint`."""
        from ..expressions import ComplexAffineExpression

        assert isinstance(x, ComplexAffineExpression)

        self.x = x

        super(DummyConstraint, self).__init__("Dummy")

    @cached_property
    def conic_membership_form(self):
        """Implement for :class:`~.constraint.ConicConstraint`."""
        from ..expressions import TheField
        return self.x.vec, TheField(dim=len(self.x))

    Subtype = namedtuple("Subtype", ())

    def _subtype(self):
        return self.Subtype()

    @classmethod
    def _cost(cls, subtype):
        return 0

    def _expression_names(self):
        yield "x"

    def _str(self):
        return "{} is {}".format(
            self.x.string, "complex" if self.x.complex else "real")

    def _get_size(self):
        return (len(self.x), 1)

    def _get_slack(self):
        return float("inf")


# --------------------------------------
__all__ = api_end(_API_START, globals())
