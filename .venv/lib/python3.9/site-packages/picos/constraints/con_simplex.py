# ------------------------------------------------------------------------------
# Copyright (C) 2012-2017 Guillaume Sagnol
# Copyright (C) 2018-2019 Maximilian Stahlberg
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

"""Implementation of :class:`SimplexConstraint`."""

from collections import namedtuple

import cvxopt

from .. import glyphs
from ..apidoc import api_end, api_start
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class SimplexConstraint(Constraint):
    """(Symmetrized, truncated) simplex membership constraint."""

    class AffineConversion(ConstraintConversion):
        """Simplex membership to affine constraint conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ..expressions import RealVariable as RV
            from . import AffineConstraint as AC

            n           = subtype.argdim
            truncated   = subtype.truncated
            symmetrized = subtype.symmetrized

            if not truncated and not symmetrized:
                yield ("con", AC.make_type(dim=(n + 1), eq=False), 1)
            elif truncated and not symmetrized:
                yield ("con", AC.make_type(dim=(2*n + 1), eq=False), 1)
            elif not truncated and symmetrized:
                yield ("var", RV.make_var_type(dim=n, bnd=0), 1)
                yield ("con", AC.make_type(dim=(n + 1), eq=False), 2)
                yield ("con", AC.make_type(dim=1, eq=False), 1)
            else:
                yield ("var", RV.make_var_type(dim=n, bnd=0), 1)
                yield ("con", AC.make_type(dim=(n + 1), eq=False), 2)
                yield ("con", AC.make_type(dim=1, eq=False), 1)
                yield ("con", AC.make_type(dim=n, eq=False), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..expressions import RealVariable
            from ..modeling import Problem

            simplex     = con.simplex
            truncated   = simplex.truncated
            symmetrized = simplex.symmetrized

            x = con.element.vec
            n = len(x)
            r = simplex.radius

            P = Problem()

            if not truncated and not symmetrized:
                aff = -x // (1 | x)
                rhs = cvxopt.sparse([0]*n)
                P.add_constraint(aff <= rhs // r)
            elif truncated and not symmetrized:
                aff = x // -x // (1 | x)
                rhs = cvxopt.sparse([1]*n + [0]*n)
                P.add_constraint(aff <= rhs // r)
            elif not truncated and symmetrized:
                v = RealVariable("__v", n)
                P.add_constraint(x <= v)
                P.add_constraint(-x <= v)
                P.add_constraint((1 | v) <= r)
            else:  # truncated and symmetrized
                v = RealVariable("__v", n)
                P.add_constraint(x <= v)
                P.add_constraint(-x <= v)
                P.add_constraint((1 | v) <= r)
                P.add_constraint(v <= 1)

            return P

    def __init__(self, simplex, element):
        """Construct a :class:`SimplexConstraint`.

        :param ~picos.expressions.AffineExpression element:
            Expression in the simplex.
        """
        from ..expressions import AffineExpression, Simplex

        assert isinstance(simplex, Simplex)
        assert isinstance(element, AffineExpression)

        self.simplex = simplex
        self.element = element

        super(SimplexConstraint, self).__init__(simplex._typeStr)

    Subtype = namedtuple("Subtype", ("argdim", "truncated", "symmetrized"))

    def _subtype(self):
        return self.Subtype(
            len(self.element),
            self.simplex.truncated,
            self.simplex.symmetrized)

    @classmethod
    def _cost(cls, subtype):
        return subtype.argdim

    def _expression_names(self):
        yield "simplex"
        yield "element"

    def _str(self):
        return glyphs.element(self.element.string, self.simplex.string)

    def _get_slack(self):
        # TODO: Compute simplex constraint slack.
        raise NotImplementedError

        # FIXME: This old code can't be right for all cases.
        # return cvxopt.matrix([1 - norm(self.element, 'inf').safe_value,
        #     self.simplex.radius - norm(self.element, 1).safe_value])


# --------------------------------------
__all__ = api_end(_API_START, globals())
