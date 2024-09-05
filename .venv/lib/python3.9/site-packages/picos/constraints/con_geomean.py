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

"""Implementation of :class:`GeometricMeanConstraint`."""

import math
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class GeometricMeanConstraint(Constraint):
    """Lower bound on a geometric mean."""

    class RSOCConversion(ConstraintConversion):
        """Geometric mean to rotated second order cone constraint conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ..expressions import RealVariable
            from . import RSOCConstraint

            j = subtype.argdim - 1
            k = int(math.log(j, 2))
            l = j + k - "{:b}".format(j - 2**k).count("1")

            yield ("var", RealVariable.make_var_type(dim=1, bnd=0), l - 1)
            yield ("con", RSOCConstraint.make_type(argdim=1), l)

        # TODO: Refactor to be human readable.
        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..expressions.algebra import rsoc
            from ..modeling import Problem

            geoMean    = con.geoMean
            lowerBound = con.lowerBound

            P     = Problem()
            x     = geoMean.x
            m     = len(x)
            lm    = [[i] for i in range(m - 1, -1, -1)]
            K     = []
            depth = 0
            u     = {}

            while len(lm) > 1:
                depth += 1
                nlm = []
                while lm:
                    i1 = lm.pop()[-1]
                    if lm:
                        i2 = lm.pop()[0]
                    else:
                        i2 = 'x'
                    nlm.insert(0, (i2, i1))
                    k = str(depth) + ':' + str(i1) + '-' + str(i2)
                    K.append(k)
                    u[k] = P.add_variable('__u[' + k + ']', 1)
                lm = nlm

            root = K[-1]
            maxd = int(K[-1].split(':')[0])
            P.remove_variable(u[root].name)  # TODO: Update for new expressions.
            u[root] = lowerBound

            for k in K:
                i1 = int(k.split('-')[0].split(':')[1])
                i2 = k.split('-')[1]
                if i2 != 'x':
                    i2 = int(i2)
                if k[:2] == '1:':
                    if i2 != 'x':
                        P.add_constraint((x[i1] & x[i2] & u[k]) << rsoc())
                    else:
                        P.add_constraint((x[i1] & lowerBound & u[k]) << rsoc())
                else:
                    d = int(k.split(':')[0])
                    if i2 == 'x' and d < maxd:
                        k2pot = [ki for ki in K
                            if ki.startswith(str(d - 1) + ':')
                            and int(ki.split(':')[1].split('-')[0]) >= i1]
                        k1 = k2pot[0]
                        if len(k2pot) == 2:
                            k2 = k2pot[1]
                            P.add_constraint((u[k1] & u[k2] & u[k]) << rsoc())
                        else:
                            P.add_constraint(
                                (u[k1] & lowerBound & u[k]) << rsoc())
                    else:
                        k1 = [ki for ki in K
                            if ki.startswith(str(d - 1) + ':' + str(i1))][0]
                        k2 = [ki for ki in K if ki.startswith(str(d - 1) + ':')
                            and ki.endswith('-' + str(i2))][0]
                        P.add_constraint((u[k1] & u[k2] & u[k]) << rsoc())

            return P

    def __init__(self, geoMean, lowerBound):
        """Construct a :class:`GeometricMeanConstraint`.

        :param ~picos.expressions.GeometricMean lhs:
            Constrained expression.
        :param ~picos.expressions.AffineExpression lowerBound:
            Lower bound on the expression.
        """
        from ..expressions import AffineExpression, GeometricMean

        assert isinstance(geoMean, GeometricMean)
        assert isinstance(lowerBound, AffineExpression)
        assert len(lowerBound) == 1

        self.geoMean    = geoMean
        self.lowerBound = lowerBound

        super(GeometricMeanConstraint, self).__init__(geoMean._typeStr)

    Subtype = namedtuple("Subtype", ("argdim",))

    def _subtype(self):
        return self.Subtype(len(self.geoMean.x))

    @classmethod
    def _cost(cls, subtype):
        return subtype.argdim + 1

    def _expression_names(self):
        yield "geoMean"
        yield "lowerBound"

    def _str(self):
        return glyphs.ge(self.geoMean.string, self.lowerBound.string)

    def _get_slack(self):
        return self.geoMean.safe_value - self.lowerBound.safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
