# ------------------------------------------------------------------------------
# Copyright (C) 2012-2017 Guillaume Sagnol
# Copyright (C)      2018 Maximilian Stahlberg
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

"""Implementation of :class:`VectorNormConstraint`."""

from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class VectorNormConstraint(Constraint):
    """Bound on a (generalized) vector :math:`p`-norm."""

    class Conversion(ConstraintConversion):
        """Bound on a (generalized) :math:`p`-norm constraint conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ..expressions import RealVariable as RV
            from . import AffineConstraint as AC, GeometricMeanConstraint as GM

            n, num, den, relation = subtype
            p = float(num) / float(den)

            assert p >= 0.0

            if relation == Constraint.LE:
                assert p >= 1.0

                if p == 1.0:
                    yield ("var", RV.make_var_type(dim=n, bnd=0), 1)
                    yield ("con", AC.make_type(dim=n, eq=False), 2)
                    yield ("con", AC.make_type(dim=1, eq=False), 1)
                elif p == float("inf"):
                    yield ("con", AC.make_type(dim=n, eq=False), 2)
                else:  # 1 < p < inf
                    yield ("var", RV.make_var_type(dim=n, bnd=0), 2)
                    yield ("con", AC.make_type(dim=1, eq=False), 2*n)
                    yield ("con", GM.make_type(argdim=num), n)
                    yield ("con", AC.make_type(dim=1, eq=False), 1)
            else:
                assert relation == Constraint.GE
                assert p <= 1.0

                if p == 1.0:
                    yield ("con", AC.make_type(dim=n, eq=False), 1)
                    yield ("con", AC.make_type(dim=1, eq=False), 1)
                elif p == float("-inf"):
                    yield ("con", AC.make_type(dim=n, eq=False), 1)
                elif p > 0:
                    yield ("var", RV.make_var_type(dim=n, bnd=0), 1)
                    yield ("con", GM.make_type(argdim=den), n)
                    yield ("con", AC.make_type(dim=1, eq=False), 1)
                else:  # -inf < p < 0
                    yield ("var", RV.make_var_type(dim=n, bnd=0), 1)
                    yield ("con", GM.make_type(argdim=(num + den)), n)
                    yield ("con", AC.make_type(dim=1, eq=False), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..expressions import GeometricMean
            from ..modeling import Problem

            norm     = con.norm
            relation = con.relation
            rhs      = con.rhs

            p = norm.p
            m = len(norm.x)

            P = Problem()

            if relation == Constraint.LE:
                if p == 1:
                    v = P.add_variable('__v', m)
                    P.add_constraint(norm.x[:] <= v)
                    P.add_constraint(-norm.x[:] <= v)
                    P.add_constraint((1 | v) <= rhs)
                elif p == float('inf'):
                    P.add_constraint(norm.x <= rhs)
                    P.add_constraint(-norm.x <= rhs)
                else:
                    x = P.add_variable('__x', m)
                    v = P.add_variable('__v', m)
                    amb = norm.pnum - norm.pden
                    b = norm.pden
                    oneamb = '|1|(' + str(amb) + ',1)'
                    oneb = '|1|(' + str(b) + ',1)'
                    for i in range(m):
                        if amb > 0:
                            if b == 1:
                                vec = (v[i]) // (rhs * oneamb)
                            else:
                                vec = (v[i] * oneb) // (rhs * oneamb)
                        else:
                            # TODO: This shouldn't be possible?
                            if b == 1:
                                vec = v[i]
                            else:
                                vec = (v[i] * oneb)
                        P.add_constraint(norm.x[i] <= x[i])
                        P.add_constraint(-norm.x[i] <= x[i])
                        P.add_constraint(x[i] <= GeometricMean(vec))
                    P.add_constraint((1 | v) <= rhs)
            else:
                if p == 1:
                    P.add_constraint(norm.x >= 0)
                    P.add_constraint((1 | norm.x) >= rhs)
                elif p == float('-inf'):
                    P.add_constraint(norm.x >= rhs)
                elif p >= 0:
                    v = P.add_variable('__v', m)
                    bma = -(norm.pnum - norm.pden)
                    a = norm.pnum
                    onebma = '|1|(' + str(bma) + ',1)'
                    onea = '|1|(' + str(a) + ',1)'
                    for i in range(m):
                        if a == 1:
                            vec = (norm.x[i]) // (rhs * onebma)
                        else:
                            vec = (norm.x[i] * onea) // (rhs * onebma)
                        P.add_constraint(v[i] <= GeometricMean(vec))
                    P.add_constraint(rhs <= (1 | v))
                else:
                    v = P.add_variable('__v', m)
                    b = abs(norm.pden)
                    a = abs(norm.pnum)
                    oneb = '|1|(' + str(b) + ',1)'
                    onea = '|1|(' + str(a) + ',1)'
                    for i in range(m):
                        if a == 1 and b == 1:
                            vec = (norm.x[i]) // (v[i])
                        elif a > 1 and b == 1:
                            vec = (norm.x[i] * onea) // (v[i])
                        elif a == 1 and b > 1:
                            vec = (norm.x[i]) // (v[i] * oneb)
                        else:
                            vec = (norm.x[i] * onea) // (v[i] * oneb)
                        P.add_constraint(rhs <= GeometricMean(vec))
                    P.add_constraint((1 | v) <= rhs)

            return P

    def __init__(self, norm, relation, rhs):
        """Construct a :class:`VectorNormConstraint`.

        :param ~picos.expressions.Norm norm:
            Left hand side expression.
        :param str relation:
            Constraint relation symbol.
        :param ~picos.expressions.AffineExpression rhs:
            Right hand side expression.
        """
        from ..expressions import AffineExpression, Norm

        assert isinstance(norm, Norm)
        assert isinstance(rhs, AffineExpression)
        assert relation in self.LE + self.GE
        assert len(rhs) == 1

        assert norm.q == norm.p, \
            "Can't create a p-norm constraint for a (p,q)-norm."

        if relation == self.LE:
            assert norm.convex, \
                "Upper bounding a p-norm requires p s.t. the norm is convex."

        if relation == self.GE:
            assert norm.concave, \
                "Lower bounding a p-norm requires p s.t. the norm is concave."

        self.norm     = norm
        self.relation = relation
        self.rhs      = rhs

        super(VectorNormConstraint, self).__init__(norm._typeStr)

    Subtype = namedtuple("Subtype", ("argdim", "num", "den", "relation"))

    def _subtype(self):
        return self.Subtype(
            len(self.norm.x), self.norm.pnum, self.norm.pden, self.relation)

    @classmethod
    def _cost(cls, subtype):
        return subtype.argdim + 1

    def _expression_names(self):
        yield "norm"
        yield "rhs"

    def _str(self):
        if self.relation == self.LE:
            str = glyphs.le(self.norm.string, self.rhs.string)
        else:
            str = glyphs.ge(self.norm.string, self.rhs.string)

        if self.norm.p < 1:
            return glyphs.and_(str, glyphs.ge(self.norm.x.string, 0))
        else:
            return str

    def _get_slack(self):
        if self.relation == self.LE:
            return self.rhs.safe_value - self.norm.safe_value
        else:
            return self.norm.safe_value - self.rhs.safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
