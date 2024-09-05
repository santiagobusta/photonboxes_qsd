# ------------------------------------------------------------------------------
# Copyright (C) 2012-2019 Guillaume Sagnol
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

"""Implementation of :class:`PowerTraceConstraint`."""

import math
from collections import namedtuple

import cvxopt as cvx

from .. import glyphs
from ..apidoc import api_end, api_start
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class PowerTraceConstraint(Constraint):
    """Bound on the trace over the :math:`p`-th power of a matrix.

    For scalar expressions, this is simply a bound on their :math:`p`-th power.
    """

    class Conversion(ConstraintConversion):
        """Bound on the :math:`p`-th power of a trace constraint conversion.

        The conversion is based on
        `this paper <http://nbn-resolving.de/urn:nbn:de:0297-zib-17511>`_.
        """

        @classmethod
        def _count_number_tree_node_types(cls, x):
            """Count number of conversion tree nodes.

            Consider a binary tree with x[i] leaves of type i, arranged from
            left to right, with sum(x) a power of two. A node of the tree is of
            type i if its 2 parents are of type i; otherwise, a new type is
            created for this node. This function counts the number of additional
            types we need to create while growing the tree.
            """
            x = [xi for xi in x if xi != 0]
            sum_x = sum(x)

            # We have reached the tree root. Stop the recursion.
            if sum_x == 1:
                return 0

            # Make sure x is a power of two.
            _log2_sum_x = math.log(sum_x, 2)
            assert _log2_sum_x == int(_log2_sum_x)

            new_x  = []
            new_t  = 0
            s      = 0
            offset = 0

            # Compute the vector new_x of types at next level.
            for x_i in x:
                s += x_i

                if s % 2 == 0:
                    if x_i - offset >= 2:
                        new_x.append((x_i - offset) // 2)

                    offset = 0
                else:
                    if   x_i - offset >= 2:
                        new_x.extend([(x_i - offset) // 2, 1])
                    elif x_i - offset == 1:
                        new_x.append(1)
                    elif x_i - offset == 0:
                        assert False, "Unexpected case."

                    offset = 1
                    new_t += 1

            assert 2*sum(new_x) == sum_x

            return new_t + cls._count_number_tree_node_types(new_x)

        @staticmethod
        def _np2(n):
            """Compute the smallest power of two that is an upper bound."""
            return 2**int(math.ceil(math.log(n, 2)))

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from ..expressions import (HermitianVariable, RealVariable,
                                       SymmetricVariable)
            from . import (AffineConstraint, ComplexLMIConstraint,
                           RSOCConstraint, LMIConstraint)

            n, num, den, hasM, complex = subtype

            if num > den > 0:
                x = [den, cls._np2(num) - num, num - den]
            elif num / den < 0:
                num = abs(num)
                den = abs(den)
                x = [den, num, cls._np2(num + den) - num - den]
            elif 0 < num < den:
                x = [num, cls._np2(den) - den, den - num]
            else:
                assert False, "Unexpected exponent."

            N = cls._count_number_tree_node_types(x)

            if n == 1:
                yield ("var", RealVariable.make_var_type(dim=1, bnd=0), N - 1)
                yield ("con", RSOCConstraint.make_type(argdim=1), N)
                if hasM:
                    yield ("var", RealVariable.make_var_type(dim=1, bnd=0), 1)
                    yield ("con",
                        AffineConstraint.make_type(dim=1, eq=False), 1)
            else:
                if complex:
                    yield ("var",
                        HermitianVariable.make_var_type(dim=n**2, bnd=0), N)
                    yield ("con",
                        ComplexLMIConstraint.make_type(diag=2*n), N)
                else:
                    yield ("var", SymmetricVariable.make_var_type(
                        dim=(n * (n + 1)) // 2, bnd=0), N)
                    yield ("con", LMIConstraint.make_type(diag=2*n), N)
                yield ("con", AffineConstraint.make_type(dim=1, eq=False), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..expressions import Constant
            from ..expressions.algebra import block, rsoc
            from ..modeling import Problem

            x   = con.power.x
            n   = con.power.n
            num = con.power.num
            den = con.power.den
            rhs = con.rhs
            m   = con.power.m

            vtype = "hermitian" if x.complex else "symmetric"

            P = Problem()

            if n == 1:
                idt = Constant('1', 1)
                if m is None:
                    varcnt = 0
                    v = []
                else:
                    varcnt = 1
                    v = [P.add_variable('__v[0]', 1)]
            else:
                idt = Constant('I', cvx.spdiag([1.] * n))
                varcnt = 1
                v = [P.add_variable('__v[0]', (n, n), vtype)]

            if con.relation == Constraint.LE and num > den:
                pown = cls._np2(num)

                if n == 1:
                    lis = [rhs]*den  + [x]*(pown - num) + [idt]*(num - den)
                else:
                    lis = [v[0]]*den + [x]*(pown - num) + [idt]*(num - den)

                while len(lis) > 2:
                    newlis = []
                    while lis:
                        v1 = lis.pop()
                        v2 = lis.pop()

                        if v1 is v2:
                            newlis.append(v2)
                        else:
                            if n == 1:
                                v0 = P.add_variable(
                                    '__v[' + str(varcnt) + ']', 1)
                                P.add_constraint((v1 & v2 & v0) << rsoc())
                            else:
                                v0 = P.add_variable(
                                    '__v[' + str(varcnt) + ']', (n, n), vtype)
                                P.add_constraint(
                                    block([[v1, v0], [v0, v2]]) >> 0)

                            varcnt += 1
                            newlis.append(v0)
                            v.append(v0)
                    lis = newlis

                if n == 1:
                    P.add_constraint((lis[0] & lis[1] & x) << rsoc())
                else:
                    P.add_constraint(block([[lis[0], x], [x, lis[1]]]) >> 0)
                    P.add_constraint((idt | v[0]) <= rhs)
            elif con.relation == Constraint.LE and num <= den:
                num = abs(num)
                den = abs(den)

                pown = cls._np2(num + den)

                if n == 1:
                    lis = [rhs] * den + [x] * num + [idt] * (pown - num - den)
                else:
                    lis = [v[0]] * den + [x] * num + [idt] * (pown - num - den)

                while len(lis) > 2:
                    newlis = []
                    while lis:
                        v1 = lis.pop()
                        v2 = lis.pop()

                        if v1 is v2:
                            newlis.append(v2)
                        else:
                            if n == 1:
                                v0 = P.add_variable(
                                    '__v[' + str(varcnt) + ']', 1)
                                P.add_constraint((v1 & v2 & v0) << rsoc())
                            else:
                                v0 = P.add_variable(
                                    '__v[' + str(varcnt) + ']', (n, n), vtype)
                                P.add_constraint(
                                    block([[v1, v0], [v0, v2]]) >> 0)

                            varcnt += 1
                            newlis.append(v0)
                            v.append(v0)
                    lis = newlis

                if n == 1:
                    P.add_constraint((lis[0] & lis[1] & 1) << rsoc())
                else:
                    P.add_constraint(block([[lis[0], idt], [idt, lis[1]]]) >> 0)
                    P.add_constraint((idt | v[0]) <= rhs)
            elif con.relation == Constraint.GE:
                pown = cls._np2(den)

                if n == 1:
                    lis = [x]*num + [rhs]*(pown - den)  + [idt]*(den - num)

                else:
                    lis = [x]*num + [v[0]]*(pown - den) + [idt]*(den - num)

                while len(lis) > 2:
                    newlis = []
                    while lis:
                        v1 = lis.pop()
                        v2 = lis.pop()

                        if v1 is v2:
                            newlis.append(v2)
                        else:
                            if n == 1:
                                v0 = P.add_variable(
                                    '__v[' + str(varcnt) + ']', 1)
                                P.add_constraint((v1 & v2 & v0) << rsoc())
                            else:
                                v0 = P.add_variable(
                                    '__v[' + str(varcnt) + ']', (n, n), vtype)
                                P.add_constraint(
                                    block([[v1, v0], [v0, v2]]) >> 0)

                            varcnt += 1
                            newlis.append(v0)
                            v.append(v0)
                    lis = newlis

                if n == 1:
                    if m is None:
                        P.add_constraint((lis[0] & lis[1] & rhs) << rsoc())
                    else:
                        P.add_constraint((lis[0] & lis[1] & v[0]) << rsoc())
                        P.add_constraint((m * v[0]) > rhs)
                else:
                    P.add_constraint(
                        block([[lis[0], v[0]], [v[0], lis[1]]]) >> 0)
                    if m is None:
                        P.add_constraint((idt | v[0]) > rhs)
                    else:
                        P.add_constraint((m | v[0]) > rhs)
            else:
                assert False, "Dijkstra-IF fallthrough."

            return P

    def __init__(self, power, relation, rhs):
        """Construct a :class:`PowerTraceConstraint`.

        :param ~picos.expressions.PowerTrace ower:
            Left hand side expression.
        :param str relation:
            Constraint relation symbol.
        :param ~picos.expressions.AffineExpression rhs:
            Right hand side expression.
        """
        from ..expressions import AffineExpression, PowerTrace

        assert isinstance(power, PowerTrace)
        assert relation in self.LE + self.GE
        assert isinstance(rhs, AffineExpression)
        assert len(rhs) == 1

        p = power.p

        assert p != 0 and p != 1, \
            "The PowerTraceConstraint should not be created for p = 0 and " \
            "p = 1 as there are more direct ways to represent such powers."

        if relation == self.LE:
            assert p <= 0 or p >= 1, \
                "Upper bounding p-th power needs p s.t. the power is convex."
        else:
            assert p >= 0 and p <= 1, \
                "Lower bounding p-th power needs p s.t. the power is concave."

        self.power    = power
        self.relation = relation
        self.rhs      = rhs

        super(PowerTraceConstraint, self).__init__(power._typeStr)

    # HACK: Support Constraint's LHS/RHS interface.
    # TODO: Add a unified interface for such constraints?
    lhs = property(lambda self: self.power)

    def is_trace(self):
        """Whether the bound concerns a trace as opposed to a scalar."""
        return self.power.n > 1

    Subtype = namedtuple("Subtype", ("diag", "num", "den", "hasM", "complex"))

    def _subtype(self):
        return self.Subtype(*self.power.subtype)

    @classmethod
    def _cost(cls, subtype):
        n = subtype.diag
        if subtype.complex:
            return n**2 + 1
        else:
            return n*(n + 1)//2 + 1

    def _expression_names(self):
        yield "power"
        yield "rhs"

    def _str(self):
        if self.relation == self.LE:
            return glyphs.le(self.power.string, self.rhs.string)
        else:
            return glyphs.ge(self.power.string, self.rhs.string)

    def _get_slack(self):
        if self.relation == self.LE:
            return self.rhs.safe_value - self.power.safe_value
        else:
            return self.power.safe_value - self.rhs.safe_value


# --------------------------------------
__all__ = api_end(_API_START, globals())
