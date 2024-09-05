# ------------------------------------------------------------------------------
# Copyright (C) 2019 Maximilian Stahlberg
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

"""Quadratic constraint types."""

from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from .constraint import Constraint, ConstraintConversion

_API_START = api_start(globals())
# -------------------------------


class NonconvexQuadraticConstraint(Constraint):
    """Bound on a nonconvex quadratic expression."""

    def __init__(self, lhs, relation, rhs):
        """Construct a :class:`NonconvexQuadraticConstraint`.

        :param lhs: Left hand side quadratic or affine expression.
        :type lhs: ~picos.expressions.QuadraticExpression or
            ~picos.expressions.AffineExpression
        :param str relation: Constraint relation symbol.
        :param rhs: Right hand side quadratic or affine expression.
        :type rhs: ~picos.expressions.QuadraticExpression or
            ~picos.expressions.AffineExpression
        """
        from ..expressions import AffineExpression, QuadraticExpression

        assert isinstance(lhs, (AffineExpression, QuadraticExpression))
        assert isinstance(rhs, (AffineExpression, QuadraticExpression))
        assert any(isinstance(exp, QuadraticExpression) for exp in (lhs, rhs))
        assert relation in self.LE + self.GE
        assert lhs.size == rhs.size

        self.lhs      = lhs
        self.rhs      = rhs
        self.relation = relation

        Constraint.__init__(self, self._get_type_term())

    def _get_type_term(self):
        return "Nonconvex Quadratic"

    # TODO: Add common interface for all the le0, ge0, smaller, greater methods?
    @cached_property
    def le0(self):
        """Quadratic expression constrained to be at most zero."""
        if self.relation == self.GE:
            return self.rhs - self.lhs
        else:
            return self.lhs - self.rhs

    @property
    def smaller(self):
        """Smaller-or-equal side of the constraint."""
        return self.rhs if self.relation == self.GE else self.lhs

    @property
    def greater(self):
        """Greater-or-equal side of the constraint."""
        return self.lhs if self.relation == self.GE else self.rhs

    Subtype = namedtuple("Subtype", ("qterms"))

    def _subtype(self):
        return self.Subtype(self.le0.num_quad_terms)

    @classmethod
    def _cost(cls, subtype):
        return subtype.qterms

    def _expression_names(self):
        yield "lhs"
        yield "rhs"

    def _str(self):
        if self.relation == self.LE:
            return glyphs.le(self.lhs.string, self.rhs.string)
        else:
            return glyphs.ge(self.lhs.string, self.rhs.string)

    def _get_size(self):
        return (1, 1)

    def _get_slack(self):
        if self.relation == self.LE:
            return self.rhs.safe_value - self.lhs.safe_value
        else:
            return self.lhs.safe_value - self.rhs.safe_value


class ConicQuadraticConstraint(NonconvexQuadraticConstraint):
    r"""Bound on a *nearly* convex quadratic expression.

    Nearly convex means that the bilinear form representing the quadratic part
    of the expression that the constraint poses to be at most zero has exactly
    one negative eigenvalue. More precisely, if the constraint is of the form
    :math:`x^TQx + a^Tx + b \leq x^TRx + b^Tx + c`, then `Q - R` needs to have
    exactly one negative eigenvalue for the constraint to be nearly convex. Such
    constraints can be posed as conic constraints under an additional assumption
    that some affine term is nonnegative.

    At this point, this class may only be used for nonconvex constraints of the
    form :math:`x^TQx + p^Tx + q \leq (a^Tx + b)(c^Tx + d)` with
    :math:`x^TQx + p^Tx + q` representable as a squared norm. In this case, the
    additional assumptions required for a conic reformulation are
    :math:`a^Tx + b \geq 0` and :math:`b^Tx + c \geq 0`.

    Whether a constraint of this type is strengthened to a conic constraint or
    relabeled as a :class:`NonconvexQuadraticConstraint` depends on the
    :ref:`assume_conic <option_assume_conic>` option.

    :Example:

    >>> from picos import Options, RealVariable
    >>> x, y, z = RealVariable("x"), RealVariable("y"), RealVariable("z")
    >>> C = x**2 + 1 <= y*z; C
    <Conic Quadratic Constraint: x² + 1 ≤ y·z>
    >>> P = C.__class__.Conversion.convert(C, Options(assume_conic=True))
    >>> list(P.constraints.values())[0]
    <4×1 RSOC Constraint: ‖fullroot(x² + 1)‖² ≤ y·z ∧ y, z ≥ 0>
    >>> Q = C.__class__.Conversion.convert(C, Options(assume_conic=False))
    >>> list(Q.constraints.values())[0]
    <Nonconvex Quadratic Constraint: x² + 1 ≤ y·z>

    .. note::

        Solver implementations must not support this constraint type so that
        the user's choice for the :ref:`assume_conic <option_assume_conic>`
        option is respected.
    """

    class Conversion(ConstraintConversion):
        """Nearly convex quadratic to (rotated) second order cone conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from . import AbsoluteValueConstraint, SOCConstraint, RSOCConstraint

            if options.assume_conic:
                n = subtype.conic_argdim

                if subtype.rotated:
                    yield ("con", RSOCConstraint.make_type(argdim=n), 1)
                elif n == 1:
                    yield ("con", AbsoluteValueConstraint.make_type(), 1)
                else:
                    yield ("con", SOCConstraint.make_type(argdim=n), 1)
            else:
                yield ("con", NonconvexQuadraticConstraint.make_type(
                    qterms=subtype.qterms), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..expressions import AffineExpression
            from ..modeling import Problem
            from . import AbsoluteValueConstraint, SOCConstraint, RSOCConstraint

            P = Problem()

            if options.assume_conic:
                q, r = con.smaller, con.greater
                root = AffineExpression.zero() if q.is0 else q.fullroot

                if r.scalar_factors[0] is not r.scalar_factors[1]:
                    P.add_constraint(RSOCConstraint(root, *r.scalar_factors))
                else:
                    bound = r.scalar_factors[0]

                    if len(root) == 1:
                        P.add_constraint(AbsoluteValueConstraint(root, bound))
                    else:
                        P.add_constraint(SOCConstraint(root, bound))
            else:
                P.add_constraint(NonconvexQuadraticConstraint(
                    con.lhs, con.relation, con.rhs))

            return P

    def __init__(self, lhs, relation, rhs):
        """Construct a :class:`ConicQuadraticConstraint`.

        See :meth:`NonconvexQuadraticConstraint.__init__` for more.
        """
        from ..expressions import QuadraticExpression

        NonconvexQuadraticConstraint.__init__(self, lhs, relation, rhs)

        # Additional usage restrictions for the near-convex case.
        assert isinstance(self.lhs, QuadraticExpression)
        assert isinstance(self.rhs, QuadraticExpression)
        assert self.smaller.is_squared_norm
        assert self.greater.scalar_factors

    def _get_type_term(self):
        return "Conic Quadratic"

    Subtype = namedtuple("Subtype", ("qterms", "conic_argdim", "rotated"))

    def _subtype(self):
        qterms = self.le0.num_quad_terms
        conic_argdim = 1 if self.smaller.is0 else len(self.smaller.fullroot)
        sf = self.greater.scalar_factors
        rotated = sf[0] is not sf[1]

        return self.Subtype(qterms, conic_argdim, rotated)


class ConvexQuadraticConstraint(NonconvexQuadraticConstraint):
    """Bound on a convex quadratic expression."""

    class ConicConversion(ConstraintConversion):
        """Convex quadratic to (rotated) second order cone conversion."""

        @classmethod
        def predict(cls, subtype, options):
            """Implement :meth:`~.constraint.ConstraintConversion.predict`."""
            from . import AbsoluteValueConstraint, SOCConstraint, RSOCConstraint

            if subtype.haslin:
                yield ("con", RSOCConstraint.make_type(argdim=subtype.rank), 1)
            elif subtype.rank == 1:
                yield ("con", AbsoluteValueConstraint.make_type(), 1)
            else:
                yield ("con", SOCConstraint.make_type(argdim=subtype.rank), 1)

        @classmethod
        def convert(cls, con, options):
            """Implement :meth:`~.constraint.ConstraintConversion.convert`."""
            from ..expressions import AffineExpression
            from ..modeling import Problem
            from . import AbsoluteValueConstraint, SOCConstraint, RSOCConstraint

            le0 = con.le0

            P = Problem()

            if con.subtype.haslin:
                one = AffineExpression.from_constant(1)

                P.add_constraint(RSOCConstraint(le0.quadroot, -le0._aff, one))
            else:
                assert le0._aff.constant
                value = -le0._aff.value

                if value < 0:
                    raise ValueError("The constraint {} is infeasible as it "
                        "upper-bounds a positive semidefinite quadratic form by"
                        " a negative constant.".format(con))

                root  = le0.quadroot
                bound = AffineExpression.from_constant(value**0.5,
                    name=glyphs.sqrt(glyphs.scalar(value)))

                if len(root) == 1:
                    P.add_constraint(AbsoluteValueConstraint(root, bound))
                else:
                    P.add_constraint(SOCConstraint(root, bound))

            return P

    def __init__(self, lhs, relation, rhs):
        """Construct a :class:`ConvexQuadraticConstraint`.

        See :meth:`NonconvexQuadraticConstraint.__init__` for more.
        """
        NonconvexQuadraticConstraint.__init__(self, lhs, relation, rhs)

    def _get_type_term(self):
        return "Convex Quadratic"

    Subtype = namedtuple("Subtype", ("qterms", "rank", "haslin"))

    def _subtype(self):
        from ..expressions import QuadraticExpression

        # HACK: Be consistent with the prediction, which cannot foresee whether
        #       the linear terms of both sides cancel out.
        lhs_aff_is_cst = self.lhs._aff.constant \
            if isinstance(self.lhs, QuadraticExpression) else self.lhs.constant
        rhs_aff_is_cst = self.rhs._aff.constant \
            if isinstance(self.rhs, QuadraticExpression) else self.rhs.constant
        haslin = not lhs_aff_is_cst or not rhs_aff_is_cst

        return self.Subtype(self.le0.num_quad_terms, self.le0.rank, haslin)


# --------------------------------------
__all__ = api_end(_API_START, globals())
