# ------------------------------------------------------------------------------
# Copyright (C) 2019 Maximilian Stahlberg
# Based on the original picos.expressions module by Guillaume Sagnol.
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

"""Implements :class:`Ball`."""

import operator
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from .data import convert_and_refine_arguments, make_fraction
from .exp_affine import AffineExpression, ComplexAffineExpression, Constant
from .exp_norm import Norm
from .set import Set

_API_START = api_start(globals())
# -------------------------------


class Ball(Set):
    r"""A ball of radius :math:`r` according to a (generalized) :math:`p`-norm.

    :Definition:

    In the following, :math:`\lVert \cdot \rVert_p` refers to the vector
    :math:`p`-norm or to the entrywise matrix :math:`p`-norm, depending on the
    argument. See :class:`~picos.Norm` for definitions.

    Let :math:`r \in \mathbb{R}`.

    1.  For :math:`p \in [1, \infty)` or :math:`p = \infty` (input as
        ``float("inf")``), this is the convex set

        .. math::

            \{x \in \mathbb{K} \mid \lVert x \rVert_p \leq r\}

        for any

        .. math::

            \mathbb{K} \in \bigcup_{m, n \in \mathbb{Z}_{\geq 1}}
                \left( \mathbb{C}^n \cup \mathbb{C}^{m \times n} \right).

    2.  For a generalized :math:`p`-norm with :math:`p \in (0, 1)`, this is the
        convex set

        .. math::

            \{x \in \mathbb{K} \mid \lVert x \rVert_p \geq r \land x \geq 0\}

        for any

        .. math::

            \mathbb{K} \in \bigcup_{m, n \in \mathbb{Z}_{\geq 1}}
                \left( \mathbb{R}^n \cup \mathbb{R}^{m \times n} \right).

    Note that :math:`x` may not be complex if :math:`p < 1` due to the implicit
    :math:`x \geq 0` constraint in this case, which is not meaningful on the
    complex field.

    Note further that :math:`r` may be any scalar affine expression, it does not
    need to be constant.

    .. note::

        Due to significant differences in scope, :class:`Ball` is not a
        subclass of :class:`~.set_ellipsoid.Ellipsoid` even though both
        classes can represent Euclidean balls around the origin.
    """

    @convert_and_refine_arguments("radius")
    def __init__(self, radius=Constant(1), p=2, denominator_limit=1000):
        """Construct a :math:`p`-norm ball of given radius.

        :param radius: The ball's radius.
        :type radius:
            float or ~picos.expressions.AffineExpression
        :param float p: The value for :math:`p`, which is cast to a limited
            precision fraction.
        :param int denominator_limit: The largest allowed denominator when
            casting :math:`p` to a fraction. Higher values can yield a greater
            precision at reduced performance.
        """
        num, den, p, pStr = make_fraction(p, denominator_limit)

        if not isinstance(radius, AffineExpression):
            raise TypeError("The ball's radius must be given as a real affine "
                "expression, not as {}.".format(type(radius).__name__))
        elif not radius.scalar:
            raise TypeError("The ball's radius must be scalar, not of shape {}."
                .format(glyphs.shape(radius.shape)))

        var  = glyphs.free_var_name(radius.string)
        unit = "Unit " if radius.is1 else ""
        if p >= 1:
            typeStr = "{}{}-norm Ball" \
                .format(unit, pStr if den == 1 else glyphs.parenth(pStr))
            symbStr = glyphs.set(glyphs.sep(
                var, glyphs.le(glyphs.pnorm(var, pStr), radius.string)))
        else:
            typeStr = "Nonneg. Compl. of {}{}-norm Ball" \
                .format(unit, pStr if den == 1 else glyphs.parenth(pStr))
            symbStr = glyphs.set(glyphs.sep(glyphs.ge(var, 0),
                glyphs.ge(glyphs.pnorm(var, pStr), radius.string)))

        self._num    = num
        self._den    = den
        self._limit  = denominator_limit
        self._radius = radius

        Set.__init__(self, typeStr, symbStr)

    @property
    def p(self):
        """The value :math:`p` defining the :math:`p`-norm used.

        This is a limited precision version of the parameter used when the ball
        was constructed.
        """
        return float(self._num) / float(self._den)

    @property
    def r(self):
        """The ball's radius :math:`r`."""
        return self._radius

    def _get_mutables(self):
        return self._radius._get_mutables()

    def _replace_mutables(self, mapping):
        return self.__class__(
            self.p, self._radius._replace_mutables(mapping), self._limit)

    Subtype = namedtuple("Subtype", ("num", "den"))

    def _get_subtype(self):
        return self.Subtype(self._num, self._den)

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        num = subtype.num
        den = subtype.den
        p   = float(num) / float(den)

        if relation == operator.__rshift__:
            if issubclass(other.clstype, ComplexAffineExpression):
                complex = not issubclass(other.clstype, AffineExpression)

                if complex and p < 1:
                    return NotImplemented

                # The shape of the real, vectorized version of the element.
                shape = (other.subtype.dim * (2 if complex else 1), 1)

                norm = Norm.make_type(shape, num, den, num, den)

                # HACK: Whether the radius is constant makes no difference.
                radius = AffineExpression.make_type((1, 1), None, None)

                if p >= 1:
                    return norm.predict(operator.__le__, radius)
                else:
                    return norm.predict(operator.__ge__, radius)

        return NotImplemented

    def _rshift_implementation(self, element):
        if isinstance(element, ComplexAffineExpression):
            if element.complex and self.p < 1:
                raise TypeError("Cannot constrain a complex expression to be "
                    "in the nonnegative complement of a generalized p-norm "
                    "ball: Nonnegativity is not clear.")

            norm = Norm(element, self.p, denominator_limit=self._limit)

            if self.p >= 1:
                return norm <= self._radius
            else:
                return norm >= self._radius
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
