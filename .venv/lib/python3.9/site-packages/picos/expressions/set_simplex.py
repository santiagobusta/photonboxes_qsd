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

"""Implements :class:`Simplex`."""

import operator
from collections import namedtuple

from .. import glyphs
from ..apidoc import api_end, api_start
from ..constraints import SimplexConstraint
from .data import convert_and_refine_arguments
from .exp_affine import AffineExpression, Constant
from .set import Set

_API_START = api_start(globals())
# -------------------------------


class Simplex(Set):
    r"""A (truncated, symmetrized) real simplex.

    :Definition:

    Let :math:`r \in \mathbb{R}_{\geq 0}` the specified radius and
    :math:`n \in \mathbb{Z}_{\geq 1}` an arbitrary dimensionality.

    1.  Without truncation and symmetrization, this is the nonnegative simplex

        .. math::

            \{x \in \mathbb{R}^n_{\geq 0} \mid \sum_{i = 1}^n x_i \leq r\}.

        For :math:`r = 1`, this is the standard (unit) :math:`n`-simplex.

    2.  With truncation but without symmetrization, this is the nonnegative
        simplex intersected with the :math:`\infty`-norm unit ball

        .. math::

            \{
                x \in \mathbb{R}^n_{\geq 0}
            \mid
                \sum_{i = 1}^n x_i \leq r \land x \leq 1
            \}.

        For :math:`r \leq 1`, this equals case (1).

    3.  With symmetrization but without truncation, this is the :math:`1`-norm
        ball of radius :math:`r`

        .. math::

            \{x \in \mathbb{R}^n \mid \sum_{i = 1}^n |x_i| \leq r\}.

    4.  With both symmetrization and truncation, this is the convex polytope

        .. math::

            \{
                x \in \mathbb{R}
            \mid
                \sum_{i = 1}^n |x_i| \leq r \land 0 \leq x \leq 1
            \}.

        For :math:`r \leq 1`, this equals case (3).
    """

    @convert_and_refine_arguments("radius")
    def __init__(self, radius=Constant(1), truncated=False, symmetrized=False):
        """Construct a :class:`Simplex`.

        :param radius: The radius of the simplex.
        :type radius:
            float or ~picos.expressions.AffineExpression
        """
        if not isinstance(radius, AffineExpression):
            raise TypeError("A simplex' radius must be given as a real affine "
                "expression, not as {}.".format(type(radius).__name__))
        elif not radius.scalar:
            raise TypeError("A simplex' radius must be scalar, not of shape {}."
                .format(glyphs.shape(radius.shape)))

        if radius.constant and radius.value <= 1:
            truncated = False

        var  = glyphs.free_var_name(radius.string)
        unit = "Unit " if radius.is1 else ""
        if not truncated and not symmetrized:
            typeStr = "{}Simplex".format(unit)
            symbStr = glyphs.set(glyphs.sep(glyphs.ge(var, 0),
                glyphs.le(glyphs.sum(var), radius.string)))
        elif truncated and not symmetrized:
            typeStr = "Box-Truncated {}Simplex".format(unit)
            symbStr = glyphs.set(glyphs.sep(glyphs.le(0, glyphs.le(var, 1)),
                glyphs.le(glyphs.sum(var), radius.string)))
        elif not truncated and symmetrized:
            typeStr = "{}1-norm Ball".format(unit)
            symbStr = glyphs.set(glyphs.sep(var,
                glyphs.le(glyphs.sum(glyphs.abs(var)), radius.string)))
        else:  # truncated and symmetrized
            typeStr = "Box-Truncated {}1-norm Ball".format(unit)
            symbStr = glyphs.set(glyphs.sep(glyphs.le(-1, glyphs.le(var, 1)),
                glyphs.le(glyphs.sum(glyphs.abs(var)), radius.string)))

        self._radius      = radius
        self._truncated   = truncated
        self._symmetrized = symmetrized

        Set.__init__(self, typeStr, symbStr)

    @property
    def radius(self):
        """The radius of the simplex."""
        return self._radius

    @property
    def truncated(self):
        r"""Whether this is intersected with the unit :math:`\infty`-ball."""
        return self._truncated

    @property
    def symmetrized(self):
        """Wether the simplex is mirrored onto all orthants."""
        return self._symmetrized

    def _get_mutables(self):
        return self._radius._get_mutables()

    def _replace_mutables(self, mapping):
        return self.__class__(self._radius._replace_mutables(mapping),
            self._truncated, self._symmetrized)

    Subtype = namedtuple("Subtype", ("truncated", "symmetrized"))

    def _get_subtype(self):
        return self.Subtype(self._truncated, self._symmetrized)

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        if relation == operator.__rshift__:
            if issubclass(other.clstype, AffineExpression):
                return SimplexConstraint.make_type(
                    argdim=other.subtype.dim,
                    truncated=subtype.truncated,
                    symmetrized=subtype.symmetrized)

        return NotImplemented

    def _rshift_implementation(self, element):
        if isinstance(element, AffineExpression):
            return SimplexConstraint(self, element)
        else:
            return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
