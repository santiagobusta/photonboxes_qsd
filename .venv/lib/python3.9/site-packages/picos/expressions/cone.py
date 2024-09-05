# ------------------------------------------------------------------------------
# Copyright (C) 2020 Maximilian Stahlberg
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

"""Backend for mathematical set type implementations."""

import operator
from abc import abstractmethod

from .. import glyphs
from ..apidoc import api_end, api_start
from ..constraints.uncertain import ScenarioUncertainConicConstraint
from .expression import ExpressionType
from .set import Set

_API_START = api_start(globals())
# -------------------------------


class Cone(Set):
    """Abstract base class for a cone."""

    def __init__(self, dim, typeStr, symbStr):
        """Perform basic initialization for :class:`Cone` instances.

        :param int or None dim: Fixed member dimensionality, or :obj:`None`.
        :param str typeStr: Short string denoting the set type.
        :param str symbStr: Algebraic string description of the set.
        """
        if dim:
            typeStr = "{}-dim. {}".format(dim, typeStr)

        Set.__init__(self, typeStr, symbStr)

        if dim is not None and not isinstance(dim, int):
            raise TypeError(
                "The cone's fixed dimensionality must be an integer or None.")

        if dim == 0:
            raise ValueError("Zero-dimensional cones are not supported.")

        self._dim = dim

    @property
    def dim(self):
        """The fixed member dimensionality, or :obj:`None`.

        If this is :obj:`None`, the instance represents any finite dimensional
        version of the cone. Such an abstract cone can not be used to define a
        :class:`~.cone_product.ProductCone`.
        """
        return self._dim

    @property
    @abstractmethod
    def dual_cone(self):
        """The dual cone."""
        pass

    def _check_dimension(self, element):
        if self.dim and self.dim != len(element):
            raise TypeError("The shape {} of {} does not match the fixed "
                "dimensionality {} of the cone {}.".format(glyphs.shape(
                element.shape), element.string, self.dim, self.string))

    @staticmethod
    def _predict_base(cls, subtype, relation, other):
        """Base :meth:`_predict` method for all cone types."""
        from .uncertain.pert_scenario import ScenarioPerturbationSet
        from .uncertain.uexp_affine import UncertainAffineExpression

        assert isinstance(subtype, cls.Subtype)

        cone = ExpressionType(cls, subtype)

        if relation == operator.__rshift__:
            if issubclass(other.clstype, UncertainAffineExpression) \
            and not subtype.dim or subtype.dim == other.subtype.dim:
                universe = other.subtype.universe_type

                if issubclass(universe.clstype, ScenarioPerturbationSet):
                    return ScenarioUncertainConicConstraint.make_type(
                        dim=other.subtype.dim,
                        scenario_count=universe.subtype.scenario_count,
                        cone_type=cone)

        return NotImplemented

    @staticmethod
    def _rshift_base(self, element):
        """Base :meth:`__rshift__` method for all cone types."""
        from .uncertain.pert_scenario import ScenarioPerturbationSet
        from .uncertain.uexp_affine import UncertainAffineExpression

        if isinstance(element, UncertainAffineExpression):
            self._check_dimension(element)

            if isinstance(element.universe, ScenarioPerturbationSet):
                return ScenarioUncertainConicConstraint(element, self)

        return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
