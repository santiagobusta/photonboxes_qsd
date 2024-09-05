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

"""Implements :class:`UncertainSquaredNorm`."""

import operator
from collections import namedtuple

import cvxopt
import numpy

from ... import glyphs
from ...apidoc import api_end, api_start
from ...caching import cached_unary_operator
from ...constraints.uncertain import (MomentAmbiguousSquaredNormConstraint,
                                      ScenarioUncertainConicConstraint,
                                      WassersteinAmbiguousSquaredNormConstraint)
from ..cone_rsoc import RotatedSecondOrderCone
from ..data import convert_operands, cvx2np
from ..exp_affine import AffineExpression
from ..exp_biaffine import BiaffineExpression
from ..exp_norm import Norm
from ..expression import Expression, refine_operands, validate_prediction
from .pert_moment import MomentAmbiguitySet
from .pert_scenario import ScenarioPerturbationSet
from .pert_wasserstein import WassersteinAmbiguitySet
from .uexp_affine import UncertainAffineExpression
from .uexpression import UncertainExpression

_API_START = api_start(globals())
# -------------------------------


class UncertainSquaredNorm(UncertainExpression, Expression):
    """Squared Euclidean or Frobenius norm of an uncertain affine expression."""

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    def __init__(self, x):
        """Construct an :class:`UncertainSquaredNorm`.

        :param x:
            The uncertain affine expression to denote the squared norm of.
        :type x:
            ~picos.expressions.uncertain.uexp_affine.UncertainAffineExpression
        """
        if not isinstance(x, UncertainAffineExpression):
            raise TypeError("Can only form the uncertain squared norm of an "
                "uncertain affine expression, not of {}."
                .format(type(x).__name__))

        typeStr = "Uncertain Squared Norm"
        symbStr = glyphs.squared(glyphs.norm(x.string))

        Expression.__init__(self, typeStr, symbStr)

        self._x = x

    # --------------------------------------------------------------------------
    # Properties.
    # --------------------------------------------------------------------------

    @property
    def x(self):
        """Uncertain affine expression under the squared norm."""
        return self._x

    # --------------------------------------------------------------------------
    # Abstract method implementations for Expression, except _predict.
    # --------------------------------------------------------------------------

    @cached_unary_operator
    def _get_refined(self):
        """Implement :meth:`~.expression.Expression._get_refined`."""
        if self.certain:
            return Norm(self._x.refined)**2
        else:
            return self

    Subtype = namedtuple("Subtype", ("argdim", "universe_type"))

    def _get_subtype(self):
        """Implement :meth:`~.expression.Expression._get_subtype`."""
        return self.Subtype(len(self._x), self.universe.type)

    def _get_value(self):
        value = self._x._get_value()

        if len(value) == 1:
            return abs(value)**2
        else:
            return cvxopt.matrix(
                numpy.linalg.norm(numpy.ravel(cvx2np(value)))**2)

    @cached_unary_operator
    def _get_mutables(self):
        return self._x.mutables

    def _is_convex(self):
        return True

    def _is_concave(self):
        return False

    def _replace_mutables(self, mapping):
        return self.__class__(self._x._replace_mutables(mapping))

    def _freeze_mutables(self, freeze):
        return self.__class__(self._x._freeze_mutables(freeze))

    # --------------------------------------------------------------------------
    # Constraint-creating operators and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        AE = AffineExpression
        BAE = BiaffineExpression
        UAE = UncertainAffineExpression
        MAS = MomentAmbiguitySet
        SPS = ScenarioPerturbationSet
        WAS = WassersteinAmbiguitySet

        if issubclass(other.clstype, BAE) and other.subtype.dim != 1:
            return NotImplemented

        if relation is not operator.__le__:
            return NotImplemented

        if issubclass(subtype.universe_type.clstype, MAS):
            if issubclass(other.clstype, AE):
                return MomentAmbiguousSquaredNormConstraint.make_type(
                    sqnorm_argdim=subtype.argdim,
                    universe_subtype=subtype.universe_type.subtype)
        elif issubclass(subtype.universe_type.clstype, WAS):
            if subtype.universe_type.subtype.p != 2:
                return NotImplemented

            if issubclass(other.clstype, AE):
                return WassersteinAmbiguousSquaredNormConstraint.make_type(
                    sqnorm_argdim=subtype.argdim,
                    universe_subtype=subtype.universe_type.subtype)
        elif issubclass(subtype.universe_type.clstype, SPS):
            if issubclass(other.clstype, (AE, UAE)):
                if issubclass(other.clstype, UAE) \
                and not issubclass(other.subtype.universe_type.clstype, SPS):
                    return NotImplemented

                return ScenarioUncertainConicConstraint.make_type(
                    dim=(subtype.argdim + 2),
                    scenario_count=subtype.universe_type.subtype.scenario_count,
                    cone_type=RotatedSecondOrderCone.make_type(dim=None))
        else:
            return NotImplemented

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        if isinstance(self._x.universe, MomentAmbiguitySet):
            if isinstance(other, AffineExpression):
                return MomentAmbiguousSquaredNormConstraint(self, other)
            elif isinstance(other, UncertainAffineExpression):
                # Raise a meaningful exception because there are other cases
                # where upper bounding with an UncertainAffineExpression works
                # so the default Python exception would be misleading.
                raise TypeError("When upper-bounding a moment-ambiguous "
                    "expected squared norm, the upper bound must be certain.")
        elif isinstance(self._x.universe, WassersteinAmbiguitySet):
            if self._x.universe.p != 2:
                raise ValueError("Upper-bounding an expected squared norm under"
                    " Wasserstein ambiguity requires p = 2.")

            if isinstance(other, AffineExpression):
                return WassersteinAmbiguousSquaredNormConstraint(self, other)
            elif isinstance(other, UncertainAffineExpression):
                # Raise a meaningful exception because there are other cases
                # where upper bounding with an UncertainAffineExpression works
                # so the default Python exception would be misleading.
                raise TypeError("When upper-bounding a Wasserstein-ambiguous "
                    "expected squared norm, the upper bound must be certain.")
        elif isinstance(self._x.universe, ScenarioPerturbationSet):
            if isinstance(other, (AffineExpression, UncertainAffineExpression)):
                # Uncertain upper bound must have equal uncertainty.
                # NOTE: Can only be predicted up to the perturbation type.
                if isinstance(other, UncertainAffineExpression) \
                and self.perturbation is not other.perturbation:
                    raise ValueError("If the upper bound to a scenario "
                        "uncertain squared norm is itself uncertain, then the "
                        "uncertainty in both sides must be equal (same "
                        "perturbation parameter).")

                return (other // 1 // self._x.vec) << RotatedSecondOrderCone()
        else:
            raise TypeError("Upper-bounding an uncertain squared norm whose "
                "perturbation parameter is described by an instance of {} is "
                "not supported.".format(self._x.universe.__class__.__name__))

        # Make sure the Python NotImplemented-triggered TypeError works.
        assert not isinstance(other,
            (AffineExpression, UncertainAffineExpression))

        return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
