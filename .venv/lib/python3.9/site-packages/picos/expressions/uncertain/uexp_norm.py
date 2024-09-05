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

"""Implements :class:`UncertainNorm`."""

import operator
from collections import namedtuple

import cvxopt
import numpy

from ... import glyphs
from ...apidoc import api_end, api_start
from ...caching import cached_unary_operator
from ...constraints.uncertain import (BallUncertainNormConstraint,
                                      ScenarioUncertainConicConstraint)
from ..cone_soc import SecondOrderCone
from ..data import convert_operands, cvx2np
from ..exp_affine import AffineExpression
from ..exp_biaffine import BiaffineExpression
from ..exp_norm import Norm
from ..expression import Expression, refine_operands, validate_prediction
from .pert_conic import ConicPerturbationSet, UnitBallPerturbationSet
from .pert_scenario import ScenarioPerturbationSet
from .uexp_affine import UncertainAffineExpression
from .uexp_sqnorm import UncertainSquaredNorm
from .uexpression import UncertainExpression

_API_START = api_start(globals())
# -------------------------------


class UncertainNorm(UncertainExpression, Expression):
    """Euclidean or Frobenius norm of an uncertain affine expression."""

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    def __init__(self, x):
        """Construct an :class:`UncertainNorm`.

        :param x:
            The uncertain affine expression to denote the norm of.
        :type x:
            ~picos.expressions.uncertain.uexp_affine.UncertainAffineExpression
        """
        if not isinstance(x, UncertainAffineExpression):
            raise TypeError("Can only form the uncertain norm of an uncertain "
                "affine expression, not of {}.".format(type(x).__name__))

        # Refine perturbation set from ellipsoidal to unit ball.
        if x.uncertain and isinstance(x.universe, ConicPerturbationSet) \
        and x.universe.ellipsoidal:
            x = x.replace_mutables(x.universe.unit_ball_form[1])
            assert isinstance(x.universe, UnitBallPerturbationSet)

        if len(x) == 1:
            typeStr = "Uncertain Absolute Value"
            symbStr = glyphs.abs(x.string)
        else:
            typeStr = "Uncertain {} Norm".format(
                "Euclidean" if 1 in x.shape else "Frobenius")
            symbStr = glyphs.norm(x.string)

        Expression.__init__(self, typeStr, symbStr)

        self._x = x

    # --------------------------------------------------------------------------
    # Properties.
    # --------------------------------------------------------------------------

    @property
    def x(self):
        """Uncertain affine expression under the norm."""
        return self._x

    # --------------------------------------------------------------------------
    # Abstract method implementations for Expression, except _predict.
    # --------------------------------------------------------------------------

    @cached_unary_operator
    def _get_refined(self):
        """Implement :meth:`~.expression.Expression._get_refined`."""
        if self.certain:
            return Norm(self._x.refined)
        else:
            return self

    Subtype = namedtuple("Subtype", ("argdim", "universe_type"))

    def _get_subtype(self):
        """Implement :meth:`~.expression.Expression._get_subtype`."""
        return self.Subtype(len(self._x), self.universe.type)

    def _get_value(self):
        value = self._x._get_value()

        if len(value) == 1:
            return abs(value)
        else:
            return cvxopt.matrix(numpy.linalg.norm(numpy.ravel(cvx2np(value))))

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
    # Python special method implementations, except constraint-creating ones.
    # --------------------------------------------------------------------------

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __pow__(self, other):
        if isinstance(other, AffineExpression):
            if not other.constant or other.value != 2:
                raise NotImplementedError(
                    "You may only take an uncertain norm to the power of two.")

            return UncertainSquaredNorm(self._x)
        else:
            return NotImplemented

    # --------------------------------------------------------------------------
    # Constraint-creating operators and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        AE = AffineExpression
        BAE = BiaffineExpression
        UAE = UncertainAffineExpression
        CPS = ConicPerturbationSet
        UBPS = UnitBallPerturbationSet
        SPS = ScenarioPerturbationSet

        if issubclass(other.clstype, BAE) and other.subtype.dim != 1:
            return NotImplemented

        if relation is not operator.__le__:
            return NotImplemented

        if issubclass(subtype.universe_type.clstype, UBPS):
            if issubclass(other.clstype, (AE, UAE)):
                if issubclass(other.clstype, UAE) \
                and not issubclass(other.subtype.universe_type.clstype, CPS):
                    return NotImplemented

                if issubclass(other.clstype, UAE):
                    bound_universe_subtype = other.subtype.universe_type.subtype
                else:
                    bound_universe_subtype = None

                return BallUncertainNormConstraint.make_type(
                    dim=subtype.argdim,
                    norm_universe_subtype=subtype.universe_type.subtype,
                    bound_universe_subtype=bound_universe_subtype)
        elif issubclass(subtype.universe_type.clstype, SPS):
            if issubclass(other.clstype, (AE, UAE)):
                if issubclass(other.clstype, UAE) \
                and not issubclass(other.subtype.universe_type.clstype, SPS):
                    return NotImplemented

                return ScenarioUncertainConicConstraint.make_type(
                    dim=(subtype.argdim + 1),
                    scenario_count=subtype.universe_type.subtype.scenario_count,
                    cone_type=SecondOrderCone.make_type(dim=None))
        else:
            return NotImplemented

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        if isinstance(self._x.universe, UnitBallPerturbationSet):
            if isinstance(other, (AffineExpression, UncertainAffineExpression)):
                # Upper bound must be certain or conically uncertain.
                if isinstance(other, UncertainAffineExpression) \
                and not isinstance(other.universe, ConicPerturbationSet):
                    raise TypeError(
                        "May only upper bound a conically uncertain norm with a"
                        " certain or another conically uncertain expression.")

                # Uncertain upper bound must have independent uncertainty.
                # NOTE: Can only be predicted up to the perturbation type.
                if isinstance(other, UncertainAffineExpression) \
                and self.perturbation is other.perturbation:
                    raise ValueError("If the upper bound to a conically "
                        "uncertain norm is itself uncertain, then the "
                        "uncertainty in both sides must be independent "
                        "(distinct perturbation parameters).")

                return BallUncertainNormConstraint(self, other)
        elif isinstance(self._x.universe, ScenarioPerturbationSet):
            if isinstance(other, (AffineExpression, UncertainAffineExpression)):
                # Upper bound must be certain or scenario uncertain.
                if isinstance(other, UncertainAffineExpression) \
                and not isinstance(other.universe, ScenarioPerturbationSet):
                    raise TypeError(
                        "May only upper bound a scenario uncertain norm with a"
                        " certain or another scenario uncertain expression.")

                # Uncertain upper bound must have equal uncertainty.
                # NOTE: Can only be predicted up to the perturbation type.
                if isinstance(other, UncertainAffineExpression) \
                and self.perturbation is not other.perturbation:
                    raise ValueError(
                        "If the upper bound to a scenario uncertain norm is "
                        "itself uncertain, then the uncertainty in both sides "
                        "must be equal (same perturbation parameter).")

                return (other // self._x.vec) << SecondOrderCone()
        elif isinstance(self._x.universe, ConicPerturbationSet):
            # The universe could not be refined to a UnitBallPerturbationSet.
            assert not self._x.universe.ellipsoidal
            raise TypeError("Upper-bounding an uncertain norm whose "
                "perturbation parameter lives in a conic perturbation set is "
                "only supported if the perturbation set is an ellipsoid.")
        else:
            raise TypeError("Upper-bounding an uncertain norm whose "
                "perturbation parameter is described by an instance of {} is "
                "not supported.".format(self._x.universe.__class__.__name__))

        # Make sure the Python NotImplemented-triggered TypeError works.
        assert not isinstance(other,
            (AffineExpression, UncertainAffineExpression))

        return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
