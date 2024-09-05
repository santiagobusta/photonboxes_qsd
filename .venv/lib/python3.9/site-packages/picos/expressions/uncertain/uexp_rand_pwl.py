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

"""Implements :class:`RandomExtremumAffine`."""

import operator
from abc import abstractmethod
from collections import namedtuple

import cvxopt

from ...apidoc import api_end, api_start
from ...caching import cached_unary_operator
from ...constraints import (Constraint,
                            MomentAmbiguousExtremumAffineConstraint,
                            WassersteinAmbiguousExtremumAffineConstraint)
from ...formatting import arguments
from ..data import convert_operands
from ..exp_affine import AffineExpression, Constant
from ..exp_extremum import (ExtremumBase, MaximumBase, MaximumConvex,
                            MinimumBase, MinimumConcave)
from ..expression import Expression, refine_operands, validate_prediction
from .pert_moment import MomentAmbiguitySet
from .pert_wasserstein import WassersteinAmbiguitySet
from .uexp_affine import UncertainAffineExpression
from .uexpression import UncertainExpression

_API_START = api_start(globals())
# -------------------------------


class RandomExtremumAffine(ExtremumBase, UncertainExpression, Expression):
    """Base class for random convex or concave piecewise linear expressions.

    .. note::

        Unlike other uncertain expression types, this class is limited to
        uncertainty of stochastic nature, where using the expression in a
        constraint or as an objective function implicitly takes the (worst-case)
        expectation of the expression. Non-stochastic uncertainty is handled
        within :class:`~picos.expressions.MaximumConvex` and
        :class:`~picos.expressions.MinimumConcave` as their behavior, although
        designed for certain expression types, already encodes the worst-case
        approach of the robust optimization paradigm.
    """

    # --------------------------------------------------------------------------
    # Additional abstract methods extending (in spirit) ExtremumBase.
    # --------------------------------------------------------------------------

    @property
    @abstractmethod
    def _certain_class(self):
        pass

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    def __init__(self, expressions):
        """Construct a :class:`RandomExtremumAffine`.

        :param expressions:
            A collection of uncertain affine expressions whose uncertainty is of
            stochastic nature.
        """
        # Load constant data and refine expressions.
        expressions = tuple(
            x.refined if isinstance(x, Expression) else Constant(x)
            for x in expressions)

        # Check expression types.
        if not all(isinstance(x, (AffineExpression, UncertainAffineExpression))
                for x in expressions):
            raise TypeError("{} can only denote the extremum of (uncertain) "
                "affine expressions.".format(self.__class__.__name__))

        # Check expression dimension.
        if not all(x.scalar for x in expressions):
            raise TypeError("{} can only denote the extremum of scalar "
                "expressions.".format(self.__class__.__name__))

        perturbations = tuple(set(
            x.perturbation for x in expressions if x.uncertain))

        # Check for a unique perturbation parameter.
        if len(perturbations) > 1:
            raise ValueError("{} can only denote the extremum of uncertain "
                "affine expressions that depend on at most one perturbation "
                "parameter, found {}."
                .format(self.__class__.__name__, len(perturbations)))

        perturbation = perturbations[0] if perturbations else None
        universe = perturbation.universe if perturbation else None

        # Check for a supported perturbation type.
        if universe and not universe.distributional:
            raise TypeError("{} can only represent uncertainty parameterized by"
                " a distribution or distributional ambiguity set, not {}."
                .format(self.__class__.__name__, universe.__class__.__name__))

        typeStr = "{} Uncertain Piecewise Linear Function".format(
            self._property_word.title())

        symbStr = self._extremum_glyph(
            arguments([x.string for x in expressions]))

        Expression.__init__(self, typeStr, symbStr)

        self._expressions = expressions
        self._perturbation = perturbation

    # --------------------------------------------------------------------------
    # Abstract method implementations for ExtremumBase.
    # --------------------------------------------------------------------------

    @property
    def expressions(self):
        """The expressions under the extremum."""
        return self._expressions

    # --------------------------------------------------------------------------
    # Method overridings for UncertainExpression.
    # --------------------------------------------------------------------------

    @property
    def perturbation(self):
        """Fast override for :class:`~.uexpression.UncertainExpression`."""
        return self._perturbation

    # --------------------------------------------------------------------------
    # Abstract method implementations for Expression, except _predict.
    # --------------------------------------------------------------------------

    @cached_unary_operator
    def _get_refined(self):
        """Implement :meth:`~.expression.Expression._get_refined`."""
        if len(self._expressions) == 1:
            return self._expressions[0]
        elif all(x.constant for x in self._expressions):
            return self._extremum(self._expressions, key=lambda x: x.safe_value)
        elif all(x.certain for x in self._expressions):
            return self._certain_class(x.refined for x in self._expressions)
        else:
            return self

    Subtype = namedtuple("Subtype", ("argnum", "universe_type"))

    def _get_subtype(self):
        """Implement :meth:`~.expression.Expression._get_subtype`."""
        return self.Subtype(self.argnum, self.universe.type)

    def _get_value(self):
        return cvxopt.matrix(self._extremum(
            x.safe_value for x in self._expressions))

    # --------------------------------------------------------------------------
    # Constraint-creating operators and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        convex = issubclass(cls, RandomMaximumAffine)
        concave = issubclass(cls, RandomMinimumAffine)

        if relation == operator.__le__:
            if not convex:
                return NotImplemented
        elif relation == operator.__ge__:
            if not concave:
                return NotImplemented
        else:
            return NotImplemented

        if not issubclass(other.clstype, AffineExpression) \
        or other.subtype.dim != 1:
            return NotImplemented

        if issubclass(subtype.universe_type.clstype, MomentAmbiguitySet):
            return MomentAmbiguousExtremumAffineConstraint.make_type(
                extremum_argnum=subtype.argnum,
                universe_subtype=subtype.universe_type.subtype)
        elif issubclass(subtype.universe_type.clstype, WassersteinAmbiguitySet):
            return WassersteinAmbiguousExtremumAffineConstraint.make_type(
                extremum_argnum=subtype.argnum,
                universe_subtype=subtype.universe_type.subtype)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        if not self.convex:
            raise TypeError("Cannot upper-bound the nonconvex expression {}."
                .format(self.string))

        if not isinstance(other, AffineExpression):
            return NotImplemented

        if isinstance(self.universe, MomentAmbiguitySet):
            return MomentAmbiguousExtremumAffineConstraint(
                self, Constraint.LE, other)
        elif isinstance(self.universe, WassersteinAmbiguitySet):
            return WassersteinAmbiguousExtremumAffineConstraint(
                self, Constraint.LE, other)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __ge__(self, other):
        if not self.concave:
            raise TypeError("Cannot lower-bound the nonconcave expression {}."
                .format(self.string))

        if not isinstance(other, AffineExpression):
            return NotImplemented

        if isinstance(self.universe, MomentAmbiguitySet):
            return MomentAmbiguousExtremumAffineConstraint(
                self, Constraint.GE, other)
        elif isinstance(self.universe, WassersteinAmbiguitySet):
            return WassersteinAmbiguousExtremumAffineConstraint(
                self, Constraint.GE, other)

        return NotImplemented


class RandomMaximumAffine(MaximumBase, RandomExtremumAffine):
    """The maximum over a set of random affine expressions."""

    # --------------------------------------------------------------------------
    # Abstract method implementations for ExtremumBase.
    # --------------------------------------------------------------------------

    @property
    def _other_class(self):
        return RandomMinimumAffine

    # --------------------------------------------------------------------------
    # Abstract method implementations for RandomExtremumAffine.
    # --------------------------------------------------------------------------

    @property
    def _certain_class(self):
        return MaximumConvex


class RandomMinimumAffine(MinimumBase, RandomExtremumAffine):
    """The minimum over a set of random affine expressions."""

    # --------------------------------------------------------------------------
    # Abstract method implementations for ExtremumBase.
    # --------------------------------------------------------------------------

    @property
    def _other_class(self):
        return RandomMaximumAffine

    # --------------------------------------------------------------------------
    # Abstract method implementations for RandomExtremumAffine.
    # --------------------------------------------------------------------------

    @property
    def _certain_class(self):
        return MinimumConcave


# --------------------------------------
__all__ = api_end(_API_START, globals())
