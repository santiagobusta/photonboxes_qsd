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

"""Implements :class:`MaximumConvex` and :class:`MinimumConcave`."""

import operator
from abc import ABC, abstractmethod
from collections import namedtuple

import cvxopt

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_selfinverse_unary_operator
from ..constraints import Constraint, ExtremumConstraint
from ..formatting import arguments
from .data import convert_operands
from .exp_affine import AffineExpression, Constant
from .expression import Expression, refine_operands, validate_prediction

_API_START = api_start(globals())
# -------------------------------


class ExtremumBase(ABC):
    """Base class for :class:`Extremum` and similar classes.

    In particular, this is also used by the uncertain
    :class:`~.uexp_rand_pwl.RandomExtremumAffine`.

    Must be inherited with priority with respect to
    :class:`~picos.expressions.Expression`.
    """

    # --------------------------------------------------------------------------
    # Implemented by MaximumBase and MinimumBase.
    # --------------------------------------------------------------------------

    @property
    @abstractmethod
    def _extremum(self):
        pass

    @property
    @abstractmethod
    def _extremum_word(self):
        pass

    @property
    @abstractmethod
    def _extremum_glyph(self):
        pass

    @abstractmethod
    def _property(self, x):
        pass

    @property
    @abstractmethod
    def _property_word(self):
        pass

    # --------------------------------------------------------------------------
    # Implemented by the expression class.
    # --------------------------------------------------------------------------

    @property
    @abstractmethod
    def _other_class(self):
        pass

    @property
    @abstractmethod
    def expressions(self):
        """The expressions under the extremum."""
        pass

    # --------------------------------------------------------------------------
    # Provided/implemented by this base class.
    # --------------------------------------------------------------------------

    @property
    def _extremum_short_word(self):
        return self._extremum_word[:3]

    def _get_mutables(self):
        return frozenset(mtb for x in self.expressions for mtb in x.mutables)

    def _replace_mutables(self, mapping):
        return self.__class__(
            x._replace_mutables(mapping) for x in self.expressions)

    def _freeze_mutables(self, freeze):
        return self.__class__(
            x._freeze_mutables(freeze) for x in self.expressions)

    @property
    def argnum(self):
        """Number of expressions under the extremum."""
        return len(self.expressions)

    @classmethod
    def _mul(cls, self, other, forward):
        if isinstance(other, AffineExpression) and other.constant:
            factor = other.safe_value

            if not factor:
                return AffineExpression.zero()
            elif factor == 1:
                return self
            elif factor == -1:
                return -self

            if forward:
                string = glyphs.clever_mul(self.string, other.string)
            else:
                string = glyphs.clever_mul(other.string, self.string)

            cls_ = self.__class__ if factor > 0 else self._other_class

            product = cls_(factor*x for x in self.expressions)
            product._typeStr = "Scaled " + product._typeStr
            product._symbStr = string

            return product

        if forward:
            return Expression.__mul__(self, other)
        else:
            return Expression.__rmul__(self, other)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __mul__(self, other):
        return ExtremumBase._mul(self, other, True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __rmul__(self, other):
        return ExtremumBase._mul(self, other, False)

    @cached_selfinverse_unary_operator
    def __neg__(self):
        return self._other_class(-x for x in self.expressions)


class MaximumBase:
    """Base implementation of :class:`ExtremumBase` for maximums."""

    # --------------------------------------------------------------------------
    # Abstract method implementations for ExtremumBase.
    # --------------------------------------------------------------------------

    @property
    def _extremum(self):
        return max

    @property
    def _extremum_word(self):
        return "maximum"

    @property
    def _extremum_glyph(self):
        return glyphs.max

    def _property(self, x):
        return x.convex

    @property
    def _property_word(self):
        return "convex"

    # --------------------------------------------------------------------------
    # Abstract method implementations for Expression.
    # --------------------------------------------------------------------------

    def _is_convex(self):
        return True

    def _is_concave(self):
        return False


class MinimumBase:
    """Base implementation of :class:`ExtremumBase` for minimums."""

    # --------------------------------------------------------------------------
    # Abstract method implementations for ExtremumBase.
    # --------------------------------------------------------------------------

    @property
    def _extremum(self):
        return min

    @property
    def _extremum_word(self):
        return "minimum"

    @property
    def _extremum_glyph(self):
        return glyphs.min

    def _property(self, x):
        return x.concave

    @property
    def _property_word(self):
        return "concave"

    # --------------------------------------------------------------------------
    # Abstract method implementations for Expression.
    # --------------------------------------------------------------------------

    def _is_convex(self):
        return False

    def _is_concave(self):
        return True


class Extremum(ExtremumBase, Expression):
    """Base class for :class:`MaximumConvex` and :class:`MinimumConcave`.

    .. note::

        This can represent the maximum (minimum) over convex (concave) uncertain
        expressions as long as the uncertainty is not of stochastic nature.
        In this case, the extremum implicitly goes over the perturbation
        parameters as well.
    """

    # --------------------------------------------------------------------------
    # Initialization and factory methods.
    # --------------------------------------------------------------------------

    def __init__(self, expressions):
        """Construct a :class:`MaximumConvex` or :class:`MinimumConcave`.

        :param expressions:
            A collection of all convex or all concave expressions.
        """
        # Multidimensional expressions are iterable and yield expressions but
        # denoting their extremum is handled by SumExtremes.
        if isinstance(expressions, Expression):
            word = self._property_word
            raise TypeError("The class {} is not designed to represent the {} "
                "over (the elements of) a single expression. This is the job of"
                " SumExtremes. Use picos.{} to use whichever is appropriate."
                .format(self.__class__.__name__, word, word[:3]))

        # Load constant data and refine expressions.
        expressions = tuple(
            x.refined if isinstance(x, Expression) else Constant(x)
            for x in expressions)

        # Validate that every expression is convex (concave) and scalar.
        for x in expressions:
            if not self._property(x):
                raise TypeError("The expression {} is not {}."
                    .format(x.string, self._property_word))

            if not x.scalar:
                raise TypeError(
                    "The expression {} is not scalar.".format(x.string))

        # Handle uncertain but not random expressions.
        if any(x.uncertain and x.random for x in expressions):
            raise NotImplementedError("The (fallback) class {} does not handle "
                "random expressions as taking the expectation does not commute "
                "with taking the extremum.".format(self.__class__.__name__))

        self._expressions = expressions

        typeStr = "{} of {} Functions".format(
            self._extremum_word.title(), self._property_word.title())

        symbStr = self._extremum_glyph(arguments([
            x.string if x.certain
            else x.worst_case_string(self._extremum_short_word)
            for x in expressions]))

        Expression.__init__(self, typeStr, symbStr)

    # --------------------------------------------------------------------------
    # Abstract method implementations for ExtremumBase.
    # --------------------------------------------------------------------------

    @property
    def expressions(self):
        """The expressions under the extremum."""
        return self._expressions

    # --------------------------------------------------------------------------
    # Abstract method implementations for Expression, except _predict.
    # --------------------------------------------------------------------------

    def _get_refined(self):
        if len(self._expressions) == 1:
            return self._expressions[0]
        elif all(x.constant for x in self._expressions):
            return self._extremum(self._expressions, key=lambda x: x.safe_value)
        else:
            return self

    Subtype = namedtuple("Subtype", ("types",))

    def _get_subtype(self):
        return self.Subtype(tuple(x.type for x in self._expressions))

    def _get_value(self):
        return cvxopt.matrix(self._extremum(
            x.safe_value if x.certain
            else x.worst_case_value(self._extremum_short_word)
            for x in self._expressions))

    # --------------------------------------------------------------------------
    # Constraint-creating operators, and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        convex = issubclass(cls, MaximumConvex)
        concave = issubclass(cls, MinimumConcave)

        if relation == operator.__le__:
            if not convex:
                return NotImplemented

            if not issubclass(other.clstype, AffineExpression) \
            or other.subtype.dim != 1:
                return NotImplemented

            return ExtremumConstraint.make_type(
                lhs_types=subtype.types, relation=Constraint.LE, rhs_type=other)
        elif relation == operator.__ge__:
            if not concave:
                return NotImplemented

            if not issubclass(other.clstype, AffineExpression) \
            or other.subtype.dim != 1:
                return NotImplemented

            return ExtremumConstraint.make_type(
                lhs_types=subtype.types, relation=Constraint.GE, rhs_type=other)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        if not self.convex:
            raise TypeError("Cannot upper-bound the nonconvex expression {}."
                .format(self.string))

        if isinstance(other, AffineExpression):
            return ExtremumConstraint(self, Constraint.LE, other)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __ge__(self, other):
        if not self.concave:
            raise TypeError("Cannot lower-bound the nonconcave expression {}."
                .format(self.string))

        if isinstance(other, AffineExpression):
            return ExtremumConstraint(self, Constraint.GE, other)

        return NotImplemented


class MaximumConvex(MaximumBase, Extremum):
    """The maximum over a set of convex scalar expressions.

    :Example:

    >>> import picos
    >>> x = picos.RealVariable("x", 4)
    >>> a = abs(x)
    >>> b = picos.sum(x)
    >>> c = picos.max([a, b]); c
    <Maximum of Convex Functions: max(‖x‖, ∑(x))>
    >>> 2*c
    <Scaled Maximum of Convex Functions: 2·max(‖x‖, ∑(x))>
    >>> c <= 5
    <Maximum of Convex Functions Constraint: max(‖x‖, ∑(x)) ≤ 5>
    """

    # --------------------------------------------------------------------------
    # Abstract method implementations for ExtremumBase.
    # --------------------------------------------------------------------------

    @property
    def _other_class(self):
        return MinimumConcave


class MinimumConcave(MinimumBase, Extremum):
    """The minimum over a set of concave scalar expressions.

    :Example:

    >>> import picos
    >>> x = picos.RealVariable("x", 4)
    >>> a = picos.sum(x)
    >>> b = 2*a
    >>> c = picos.min([a, b]); c
    <Minimum of Concave Functions: min(∑(x), 2·∑(x))>
    >>> -1*c
    <Maximum of Convex Functions: max(-∑(x), -2·∑(x))>
    >>> C = 5 <= c; C
    <Minimum of Concave Functions Constraint: min(∑(x), 2·∑(x)) ≥ 5>
    >>> x.value = 1
    >>> C.slack
    -1.0
    """

    # --------------------------------------------------------------------------
    # Abstract method implementations for ExtremumBase.
    # --------------------------------------------------------------------------

    @property
    def _other_class(self):
        return MaximumConvex


# --------------------------------------
__all__ = api_end(_API_START, globals())
