# ------------------------------------------------------------------------------
# Copyright (C) 2021 Maximilian Stahlberg
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

"""Implements the :class:`WeightedSum` fallback class."""

import operator
from collections import namedtuple
from functools import reduce

import cvxopt

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property, cached_selfinverse_unary_operator
from ..constraints import Constraint, WeightedSumConstraint
from .data import convert_operands, load_dense_data
from .exp_affine import AffineExpression, Constant
from .expression import Expression, refine_operands, validate_prediction

_API_START = api_start(globals())
# -------------------------------


class WeightedSum(Expression):
    """A convex or concave weighted sum of scalar expressions."""

    # --------------------------------------------------------------------------
    # Initialization and properties.
    # --------------------------------------------------------------------------

    def __init__(self, expressions, weights=1, opstring=None):
        """Construct a weighted sum of expressions.

        :param expressions:
            A collection of scalar expressions.

        :param weights:
            A constant weight vector.

        :param str opstring:
            Used by PICOS internally when this class is tried as a last fallback
            to represent the result of an otherwise unsupported product or sum.
        """
        try:
            # Avoid iterating over affine expressions.
            if isinstance(expressions, Expression):
                raise TypeError("{} is not designed to represent the sum over "
                    "(the elements of) a single expression. Use picos.sum to "
                    "select the correct class automatically."
                    .format(self.__class__.__name__))

            # Load constant data and refine expressions.
            expressions = tuple(
                x.refined if isinstance(x, Expression) else Constant(x)
                for x in expressions)

            if not expressions:
                raise ValueError("Need at least one expression.")

            # Require that every expression is scalar.
            if not all(x.scalar for x in expressions):
                raise TypeError("Not all summands are scalar.")

            # Load weights as a CVXOPT dense column vector.
            weights = load_dense_data(weights, (len(expressions), 1), "d")[0]

            # Never create a nested WeightedSum.
            # NOTE: This ensures that WeightedSumConstraintReformulation needs
            #       to run just once to get rid of all WeightedSumConstraint.
            if any(isinstance(x, WeightedSum) for x in expressions):
                ux, uw = [], []  # Unpacked expressions/weights.

                for x, w in zip(expressions, weights):
                    if isinstance(x, WeightedSum):
                        ux.extend(x._expressions)
                        uw.extend(w * x._weights)
                    else:
                        ux.append(x)
                        uw.append(w)

                assert not any(isinstance(x, WeightedSum) for x in ux)

                expressions = tuple(ux)
                weights = load_dense_data(uw, (len(ux), 1), "d")[0]

            # Determine convexity of expressions.
            convex = all((x.convex and w >= 0) or (x.concave and w <= 0)
                for x, w in zip(expressions, weights))
            concave = all((x.concave and w >= 0) or (x.convex and w <= 0)
                for x, w in zip(expressions, weights))

            # Don't handle uncertain expressions.
            # TODO: Consider handling sums with one uncertain summand.
            if any(x.uncertain for x in expressions):
                raise NotImplementedError(
                    "{} does not handle uncertain summands at this point."
                    .format(self.__class__.__name__))

            self._expressions = expressions
            self._weights = weights
            self._convex = convex
            self._concave = concave

            typeStrWords = []

            if convex and concave:
                typeStrWords.append("Affine")  # Manually crafted.
            elif convex:
                typeStrWords.append("Convex")
            elif concave:
                typeStrWords.append("Concave")

            if not all(w == 1 for w in weights):
                typeStrWords.append("Weighted")

            typeStrWords.append("Sum")

            typeStr = " ".join(typeStrWords)
            symbStr = reduce(glyphs.clever_add, (
                glyphs.clever_mul(glyphs.scalar(w), x.string)
                for w, x in zip(weights, expressions)))

            Expression.__init__(self, typeStr, symbStr)
        except Exception as error:
            if not opstring:
                raise

            raise TypeError("Cannot represent {} as a weighted sum: {}"
                .format(opstring, error)) from None

    @property
    def expressions(self):
        """The expressions being summed, without their coefficients."""
        return self._expressions

    @cached_property
    def weights(self):
        """The coefficient vector as a PICOS column vector."""
        return Constant("w", self._weights)

    # --------------------------------------------------------------------------
    # Abstract method implementations for Expression, except _predict.
    # --------------------------------------------------------------------------

    # TODO: Merge expressions that can be merged.
    def _get_refined(self):
        if not self._weights:
            return AffineExpression.zero()
        elif all(x.constant for x in self._expressions):
            return Constant(self.string, self.safe_value, (1, 1))
        elif len(self._expressions) == 1 and self._weights[0] == 1:
            return self._expressions[0]
        elif 0 in self._weights:
            return self.__class__(*(zip(*(
                ew for ew in zip(self._expressions, self._weights) if ew[1]))))
        else:
            return self

    Subtype = namedtuple("Subtype", (
        "convex", "concave", "types", "nonneg_weights"))

    def _get_subtype(self):
        return self.Subtype(self.convex, self.concave,
            tuple(x.type for x in self._expressions),
            tuple(self.weights.np >= 0))

    def _get_value(self):
        values = cvxopt.matrix([x.safe_value for x in self._expressions],
            (1, len(self._expressions)))
        return values * self._weights

    def _get_mutables(self):
        return reduce(
            frozenset.union, (x._get_mutables() for x in self._expressions))

    def _is_convex(self):
        return self._convex

    def _is_concave(self):
        return self._concave

    def _replace_mutables(self, mapping):
        return self.__class__(
            (x._replace_mutables(mapping) for x in self._expressions),
            self._weights)

    def _freeze_mutables(self, freeze):
        return self.__class__(
            (x._freeze_mutables(freeze) for x in self._expressions),
            self._weights)

    # --------------------------------------------------------------------------
    # Python special method implementations, except constraint-creating ones.
    # NOTE: WeightedSum is used by Expression as a fallback class, so all
    #       operations are concluded here (return result or raise exception).
    # --------------------------------------------------------------------------

    @cached_selfinverse_unary_operator
    def __neg__(self):
        return self.__class__(self._expressions, -self._weights)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __add__(self, other):
        opstring = "{} plus {}".format(repr(self), repr(other))
        return self.__class__(self._expressions + (other,),
            cvxopt.matrix([self._weights, 1]), opstring)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __radd__(self, other):
        opstring = "{} plus {}".format(repr(other), repr(self))
        return self.__class__((other,) + self._expressions,
            cvxopt.matrix([1, self._weights]), opstring)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __sub__(self, other):
        opstring = "{} minus {}".format(repr(self), repr(other))
        return self.__class__(self._expressions + (other,),
            cvxopt.matrix([self._weights, -1]), opstring)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __rsub__(self, other):
        opstring = "{} minus {}".format(repr(other), repr(self))
        return self.__class__((other,) + self._expressions,
            cvxopt.matrix([1, -self._weights]), opstring)

    def _mul(self, other, forward):
        if isinstance(other, AffineExpression) and other.constant:
            value = other.safe_value

            if value == 0:
                return Constant(0)
            elif value == 1:
                return self
            else:
                p = self.__class__(self._expressions, value*self._weights)

                if forward:
                    p._symbStr = glyphs.clever_mul(self.string, other.string)
                else:
                    p._symbStr = glyphs.clever_mul(other.string, self.string)

                return p
        else:
            return NotImplemented

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __mul__(self, other):
        return self._mul(other, True)

    @convert_operands(scalarRHS=True)
    @refine_operands()
    def __rmul__(self, other):
        return self._mul(other, False)

    # --------------------------------------------------------------------------
    # Constraint-creating operators, and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        assert isinstance(subtype, cls.Subtype)

        if relation == operator.__le__:
            if not subtype.convex:
                return NotImplemented

            if not issubclass(other.clstype, AffineExpression) \
            or other.subtype.dim != 1:
                return NotImplemented

            return WeightedSumConstraint.make_type(
                lhs_types=subtype.types, relation=Constraint.LE, rhs_type=other,
                nonneg_weights=subtype.nonneg_weights)
        elif relation == operator.__ge__:
            if not subtype.concave:
                return NotImplemented

            if not issubclass(other.clstype, AffineExpression) \
            or other.subtype.dim != 1:
                return NotImplemented

            return WeightedSumConstraint.make_type(
                lhs_types=subtype.types, relation=Constraint.GE, rhs_type=other,
                nonneg_weights=subtype.nonneg_weights)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        if not self.convex:
            raise TypeError("Cannot upper-bound the nonconvex expression {}."
                .format(self.string))

        if isinstance(other, AffineExpression):
            return WeightedSumConstraint(self, Constraint.LE, other)

        return NotImplemented

    @convert_operands(scalarRHS=True)
    @validate_prediction
    @refine_operands()
    def __ge__(self, other):
        if not self.concave:
            raise TypeError("Cannot lower-bound the nonconcave expression {}."
                .format(self.string))

        if isinstance(other, AffineExpression):
            return WeightedSumConstraint(self, Constraint.GE, other)

        return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
