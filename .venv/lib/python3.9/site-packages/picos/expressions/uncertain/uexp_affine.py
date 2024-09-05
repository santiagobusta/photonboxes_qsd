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

"""Implements :class:`UncertainAffineExpression`."""

import operator
from collections import namedtuple

from ...apidoc import api_end, api_start
from ...caching import cached_property, cached_unary_operator
from ...constraints.uncertain import (ConicallyUncertainAffineConstraint,
                                      ScenarioUncertainConicConstraint)
from ..data import convert_operands, cvxopt_K
from ..exp_affine import AffineExpression
from ..exp_biaffine import BiaffineExpression
from ..expression import ExpressionType, refine_operands, validate_prediction
from ..variables import BaseVariable
from .uexpression import UncertainExpression

# NOTE: May not import ConicPerturbationSet from .pert_conic here because the
#       latter imports from .perturbation which in turn needs to import
#       UncertainAffineExpression as a base class of Perturbation from here.


_API_START = api_start(globals())
# -------------------------------


class UncertainAffineExpression(UncertainExpression, BiaffineExpression):
    r"""A multidimensional uncertain affine expression.

    This expression has the form

    .. math::

        A(x,\theta) = B(x,\theta) + P(x) + Q(\theta) + C

    where :math:`B`, :math:`P`, :math:`Q`, :math:`C` and :math:`x` are defined
    as for the :class:`~.exp_biaffine.BiaffineExpression` base class and
    :math:`\theta` is an uncertain perturbation parameter confined to
    (distributed according to) a perturbation set (distribution) :math:`\Theta`.

    If no coefficient matrices defining :math:`B` and :math:`P` are provided,
    then this expression represents uncertain data confined to an uncertainty
    set :math:`\{Q(\theta) + C \mid \theta \in \Theta\}` (distributed according
    to :math:`Q(\Theta) + C`) where :math:`C` can be understood as a nominal
    data value while :math:`Q(\theta)` quantifies the uncertainty on the data.
    """

    # --------------------------------------------------------------------------
    # Abstract method implementations for Expression, except _predict.
    # --------------------------------------------------------------------------

    Subtype = namedtuple("Subtype", ("shape", "universe_type"))
    Subtype.dim = property(lambda self: self.shape[0] * self.shape[1])

    def _get_subtype(self):
        """Implement :meth:`~.expression.Expression._get_subtype`."""
        return self.Subtype(self._shape, self.universe.type)

    # --------------------------------------------------------------------------
    # Method overridings for Expression.
    # --------------------------------------------------------------------------

    def _get_refined(self):
        """Implement :meth:`~.expression.Expression._get_refined`."""
        if self.certain:
            return AffineExpression(self.string, self.shape, self._coefs)
        else:
            return self

    # --------------------------------------------------------------------------
    # Abstract method implementations for BiaffineExpression.
    # --------------------------------------------------------------------------

    @classmethod
    def _get_bilinear_terms_allowed(cls):
        """Implement for :class:`~.exp_biaffine.BiaffineExpression`."""
        return True

    @classmethod
    def _get_parameters_allowed(cls):
        """Implement for :class:`~.exp_biaffine.BiaffineExpression`."""
        return True

    @classmethod
    def _get_basetype(cls):
        """Implement :meth:`~.exp_biaffine.BiaffineExpression._get_basetype`."""
        return UncertainAffineExpression

    @classmethod
    def _get_typecode(cls):
        """Implement :meth:`~.exp_biaffine.BiaffineExpression._get_typecode`."""
        return "d"

    # --------------------------------------------------------------------------
    # Method overridings for BiaffineExpression.
    # --------------------------------------------------------------------------

    @classmethod
    def _get_type_string_base(cls):
        """Override for :class:`~.exp_biaffine.BiaffineExpression`."""
        # TODO: Allow the strings "Uncertain (Linear Expression|Constant)".
        return "Uncertain {}".format("Affine Expression")

    def __init__(self, string, shape=(1, 1), coefficients={}):
        """Construct an :class:`UncertainAffineExpression`.

        Extends :meth:`.exp_biaffine.BiaffineExpression.__init__`.

        This constructor is meant for internal use. As a user, you will want to
        first define a universe (e.g.
        :class:`~.pert_conic.ConicPerturbationSet`) for a
        :class:`perturbation parameter <.perturbation.Perturbation>` and use
        that parameter as a building block to create more complex uncertain
        expressions.
        """
        from .perturbation import Perturbation

        BiaffineExpression.__init__(self, string, shape, coefficients)

        if not all(isinstance(prm, Perturbation) for prm in self.parameters):
            raise TypeError("Uncertain affine expressions may not depend on "
                "parameters other than perturbation parameters.")

        for pair in self._bilinear_coefs:
            x, y = pair

            d = sum(isinstance(var, BaseVariable) for var in pair)
            p = sum(isinstance(var, Perturbation) for var in pair)

            # Forbid a quadratic part.
            if d > 1:
                raise TypeError("Tried to create an uncertain affine "
                    "expression that is {}.".format(
                        "quadratic in {}".format(x.string) if x is y else
                        "biaffine in {} and {}".format(x.string, y.string)))

            # Forbid quadratic dependence on the perturbation parameter.
            if p > 1:
                assert x is y
                raise NotImplementedError("Uncertain affine expressions may "
                    "only depend affinely on the perturbation parameter. Tried "
                    "to create one that is quadratic in {}.".format(x.string))

            assert d == 1 and p == 1

    @classmethod
    def _common_basetype(cls, other, reverse=False):
        from ..exp_affine import AffineExpression

        # HACK: AffineExpression is not a subclass of UncertainAffineExpression
        #       but we can treat it as one when it comes to basetype detection.
        if issubclass(other._get_basetype(), AffineExpression):
            return UncertainAffineExpression
        else:
            return BiaffineExpression._common_basetype.__func__(
                cls, other, reverse)

    def _is_convex(self):
        return True

    def _is_concave(self):
        return True

    # --------------------------------------------------------------------------
    # Class-specific properties.
    # --------------------------------------------------------------------------

    @cached_property
    def _sorted_bilinear_coefs(self):
        """Bilinear part coefficients with perturbation on the right side."""
        from .perturbation import Perturbation

        coefs = {}
        for mtbs, coef in self._bilinear_coefs.items():
            x, y = mtbs

            if isinstance(x, Perturbation):
                # Obtain a fitting commutation matrix.
                K = cvxopt_K(y.dim, x.dim, self._typecode)

                # Make coef apply to vec(y*x.T) instead of vec(x*y.T).
                coef = coef * K

                # Swap x and y.
                x, y = y, x

            assert isinstance(x, BaseVariable)
            assert isinstance(y, Perturbation)

            coefs[x, y] = coef

        return coefs

    # --------------------------------------------------------------------------
    # Expression-creating operators.
    # --------------------------------------------------------------------------

    @cached_unary_operator
    def __abs__(self):
        """Denote the Euclidean or Frobenius norm of the expression."""
        from .uexp_norm import UncertainNorm

        return UncertainNorm(self)

    # --------------------------------------------------------------------------
    # Constraint-creating operators and _predict.
    # --------------------------------------------------------------------------

    @classmethod
    def _predict(cls, subtype, relation, other):
        from ..cone_nno import NonnegativeOrthant
        from ..cone_psd import PositiveSemidefiniteCone
        from ..set import Set
        from .pert_conic import ConicPerturbationSet
        from .pert_scenario import ScenarioPerturbationSet

        assert isinstance(subtype, cls.Subtype)

        universe = subtype.universe_type

        if relation in (operator.__le__, operator.__ge__):
            if issubclass(universe.clstype, ConicPerturbationSet):
                if issubclass(other.clstype,
                    (AffineExpression, UncertainAffineExpression)) \
                and other.subtype.shape == subtype.shape:
                    return ConicallyUncertainAffineConstraint.make_type(
                        dim=subtype.dim,
                        universe_subtype=universe.subtype)
            elif issubclass(universe.clstype, ScenarioPerturbationSet):
                if issubclass(other.clstype,
                    (AffineExpression, UncertainAffineExpression)) \
                and other.subtype.shape == subtype.shape:
                    return ScenarioUncertainConicConstraint.make_type(
                        dim=subtype.dim,
                        scenario_count=universe.subtype.scenario_count,
                        cone_type=NonnegativeOrthant.make_type(subtype.dim))
        elif relation in (operator.__lshift__, operator.__rshift__):
            if relation == operator.__lshift__ \
            and issubclass(other.clstype, Set):
                own_type = ExpressionType(cls, subtype)
                return other.predict(operator.__rshift__, own_type)

            if issubclass(other.clstype,
                (AffineExpression, UncertainAffineExpression)) \
            and other.subtype.shape == subtype.shape:
                if subtype.shape[0] != subtype.shape[1]:
                    return NotImplemented

                self = ExpressionType(cls, subtype)

                return self.predict(operator.__lshift__,
                    PositiveSemidefiniteCone.make_type(dim=None))

        return NotImplemented

    @convert_operands(sameShape=True)
    @validate_prediction
    @refine_operands()
    def __le__(self, other):
        from ..cone_nno import NonnegativeOrthant
        from .pert_conic import ConicPerturbationSet
        from .pert_scenario import ScenarioPerturbationSet

        if isinstance(self.perturbation.universe, ConicPerturbationSet):
            if isinstance(other, (AffineExpression, UncertainAffineExpression)):
                return ConicallyUncertainAffineConstraint(self - other)
        elif isinstance(self.perturbation.universe, ScenarioPerturbationSet):
            if isinstance(other, (AffineExpression, UncertainAffineExpression)):
                return ScenarioUncertainConicConstraint(
                    other - self, NonnegativeOrthant(len(self)))
        else:
            raise NotImplementedError("Uncertain affine constraints "
                "parameterized by {} are not supported.".format(
                self.perturbation.universe.__class__.__name__))

        return NotImplemented

    @convert_operands(sameShape=True)
    @validate_prediction
    @refine_operands()
    def __ge__(self, other):
        from ..cone_nno import NonnegativeOrthant
        from .pert_conic import ConicPerturbationSet
        from .pert_scenario import ScenarioPerturbationSet

        if isinstance(self.perturbation.universe, ConicPerturbationSet):
            if isinstance(other, (AffineExpression, UncertainAffineExpression)):
                return ConicallyUncertainAffineConstraint(other - self)
        elif isinstance(self.perturbation.universe, ScenarioPerturbationSet):
            if isinstance(other, (AffineExpression, UncertainAffineExpression)):
                return ScenarioUncertainConicConstraint(
                    self - other, NonnegativeOrthant(len(self)))
        else:
            raise NotImplementedError("Uncertain affine constraints "
                "parameterized by {} are not supported.".format(
                self.perturbation.universe.__class__.__name__))

        return NotImplemented

    @staticmethod
    def _lmi_helper(lower, greater):
        from ..cone_psd import PositiveSemidefiniteCone

        if isinstance(lower, UncertainAffineExpression) \
        and isinstance(greater, UncertainAffineExpression) \
        and lower.perturbation is not greater.perturbation:
            # NOTE: This failure cannot be predicted.
            raise ValueError("Can only form a linear matrix inequality if one "
                "side is certain or both sides depend on the same uncertainty.")

        diff = greater - lower

        if not diff.square:
            raise TypeError("Can only form a linear matrix inequality from "
                "square matrices.")

        if not diff.hermitian:
            # NOTE: This failure cannot be predicted.
            raise TypeError("Can only form a linear matrix inequality from "
                "hermitian matrices.")

        return diff << PositiveSemidefiniteCone()

    def _lshift_implementation(self, other):
        if isinstance(other, (AffineExpression, UncertainAffineExpression)):
            return self._lmi_helper(self, other)

        return NotImplemented

    def _rshift_implementation(self, other):
        if isinstance(other, (AffineExpression, UncertainAffineExpression)):
            return self._lmi_helper(other, self)

        return NotImplemented


# --------------------------------------
__all__ = api_end(_API_START, globals())
