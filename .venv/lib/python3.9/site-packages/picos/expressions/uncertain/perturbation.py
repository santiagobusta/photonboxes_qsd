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

"""Implements a parameterization for (random) noise in data."""

from abc import ABC, abstractmethod

from ...apidoc import api_end, api_start
from ...containers import DetailedType
from ..data import load_shape
from ..mutable import Mutable
from ..vectorizations import FullVectorization
from .uexp_affine import UncertainAffineExpression
from .uexpression import IntractableWorstCase, UncertainExpression

_API_START = api_start(globals())
# -------------------------------


class PerturbationUniverseType(DetailedType):
    """Container for a pair of perturbation universe class type and subtype."""

    pass


class PerturbationUniverse(ABC):
    """Base class for uncertain perturbation sets and distributions.

    See :attr:`distributional` for a distinction between perturbation sets,
    random distributions and distributional ambiguity sets, all three of which
    can be represented by this class.

    The naming scheme for implementing classes is as follows:

    - Perturbation sets (robust optimization) end in ``PerturbationSet``,
    - random distributions (stochastic programming) end in ``Distribution``,
    - distributional ambiguity sets (DRO) end in ``AmbiguitySet``.
    """

    # --------------------------------------------------------------------------
    # Prediction related.
    # --------------------------------------------------------------------------

    @property
    def type(self):
        """Detailed type of a perturbation parameter universe."""
        return PerturbationUniverseType(self.__class__, self._subtype())

    subtype = property(lambda self: self._subtype())

    @classmethod
    def make_type(cls, *args, **kwargs):
        """Create a detailed universe type from subtype parameters."""
        return PerturbationUniverseType(cls, cls.Subtype(*args, **kwargs))

    @abstractmethod
    def _subtype(self):
        """Subtype of the perturbation parameter universe."""
        pass

    # --------------------------------------------------------------------------
    # Other.
    # --------------------------------------------------------------------------

    @property
    @abstractmethod
    def parameter(self):
        r"""The perturbation parameter."""
        pass

    @property
    @abstractmethod
    def distributional(self):
        r"""Whether this is a distribution or distributional ambiguity set.

        If this is :obj:`True`, then this represents a random distribution
        (stochastic programming) or an ambiguity set of random distributions
        (distributionally robust optimization) and any expression that depends
        on its random :attr:`parameter`, when used in a constraint or as an
        objective function, is understood as a (worst-case) *expected* value.

        If this is :obj:`False`, then this represents a perturbation set (robust
        optimization) and any expression that depends on its perturbation
        :attr:`parameter`, when used in a constraint or as an objective
        function, is understood as a worst-case value.
        """
        pass

    def _check_worst_case_argument_scalar(self, scalar):
        """Support implementations of :meth:`worst_case`."""
        if not isinstance(scalar, UncertainExpression):
            raise TypeError("{} can only compute the worst-case value of "
                "uncertain expressions, not of {}."
                .format(type(self).__name__, type(scalar).__name__))

        if not scalar.scalar:
            raise TypeError(
                "{} can only compute the worst-case value of a scalar "
                "expression.".format(type(self).__name__))

        p = self.parameter
        if scalar.mutables != set([p]):
            raise ValueError(
                "{} can only compute the worst-case value of expressions that "
                "depend exactly on its perturbation parameter {}.".format(
                type(self).__name__, p.name))

    def _check_worst_case_argument_direction(self, direction):
        """Support implementations of :meth:`worst_case`."""
        if not isinstance(direction, str):
            raise TypeError("Optimization direction must be given as a string.")

        # NOTE: "find" is OK even though it is not documented.
        if direction not in ("min", "max", "find"):
            raise ValueError(
                "Invalid optimization direction '{}'.".format(direction))

    def _check_worst_case_f_and_x(self, f, x):
        """Support implementations of :meth:`worst_case`.

        :param f:
            The certain scalar function to minimize or maximize.

        :param x:
            The decision variable that replaces the uncertain parameter in f.
        """
        assert f.scalar
        assert f.mutables == set([x])

        assert not isinstance(f, UncertainExpression), \
            "An instance of {} did not refine to a certain expression type " \
            "after its perturbation parameter was replaced with a real " \
            "variable.".format(type(f).__name__)

    def worst_case(self, scalar, direction):
        """Find a worst-case realization of the uncertainty for an expression.

        :param scalar:
            A scalar uncertain expression that depends only on the perturbation
            :attr:`parameter`.
        :type scalar:
            ~picos.expressions.uncertain.uexpression.UncertainExpression

        :param str direction:
            Either ``"min"`` or ``"max"``, denoting the worst-case direction.

        :returns:
            A pair where the first element is the worst-case (expeceted) value
            as a :obj:`float` and where the second element is a realization of
            the perturbation parameter that attains this worst case as a
            :obj:`float` or CVXOPT matrix (or :obj:`None` for stochastic
            uncertainty).

        :raises TypeError:
            When the function is not scalar.

        :raises ValueError:
            When the function depends on other mutables than exactly the
            :attr:`parameter`.

        :raises picos.uncertain.IntractableWorstCase:
            When computing the worst-case (expected) value is not supported, in
            particular when it would require solving a nonconvex problem.

        :raises RuntimeError:
            When the computation is supported but fails.
        """
        raise IntractableWorstCase("Computing a worst-case (expected) value is "
            "not supported for uncertainty defined through an instance of {}."
            .format(self.__class__.__name__))


class Perturbation(Mutable, UncertainAffineExpression):
    r"""A parameter that can be used to describe (random) noise in data.

    This is the initial building block for an
    :class:`~.uexp_affine.UncertainAffineExpression`. In particular, an affine
    transformation of this parameter represents uncertain data.
    """

    @classmethod
    def _get_type_string_base(cls):
        # TODO: Make type string depend on the perturbation set/distribution.
        # NOTE: It would probably be best to replace Expression._typeStr and
        #       _symbStr with abstract instance methods and implement them with
        #       the cached_property decorator.
        return "Perturbation"

    def __init__(self, universe, name, shape):
        """Create a :class:`~.perturbation.Perturbation`.

        :param universe:
            Either the set that the perturbation parameter lives in or the
            distribution according to which the perturbation is distributed.
        :type universe:
            ~picos.expressions.uncertain.perturbation.PerturbationUniverse

        :param str name:
            Symbolic string description of the perturbation, similar to a
            variable's name.

        :param shape:
            Algebraic shape of the perturbation parameter.
        :type shape:
            int or tuple or list

        This constructor is meant for internal use. As a user, you will want to
        first define a universe (e.g.
        :class:`~.pert_conic.ConicPerturbationSet`) for the parameter and obtain
        the parameter from it.
        """
        shape = load_shape(shape)
        vec = FullVectorization(shape)
        Mutable.__init__(self, name, vec)
        UncertainAffineExpression.__init__(
            self, self.name, shape, {self: vec.identity})

        assert isinstance(universe, PerturbationUniverse)

        self._universe = universe

    def copy(self, new_name=None):
        """Return an independent copy of the perturbation."""
        name = self.name if new_name is None else new_name

        return self.__class__(self._universe, name, self.shape)

    @property
    def universe(self):
        """The uncertainty universe that the parameter belongs to."""
        return self._universe


# --------------------------------------
__all__ = api_end(_API_START, globals())
