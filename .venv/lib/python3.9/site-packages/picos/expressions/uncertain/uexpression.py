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

"""Implements the :class:`UncertainExpression` base class."""

import cvxopt

from ... import glyphs
from ...apidoc import api_end, api_start
from ...caching import cached_property
from ..expression import NotValued

_API_START = api_start(globals())
# -------------------------------


class IntractableWorstCase(RuntimeError):
    """Computing a worst-case (expected) value is hard and not supported.

    Raised by :meth:`~.uexpression.UncertainExpression.worst_case` and methods
    that depend on it.
    """


class UncertainExpression:
    """Primary base class for uncertainty affected expression types.

    The secondary base class must be :class:`~.expression.Expression` or a
    subclass thereof.

    Uncertain expressions have a distinct behavior when used to form a
    constraint or when posed as an objective function. The exact behavior
    depends on the type of uncertainty involved. If the perturbation parameter
    that describes the uncertainty is confied to a perturbation set, then the
    worst-case realization of the parameter is assumed when determining
    feasibility and optimality. If the perturbation parameter is a random
    variable (whose distribution may itself be ambiguous), then the constraint
    or objective implicitly considers the expected value of the uncertain
    expression (under the worst-case distribution). Uncertain expressions are
    thus used in the contexts of robust optimization, stochastic programming and
    distributionally robust optimization.
    """

    @cached_property
    def perturbation(self):
        """The parameter controlling the uncertainty, or :obj:`None`."""
        from .perturbation import Perturbation

        perturbations = tuple(
            prm for prm in self.parameters if isinstance(prm, Perturbation))

        if len(perturbations) > 1:
            raise NotImplementedError("Uncertain expressions may depend "
                "on at most one perturbation parameter. Found {}."
                .format(" and ".join(prt.name for prt in perturbations)))

        return perturbations[0] if perturbations else None

    @property
    def random(self):
        """Whether the uncertainty is of stochastic nature.

        See also :attr:`~.perturbation.PerturbationUniverse.distributional`.
        """
        if self.certain:
            return False
        else:
            return self.universe.distributional

    @cached_property
    def universe(self):
        """Universe that the perturbation parameter lives in, or :obj:`None`.

        If this is not :obj:`None`, then this is the same as
        :attr:`perturbation`.:attr:`~.perturbation.Perturbation.universe`.
        """
        return self.perturbation.universe if self.perturbation else None

    @property
    def certain(self):
        """Whether the uncertain expression is actually certain."""
        return not self.perturbation

    @property
    def uncertain(self):
        """Whether the uncertain expression is in fact uncertain."""
        return bool(self.perturbation)

    def worst_case(self, direction):
        """Find a worst-case realization of the uncertainty for the expression.

        Expressions that are affected by uncertainty are only partially valued
        once an optimization solution has been applied. While their decision
        values are populated with a robust optimal solution, the parameter that
        controls the uncertainty is not valued unless the user assigned it a
        particular realization by hand. This method computes a worst-case
        (expected) value of the expression and returns it together with a
        realization of the perturbation parameter for which the worst case is
        attained (or :obj:`None` in the case of stochastic uncertainty).

        For multidimensional expressions, this method computes the entrywise
        worst case and returns an attaining realization for each entry.

        :param str direction:
            Either ``"min"`` or ``"max"``, denoting the worst-case direction.

        :returns:
            A pair ``(value, realization)``. For a scalar expression, ``value``
            is its worst-case (expected) value as a :obj:`float` and
            ``realization`` is a realization of the :attr:`perturbation`
            parameter that attains this worst case as a :obj:`float` or CVXOPT
            matrix. For a multidimensional expression, ``value`` is a CVXOPT
            dense matrix denoting the entrywise worst-case values and
            ``realization`` is a :obj:`tuple` of attaining realizations
            corresponding to the expression vectorized in in column-major order.
            Lastly, ``realization`` is :obj:`None` if the expression is
            :attr:`certain` or when its uncertainty is of stochastic nature.

        :raises picos.NotValued:
            When the decision variables that occur in the expression are not
            fully valued.

        :raises picos.uncertain.IntractableWorstCase:
            When computing the worst-case (expected) value is not supported, in
            particular when it would require solving a nonconvex problem.

        :raises RuntimeError:
            When the computation is supported but fails.
        """
        if not all(var.valued for var in self.variables):
            raise NotValued("Not all decision variables that occur in the "
                "uncertain expression {} are valued, so PICOS cannot compute "
                "its worst-case (expected) value.".format(self.string))

        if self.certain:
            return self.safe_value, None

        outcome = self.frozen(self.variables)
        assert outcome.mutables == set([self.perturbation])

        if self.scalar:
            return self.universe.worst_case(outcome, direction)
        else:
            values, realizations = zip(*(
                self.universe.worst_case(outcome[i], direction)
                for i in range(len(self))))

            return cvxopt.matrix(values, self.shape), realizations

    def worst_case_value(self, direction):
        """A shorthand for the first value returned by :meth:`worst_case`."""
        return self.worst_case(direction)[0]

    def worst_case_string(self, direction):
        """A string describing the expression within a worst-case context.

        :param str direction:
            Either ``"min"`` or ``"max"``, denoting the worst-case direction.
        """
        # NOTE: The following distinguishes only RO and DRO and needs to be
        #       extended when SP models are supported.
        if self.random:
            over = glyphs.probdist(self.perturbation.string)
            base = glyphs.exparg(self.perturbation.string, self.string)
        else:
            over = self.perturbation.string
            base = self.string

        if direction == "min":
            return glyphs.minarg(over, base)
        elif direction == "max":
            return glyphs.maxarg(over, base)
        else:
            raise ValueError("Invalid direction.")


# --------------------------------------
__all__ = api_end(_API_START, globals())
