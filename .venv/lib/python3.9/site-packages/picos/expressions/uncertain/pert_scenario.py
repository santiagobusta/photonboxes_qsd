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

"""Implements :class:`ScenarioPerturbationSet`."""

from collections import namedtuple

import cvxopt

from ... import glyphs, settings
from ...apidoc import api_end, api_start
from ..data import cvx2np
from ..expression import NotValued
from ..samples import Samples
from ..variables import RealVariable
from .perturbation import Perturbation, PerturbationUniverse
from .uexpression import IntractableWorstCase

_API_START = api_start(globals())
# -------------------------------


class ScenarioPerturbationSet(PerturbationUniverse):
    r"""A scenario description of a :class:`~.perturbation.Perturbation`.

    :Definition:

    An instance :math:`\Theta` of this class defines a perturbation parameter

    .. math::

        \theta \in \Theta = \operatorname{conv}(S)

    where :math:`S \subset \mathbb{R}^{m \times n}` is a finite set of
    *scenarios* and :math:`\operatorname{conv}` denotes the convex hull.

    Usually, the scenarios are observed or projected realizations of the
    uncertain data and :math:`\theta` is used to represent the data directly.

    :Example:

    >>> from picos.uncertain import ScenarioPerturbationSet
    >>> scenarios = [[1, -1], [1, 1], [-1, -1], [-1, 1], [0, 0]]
    >>> S = ScenarioPerturbationSet("s", scenarios, False); S
    <2×1 Scenario Perturbation Set: conv({5 2×1 scenarios})>
    >>> s = S.parameter; s
    <2×1 Perturbation: s>
    >>> # Compute largest sum of entries over all points in S.
    >>> value, realization = S.worst_case(s[0] + s[1], "max")
    >>> round(value,  4)
    2.0
    >>> print(realization)
    [ 1.00e+00]
    [ 1.00e+00]
    <BLANKLINE>
    """

    def __init__(self, parameter_name, scenarios, compute_hull=None):
        """Create a :class:`ScenarioPerturbationSet`.

        :param str parameter_name:
            Name of the parameter that lives in the set.

        :param scenarios:
            A collection of data points of same shape representing :math:`S`.
        :type scenarios:
            anything recognized by :class:`picos.Samples`

        :param bool compute_hull:
            Whether to use SciPy to compute the convex hull of the data points
            and discard points in the interior. This can speed up the solution
            process significantly, in particular when the scenarios come from
            observations and when the data is low-dimensional. On the other
            hand, when the given scenarios are known to be on the boundary of
            their convex hull, then disabling this speeds up initialization of
            the perturbation set. The default value of :obj:`None` means
            :obj:`True` when SciPy is available and :obj:`False` otherwise.
        """
        S = Samples(scenarios)

        if compute_hull is not False:
            if S.dim == 1:
                # scipy.spatial.ConvexHull does not work on scalars.
                S = Samples([min(S._cvxopt_matrix), max(S._cvxopt_matrix)])
            else:
                try:
                    from scipy.spatial import qhull
                except ModuleNotFoundError as error:
                    if compute_hull:
                        raise RuntimeError("PICOS requires SciPy to compute a "
                            "convex hull.") from error
                else:
                    try:
                        hull = qhull.ConvexHull(cvx2np(S._cvxopt_matrix.T))
                    except qhull.QhullError as error:
                        if compute_hull:
                            raise RuntimeError("SciPy failed to compute a "
                                "convex hull.") from error
                    else:
                        S = S.select(hull.vertices.tolist())

        # Define a convex combination of the scenarios for anticipation.
        t = RealVariable("t", S.num, lower=0)
        A = (S.matrix*t).reshaped(S.original_shape)

        self._scenarios = S
        self._convex_combination = t, A
        self._parameter = Perturbation(self, parameter_name, S.original_shape)

    @property
    def scenarios(self):
        """The registered scenarios as a :class:`~.samples.Samples` object."""
        return self._scenarios

    Subtype = namedtuple("Subtype", ("param_dim", "scenario_count"))

    def _subtype(self):
        return self.Subtype(self._parameter.dim, self._scenarios.num)

    def __str__(self):
        return "conv({})".format(glyphs.set("{} {} scenarios".format(
            self._scenarios.num, glyphs.shape(self.parameter.shape))))

    @classmethod
    def _get_type_string_base(cls):
        return "Scenario Perturbation Set"

    def __repr__(self):
        return glyphs.repr2("{} {}".format(glyphs.shape(self._parameter.shape),
            self._get_type_string_base()), self.__str__())

    @property
    def distributional(self):
        """Implement for :class:`~.perturbation.PerturbationUniverse`."""
        return False

    @property
    def parameter(self):
        """Implement for :class:`~.perturbation.PerturbationUniverse`."""
        return self._parameter

    def worst_case(self, scalar, direction):
        """Implement for :class:`~.perturbation.PerturbationUniverse`."""
        from ...modeling import Problem, SolutionFailure

        self._check_worst_case_argument_scalar(scalar)
        self._check_worst_case_argument_direction(direction)

        p = self._parameter
        x = RealVariable(p.name, p.shape)
        f = scalar.replace_mutables({p: x})

        self._check_worst_case_f_and_x(f, x)

        if direction == "find" \
        or (f.convex and direction == "min") \
        or (f.concave and direction == "max"):
            # Use convex optimization.
            t, A = self._convex_combination

            P = Problem()
            P.set_objective(direction, f)
            P.add_constraint(x == A)
            P.add_constraint((t | 1) == 1)

            try:
                P.solve(**settings.INTERNAL_OPTIONS)
                return f.safe_value, x.safe_value
            except (SolutionFailure, NotValued) as error:
                raise RuntimeError(
                    "Failed to compute {}({}) for {}.".format(direction,
                    f.string, glyphs.element(x.string, self))) from error
        elif (f.convex and direction == "max") \
        or (f.concave and direction == "min"):
            # Use enumeration since at least one scenario is optimal.
            minimizing = direction == "min"

            value = float("inf") if minimizing else float("-inf")
            realization = None

            for scenario in self._scenarios._cvxopt_vectors:
                x.value = scenario
                f_value = f.value

                if (minimizing and f_value < value) \
                or (not minimizing and f_value > value):
                    value = f_value
                    realization = scenario

            assert realization is not None

            if realization.size == (1, 1):
                realization = realization[0]
            else:
                realization = cvxopt.matrix(realization, p.shape)

            return value, realization
        else:
            assert not f.convex and not f.concave

            raise IntractableWorstCase("PICOS does not know how to compute "
                "{}({}) for {} as the objective function seems to be neither "
                "convex nor concave.".format(direction, scalar.string,
                glyphs.element(p.name, self)))


# --------------------------------------
__all__ = api_end(_API_START, globals())
