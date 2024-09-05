# ------------------------------------------------------------------------------
# Copyright (C) 2019 Maximilian Stahlberg
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

"""Functions to quickly solve a problem."""

from . import Problem
from ..apidoc import api_end, api_start

_API_START = api_start(globals())
# -------------------------------


def _make_doc(minimize):
    """Create the docstring for :func:`minimize` and :func:`maximize`."""
    lower = "minimize" if minimize else "maximize"
    capital = lower.capitalize()
    sign = "" if minimize else "-"

    return \
    """{capital} a scalar expression subject to constraints.

    Internally, this creates a :class:`~.problem.Problem`, :meth:`sets an
    objective <.problem.Problem.set_objective>`, :meth:`adds constraints
    <.problem.Problem.add_list_of_constraints>`, performs a :meth:`solution
    search <.problem.Problem.solve>` and returns an optimum value found.

    :param ~picos.expressions.Expression function:
        The objective function to {lower}.

    :param list(~picos.constraints.Constraint) subject_to:
        A collection of constraints to obey.

    :param options:
        A keyword argument sequence of solver options to use. See
        :class:`~picos.Options`.

    :returns:
        The optimum value, as computed from an applied solution.

    :raises ~picos.SolutionFailure:
        See :meth:`~.problem.Problem.solve`.

    :Example:

    >>> from picos import {lower}, RealVariable
    >>> x = RealVariable("x")
    >>> p = {lower}({sign}x**2, [(x - 2)**2 <= x - 2], solver="cvxopt")
    >>> round(p, 5)
    {sign}4.0
    >>> round(x, 5)
    2.0
    """.format(capital=capital, lower=lower, sign=sign)


def minimize(function, subject_to = [], **options):  # noqa
    P = Problem(**options)
    P.set_objective("min", function)
    P.add_list_of_constraints(subject_to)
    P.solve()
    return P.value


def maximize(function, subject_to = [], **options):  # noqa
    P = Problem(**options)
    P.set_objective("max", function)
    P.add_list_of_constraints(subject_to)
    P.solve()
    return P.value


minimize.__doc__ = _make_doc(minimize=True)
maximize.__doc__ = _make_doc(minimize=False)


def find_assignment(subject_to=[], **options):
    """Find a feasible variable assignment.

    Internally, this creates a :class:`~.problem.Problem`, :meth:`adds
    constraints <.problem.Problem.add_list_of_constraints>` and performs a
    :meth:`solution search <.problem.Problem.solve>`.

    :param list(~picos.constraints.Constraint) subject_to:
        A collection of constraints to obey.

    :param options:
        A keyword argument sequence of solver options to use. See
        :class:`~picos.Options`.

    :returns:
        Nothing. Check the concerned variables' :attr:`values
        <.valuable.Valuable.value>`.

    :raises ~picos.SolutionFailure:
        See :meth:`~.problem.Problem.solve`.

    :Example:

    >>> from picos import find_assignment, RealVariable
    >>> x = RealVariable("x")
    >>> values = find_assignment([x**2 + 1 <= x], solver="cvxopt")
    ... # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    picos.modeling.problem.SolutionFailure: Code 3: ...
    >>> x.value
    >>> find_assignment([x**2 + 0.25 <= x], solver="cvxopt")
    >>> round(x.value, 5)
    0.5
    """
    P = Problem(**options)
    P.add_list_of_constraints(subject_to)
    P.solve()


# --------------------------------------
__all__ = api_end(_API_START, globals())
