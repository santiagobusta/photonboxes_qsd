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

"""Contains global settings for PICOS.

All settings can be set via environment variables using the ``PICOS_`` prefix,
e.g. ``PICOS_SOLVER_WHITELIST='["cvxopt", "glpk"]' ./application.py`` would set
:data:`SOLVER_WHITELIST` to ``["cvxopt", "glpk"]`` for this execution of
``application.py`` only. :func:`Safe evaluation <ast.literal_eval>` is used to
convert the given value to a Python object.

Applications that use PICOS may assign to these settings directly (silently
overwriting any environment variable) but libraries that depend on PICOS should
not do so as this would affect also applications and other libraries that use
both PICOS and the library making the modificaiton.
"""

import os
import sys
from ast import literal_eval

from .apidoc import api_end, api_start

_API_START = api_start(globals())
# -------------------------------


SOLVER_WHITELIST = []
"""A list of names of solvers; PICOS considers all others not available."""

SOLVER_BLACKLIST = []
"""A list of names of solvers that PICOS considers to be not available."""

NONFREE_SOLVERS = True
"""Whether PICOS may perform solution search with proprietary solvers."""

LICENSE_WARNINGS = True
"""Let solvers ignore verbosity settings to print licensing related warnings.

License warnings are only printed if both :data:`LICENSE_WARNINGS` and the
solution search option :ref:`license_warnings <option_license_warnings>` are set
to true, or if the verbosity setting allows solver output in general.
"""

RETURN_SOLUTION = True
"""Whether :meth:`~.Problem.solve` returns a :class:`~.solution.Solution`."""

INTERNAL_OPTIONS = {"solver": "cvxopt"}
"""Solution search options used whenever PICOS solves a problem internally.

By default, this limits the solver used to CVXOPT for reproducibility and to
avoid licensing issues when non-free solvers are installed.

This setting is given as a dictionary. For keys and possible values see
:class:`~picos.Options`.
"""

DEFAULT_CHARSET = "unicode"
"""Default charset to use for :mod:`console output <picos.glyphs>`.

Can be any of ``"ascii"``, ``"latin1"`` or ``"unicode"`` (default).

Note that applications can change the charset at any time using the respective
function in the :mod:`~picos.glyphs` module.
"""

RELATIVE_HERMITIANNESS_TOLERANCE = 1e-10
r"""Relative tolerance used when checking whether a matrix is hermitian.

A matrix :math:`A \in \mathbb{C}^{n \times n}` is considered numerically
hermitian if

.. math::

    \max_{1 \leq i, j \leq n} |(A - A^H)_{ij}|
    \leq
    \varepsilon \max_{1 \leq i, j \leq n} |A_{ij}|

where :math:`\varepsilon` is this tolerance.
"""

RELATIVE_SEMIDEFINITENESS_TOLERANCE = 1e-10
r"""Relative tolerance used when checking if a matrix is positive semidefinite.

A hermitian matrix :math:`A \in \mathbb{C}^{n \times n}` is considered
numerically positive semidefinite if

.. math::

    \min \operatorname{eigvals}(A)
    \geq
    -\varepsilon \max \left( \{ 1 \} \cup \operatorname{eigvals}(A) \right)

where :math:`\varepsilon` is this tolerance.
"""

ABSOLUTE_INTEGRALITY_TOLERANCE = 1e-4
"""Absolute tolerance used to validate integrality of integral variables."""

WARN_MISSING_DOCSTRINGS = False
"""Whether to warn about missing docstrings for top level objects in a module.

Must be set via an environment variable to have an effect.
"""

PREFER_GUROBI_MATRIX_INTERFACE = True
"""Whether to prefer Gurobi's matrix interface when it is available.

This default can be overwritten by the :ref:`gurobi_matint
<option_gurobi_matint>` solver option.
"""

UNRELIABLE_STRATEGIES = False
"""Whether to pass solvers problems that they are known to struggle with.

This allows all problem/solver combinations that are skipped by "PICOS' choice".
"""


# --------------------------------------
__all__ = api_end(_API_START, globals())


_THIS = sys.modules[__name__]
_ENV_PREFIX = "PICOS_"


# Modify settings given by the environment.
for key, value in os.environ.items():
    key = key.upper()

    if not key.startswith(_ENV_PREFIX):
        continue

    setting = key.split(_ENV_PREFIX, 1)[1]

    if setting not in __all__:
        raise AttributeError(
            "The setting '{}' referenced by the environment variable '{}' is "
            "not known to PICOS.".format(setting, key))

    try:
        value = literal_eval(value)
    except Exception as error:
        raise ValueError(
            "The value '{}' of the environment variable '{}' could not be "
            "parsed as a Python literal.".format(value, key)) from error

    setattr(_THIS, setting, value)
