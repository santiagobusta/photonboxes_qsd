# ------------------------------------------------------------------------------
# Copyright (C) 2012-2017 Guillaume Sagnol
# Copyright (C) 2018-2019 Maximilian Stahlberg
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

"""A Python Interface to Conic Optimization Solvers.

The :mod:`picos` namespace gives you quick access to the most important classes
and functions for optimizing with PICOS, so that ``import picos`` is often
sufficient for implementing your model.
"""

from pathlib import Path

from .apidoc import api_start, api_end


with (Path(__file__).parent / ".version").open() as versionFile:
    __version_info__ = tuple(
        int(x) for x in versionFile.read().strip().split("."))

__version__ = ".".join(str(x) for x in __version_info__)


_API_START = api_start(globals())
# -------------------------------

# Namespaces.
from . import settings, uncertain  # noqa

# Character set changes.
from .glyphs import ascii, latin1, unicode, default as default_charset  # noqa

# Model setup.
from .modeling import (find_assignment, maximize, minimize, Objective,  # noqa
                       Options, Problem, Solution)

# Constants.
from .expressions import Constant  # noqa

# Variables.
from .expressions.variables import (  # noqa
    BinaryVariable, ComplexVariable, HermitianVariable, IntegerVariable,
    LowerTriangularVariable, RealVariable, SkewSymmetricVariable,
    SymmetricVariable, UpperTriangularVariable)

# Cones.
from .expressions import (  # noqa
    ExponentialCone, NonnegativeOrthant, PositiveSemidefiniteCone, ProductCone,
    RotatedSecondOrderCone, SecondOrderCone, ZeroSpace, TheField)

# Other sets.
from .expressions import Ball, Ellipsoid, Simplex  # noqa

# Algebraic function-like classes.
from .constraints import FlowConstraint  # noqa
from .expressions import (  # noqa
    DetRootN, Entropy, GeometricMean, Logarithm, LogSumExp,
    NegativeEntropy, Norm, SpectralNorm, SquaredNorm, NuclearNorm, PowerTrace,
    SumExtremes, SumExponentials)

# Algebraic functions, including legacy ones.
from .expressions.algebra import *  # noqa

# Utilities.
from .expressions.data import value  # noqa
from .expressions.samples import Samples  # noqa
from .solvers import available_solvers  # noqa

# Non-algebraic legacy imports.
from .modeling.file_in import import_cbf  # noqa

# Exceptions.
from .modeling import SolutionFailure  # noqa
from .expressions import NotValued  # noqa

# Allow users to work around https://github.com/scipy/scipy/issues/4819.
from .valuable import patch_scipy_array_priority  # noqa

# --------------------------------------
__all__ = api_end(_API_START, globals())
