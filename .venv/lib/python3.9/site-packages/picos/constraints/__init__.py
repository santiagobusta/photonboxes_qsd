# ------------------------------------------------------------------------------
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

"""Optimization constraint types.

This package contains the constraint types that are used to express optimization
constraints. You do not need to instanciate these constraints directly; it is
more convenient to create them by applying Python's comparison operators to
algebraic expressions (see the :ref:`cheatsheet`).
"""

from ..apidoc import api_end, api_start


_API_START = api_start(globals())
# -------------------------------


# Import the base classes.
from .constraint import (ConicConstraint, ConstraintType, Constraint,  # noqa
                         ConstraintConversion)

# Import the implementation classes.
from .con_absolute  import AbsoluteValueConstraint  # noqa
from .con_affine    import ComplexAffineConstraint, AffineConstraint  # noqa
from .con_detrootn  import DetRootNConstraint  # noqa
from .con_expcone   import ExpConeConstraint  # noqa
from .con_extremum  import ExtremumConstraint  # noqa
from .con_flow      import FlowConstraint  # noqa
from .con_dummy     import DummyConstraint  # noqa
from .con_geomean   import GeometricMeanConstraint  # noqa
from .con_kldiv     import KullbackLeiblerConstraint  # noqa
from .con_lmi       import ComplexLMIConstraint, LMIConstraint  # noqa
from .con_log       import LogConstraint  # noqa
from .con_logsumexp import LogSumExpConstraint  # noqa
from .con_matnorm   import (MatrixNormConstraint, # noqa
                            SpectralNormConstraint, NuclearNormConstraint)
from .con_powtrace  import PowerTraceConstraint  # noqa
from .con_prodcone  import ProductConeConstraint  # noqa
from .con_quadratic import (NonconvexQuadraticConstraint, # noqa
                           ConvexQuadraticConstraint, ConicQuadraticConstraint)
from .con_rsoc      import RSOCConstraint  # noqa
from .con_simplex   import SimplexConstraint  # noqa
from .con_soc       import SOCConstraint  # noqa
from .con_sqnorm    import SquaredNormConstraint  # noqa
from .con_sumexp    import SumExponentialsConstraint  # noqa
from .con_sumxtr    import SumExtremesConstraint  # noqa
from .con_vecnorm   import VectorNormConstraint  # noqa
from .con_wsum      import WeightedSumConstraint  # noqa

# Import additional implementation classes from subpackages.
from .uncertain     import *  # noqa


# --------------------------------------
__all__ = api_end(_API_START, globals())
