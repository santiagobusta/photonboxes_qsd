# ------------------------------------------------------------------------------
# Copyright (C) 2019 Maximilian Stahlberg
# Based on the original picos.expressions module by Guillaume Sagnol.
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

"""Mathematical expression types."""

from ..apidoc import api_end, api_start

_API_START = api_start(globals())
# -------------------------------


from .algebra import *  # noqa
from .cone import Cone  # noqa
from .cone_expcone import ExponentialCone  # noqa
from .cone_nno import NonnegativeOrthant  # noqa
from .cone_product import ProductCone  # noqa
from .cone_psd import PositiveSemidefiniteCone  # noqa
from .cone_rsoc import RotatedSecondOrderCone  # noqa
from .cone_soc import SecondOrderCone  # noqa
from .cone_trivial import ZeroSpace, TheField  # noqa
from .exp_affine import (AffineExpression, ComplexAffineExpression,  # noqa
                         Constant)
from .exp_biaffine import BiaffineExpression  # noqa
from .exp_detrootn import DetRootN  # noqa
from .exp_entropy import Entropy, NegativeEntropy  # noqa
from .exp_extremum import Extremum, MaximumConvex, MinimumConcave  # noqa
from .exp_geomean import GeometricMean  # noqa
from .exp_logarithm import Logarithm  # noqa
from .exp_logsumexp import LogSumExp  # noqa
from .exp_norm import Norm  # noqa
from .exp_nucnorm import NuclearNorm  # noqa
from .exp_powtrace import PowerTrace  # noqa
from .exp_quadratic import QuadraticExpression  # noqa
from .exp_specnorm import SpectralNorm  # noqa
from .exp_sqnorm import SquaredNorm  # noqa
from .exp_sumexp import SumExponentials  # noqa
from .exp_sumxtr import SumExtremes  # noqa
from .exp_wsum import WeightedSum  # noqa
from .expression import (Expression, ExpressionType, no_refinement,  # noqa
                         NotValued, PredictedFailure)
from .mutable import Mutable  # noqa
from .variables import (BaseVariable, BinaryVariable, ComplexVariable,  # noqa
                        HermitianVariable, IntegerVariable,
                        LowerTriangularVariable, RealVariable,
                        SkewSymmetricVariable, SymmetricVariable,
                        UpperTriangularVariable, CONTINUOUS_VARTYPES)
from .samples import Samples  # noqa
from .set import Set, SetType  # noqa
from .set_ball import Ball  # noqa
from .set_ellipsoid import Ellipsoid  # noqa
from .set_simplex import Simplex  # noqa
from .uncertain import *  #noqa


# --------------------------------------
__all__ = api_end(_API_START, globals())
