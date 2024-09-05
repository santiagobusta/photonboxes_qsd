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

"""Backwards compatibility version of an older module."""

from .apidoc import api_end, api_start
from .expressions.data import load_data
from .legacy import deprecated, throw_deprecation_warning

# Deprecated as of 2.0.
throw_deprecation_warning("The 'picos.tools' module is deprecated and will be "
    "removed in a future release. Try importing from 'picos' or 'picos.algebra'"
    "instead.")

_API_START = api_start(globals())
# -------------------------------


from builtins import sum as builtin_sum  # noqa

from . import (  # noqa
    sum, geomean, norm, tracepow, trace, sum_k_largest, sum_k_largest_lambda,
    lambda_max, sum_k_smallest, sum_k_smallest_lambda, lambda_min,
    partial_transpose, partial_trace, detrootn, ball, simplex,
    truncated_simplex, expcone, exp, log, sumexp, kullback_leibler, logsumexp,
    lse, diag, diag_vect, new_param, import_cbf, kron, flow_Constraint
)

from .formatting import (  # noqa
    detect_range, parameterized_string
)


@deprecated("2.0", reason="Write '{k: v.value for k, v in x.items()}' instead.")
def eval_dict(dict_of_variables):
    """Evaluate all values of a dictionary.

    :param dict dict_of_variables:
        A dictionary mapping arbitrary keys to PICOS objects that have a
        ``value`` attribute, such as variables.

    :returns:
        Another dictionary with the original dicitonary's values replaced by the
        value of their ``value`` attribute.
    """
    return {k: v.value for k, v in dict_of_variables.items()}


@deprecated("2.0", useInstead="picos.expressions.data.load_data")
def retrieve_matrix(mat, exSize=None):
    """Legacy wrapper around :func:`~.expressions.data.load_data`."""
    return load_data(value=mat, shape=exSize, sparse=True, legacy=True)


# --------------------------------------
__all__ = api_end(_API_START, globals())
