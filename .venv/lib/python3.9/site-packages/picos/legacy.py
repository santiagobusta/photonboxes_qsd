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

"""Backwards-compatibility and deprecation helpers."""

import functools
import warnings

from .apidoc import api_end, api_start
from .formatting import doc_cat

_API_START = api_start(globals())
# -------------------------------


def throw_deprecation_warning(reason, decoratorLevel=0):
    """Raise a deprecation warning."""
    warnings.warn(reason, DeprecationWarning, stacklevel=3 + decoratorLevel)


def deprecated(since, reason=None, useInstead=None, decoratorLevel=0):
    """Decorate a deprecated function or method.

    :param str reason: Textual deprecation reason.
    :param str useInstead: Alternative to the deprecated object.
    :param int decoratorLevel: If not the only decorator, number of decorators
        above this one that add to the stack level of a warning thrown.
    """
    assert len([x for x in (reason, useInstead) if x is not None]) == 1, \
        "Need to supply exactly one reason or reference to @deprecated."

    if reason is not None:
        rest, text = reason, reason
    if useInstead is not None:
        rest = "Use :py:obj:`{0}` instead.".format(useInstead)
        text = "Use {} instead.".format(useInstead.split(".")[-1]
            if useInstead.startswith("~") else useInstead)

    def decorator(wrapped):
        # Make the wrapper copy the wrapped object's documentation …
        @functools.wraps(wrapped)
        def wrapper(*args, **kwargs):
            warnings.warn(
                "{} is deprecated: {}".format(wrapped.__qualname__, text),
                DeprecationWarning, stacklevel=2 + decoratorLevel)
            return wrapped(*args, **kwargs)

        # … and extend it.
        wrapper.__doc__ = doc_cat(
            wrapper.__doc__ if wrapper.__doc__ else "Deprecated name.",
            ".. deprecated:: {}\n    {}".format(since, rest))

        return wrapper
    return decorator


_option_name_old2new = {
    "allow_license_warnings": "license_warnings",
    "verbose":      "verbosity",
    "noprimals":    "primals",
    "noduals":      "duals",
    "tol":          ("*_fsb_tol", "*_ipm_opt_tol"),
    "gaplim":       "rel_bnb_opt_tol",
    "maxit":        "max_iterations",
    "nbsol":        "max_fsb_nodes",
    "pool_relgap":  "pool_rel_gap",
    "pool_absgap":  "pool_abs_gap",
    "lboundlimit":  "cplex_lwr_bnd_limit",
    "uboundlimit":  "cplex_upr_bnd_limit",
    "boundMonitor": "cplex_bnd_monitor",
    "solve_via_dual": "dualize"
}

_option_value_old2new = {
    "noprimals": lambda x: not x,
    "noduals": lambda x: not x,
    "solve_via_dual": lambda x: bool(x)  # Map None to False.
}


def map_legacy_options(options={}, **kwOptions):
    """Map deprecated solver options to those replacing them."""
    kwOptions.update(options)

    newOptions = {}
    for name, value in kwOptions.items():
        if name in _option_name_old2new:
            newNames = _option_name_old2new[name]

            if isinstance(newNames, str):
                newNames = (newNames,)

            warnings.warn(
                "Option '{}' is deprecated, use '{}'.".format(
                name, ", ".join(newNames)), DeprecationWarning, stacklevel=3)

            if name in _option_value_old2new:
                value = _option_value_old2new[name](value)

            for newName in newNames:
                newOptions[newName] = value
        else:
            newOptions[name] = value

    return newOptions


# --------------------------------------
__all__ = api_end(_API_START, globals())
