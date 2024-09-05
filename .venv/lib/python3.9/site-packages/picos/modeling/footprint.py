# ------------------------------------------------------------------------------
# Copyright (C) 2019-2020 Maximilian Stahlberg
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

"""Optimization problem description classes.

This module implements a footprint of a single problem, storing number and types
of expressions and constraints, and the specification of a problem class.
"""

import math
from inspect import isclass
from itertools import chain

from .. import constraints, expressions, glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from ..containers import RecordTree
from ..expressions import variables
# from .options import OPTION_OBJS, Options  # Circular import.

_API_START = api_start(globals())
# -------------------------------


# Constants used by both Footprint and Specification.
_OBJS = tuple(exp for exp in expressions.__dict__.values() if isclass(exp)
    and issubclass(exp, expressions.Expression)
    and exp != expressions.Expression)  # ALl proper Expression subclasses.
_DIRS = ("find", "min", "max")
_VARS = tuple(var for var in expressions.__dict__.values() if isclass(var)
    and issubclass(var, expressions.BaseVariable)
    and var != expressions.BaseVariable)  # ALl proper BaseVariable subclasses.
_CONS = tuple(con for con in constraints.__dict__.values() if isclass(con)
    and issubclass(con, constraints.Constraint)
    and con != constraints.Constraint)  # All proper Constraint subclasses.
# _OPTS = tuple(option.name for option in OPTION_OBJS)  # All option names.


class Footprint(RecordTree):
    """Statistics on an optimization problem."""

    _ADD_VALUES = [("var",), ("con",)]

    def __init__(self, recordsOrDict):
        """Construct a :class:`Footprint` from raw data.

        See :class:`picos.containers.RecordTree.__init__`. The ``addValues``
        argument is fixed; only variable and constraint paths are added.
        """
        super(Footprint, self).__init__(recordsOrDict, self._ADD_VALUES)

        # Sanity check records.
        for path, value in self.items:
            pathLen  = len(path)
            category = path[0] if pathLen > 0 else None
            suptype  = path[1] if pathLen > 1 else None
            subtype  = path[2] if pathLen > 2 else None

            if category == "dir":
                if pathLen == 2 and suptype in _DIRS and value is None:
                    continue
            elif category == "obj":
                if pathLen == 3 and suptype in _OBJS \
                and isinstance(subtype, suptype.Subtype) and value is None:
                    continue
            elif category == "var":
                if pathLen == 3 and suptype in _VARS \
                and isinstance(subtype, suptype.VarSubtype) \
                and isinstance(value, int):
                    continue
            elif category == "con":
                if pathLen == 3 and suptype in _CONS \
                and isinstance(subtype, suptype.Subtype) \
                and isinstance(value, int):
                    continue
            elif category == "opt":
                # if pathLen == 2 and suptype in _OPTS:
                if pathLen == 2 and isinstance(suptype, str):
                    continue

            raise ValueError("Invalid problem footprint record: {} = {}."
                .format(path, value))

        # Sanity check for inconsistent duplicates or missing fields.
        if len(self["dir"]) != 1:
            raise TypeError("Not exactly one optimization direction defined for"
                " a problem footprint (but {}).".format(len(self["dir"])))
        elif len(self["obj"]) != 1:
            raise TypeError("Not exactly one objective function defined for a "
                "problem footprint (but {}).".format(len(self["obj"])))

    def updated(self, recordsOrDict):
        """Override :class:`~picos.containers.RecordTree.updated`.

        This method, just like :meth:`__init__`, does not take the additional
        ``addValues`` argument.
        """
        return self.__class__(
            chain(self.records, self._record_iterator(recordsOrDict)))

    @property
    def direction(self):
        """Objective function optimization direction."""
        return next(self["dir"].paths)[0]

    @property
    def objective(self):
        """Detailed type of the objective function."""
        clstype, subtype = next(self["obj"].paths)
        return expressions.ExpressionType(clstype, subtype)

    @cached_property
    def variables(self):
        """A dictionary mapping detailed variable types to their quantity."""
        return {variables.VariableType(*ts): n
            for ts, n in self.get("var").items}

    @cached_property
    def constraints(self):
        """A dictionary mapping detailed constraint types to their quantity."""
        return {constraints.ConstraintType(*ts): n
            for ts, n in self.get("con").items}

    @cached_property
    def nondefault_options(self):
        """A dictionary mapping option names to their nondefault values.

        .. warning::

            This property is cached for performance reasons, do not modify any
            mutable option value (make a deep copy instead)!
        """
        return {path[0]: value for path, value in self.get("opt").items}

    @cached_property
    def options(self):
        """An :class:`~.options.Options` object.

        .. warning::

            This property is cached for performance reasons, do not modify the
            returned object (make a :meth:`~.options.Options.copy` instead)!
        """
        from .options import Options
        return Options(**self.nondefault_options)

    @cached_property
    def integer(self):
        """Whether an integral variable type is present."""
        return any(("var", vtype) in self for vtype in (
            expressions.BinaryVariable, expressions.IntegerVariable))

    @property
    def continuous(self):
        """Whether no integral variable type is present."""
        return not self.integer

    @cached_property
    def nonconvex_quadratic_objective(self):
        """Whether the problem has a nonconvex quadratic objective."""
        if issubclass(self.objective.clstype, expressions.QuadraticExpression):
            direction = self.direction
            subtype   = self.objective.subtype

            assert direction in ("min", "max")

            if self.objective.clstype is expressions.SquaredNorm:
                if direction == "max":
                    return True
            else:
                if direction == "min" and not subtype.convex:
                    return True
                elif direction == "max" and not subtype.concave:
                    return True

        return False

    def __str__(self):
        dirStr = self.direction.capitalize()
        objStr = str(self.objective)
        varStr = ", ".join(sorted("{} {}".format(n, v)
            for v, n in self.variables.items()))
        conStr = ", ".join(sorted("{} {}".format(n, c)
            for c, n in self.constraints.items()))
        optStr = ", ".join(sorted("{}={}".format(n, v)
            for n, v in self.nondefault_options.items()))

        return "{} {} subject to {} using {} and {}.".format(
            dirStr, objStr,
            conStr if conStr else "no constraints",
            varStr if varStr else "no variables",
            optStr if optStr else "default options")

    def __repr__(self):
        return glyphs.repr2("Footprint", str(self))

    @classmethod
    def from_problem(cls, problem):
        """Create a footprint from a problem instance."""
        return cls(chain(
            (("dir", problem.objective.direction, None),
             ("obj", problem.objective.normalized.function.type, None)),
            (("var", v.var_type, 1) for v in problem.variables.values()),
            (("con", con.type, 1) for con in problem.constraints.values()),
            (("opt", n, v) for n, v in problem.options.nondefaults.items())))

    @classmethod
    def from_types(cls, obj_dir, obj_func, vars=[], cons=[], nd_opts={}):
        """Create a footprint from collections of detailed types.

        :param str obj_dir:
            Objective direction.

        :param obj_func:
            Detailed objective function type.

        :parm list(tuple) vars:
            A list of pairs of detailed variable type and occurence count.

        :parm list(tuple) cons:
            A list of pairs of detailed constraint type and occurence count.

        :param list(str) nd_opts:
            A dictionary mapping option names to nondefault values.
        """
        return cls(chain(
            (("dir", obj_dir, None),
             ("obj", obj_func, None)),
            (("var", vn[0], vn[1]) for vn in vars),
            (("con", cn[0], cn[1]) for cn in cons),
            (("opt", name, value) for name, value in nd_opts.items())))

    def with_extra_options(self, **extra_options):
        """Return a copy with additional solution search options applied."""
        # Determine the new option set.
        options = self.options.self_or_updated(**extra_options)

        # Delete old nondefault options and set new ones.
        return self.updated(chain(
            (("opt", self.NONE),),
            (("opt", n, v) for n, v in options.nondefaults.items())
        ))

    @cached_property
    def size(self):
        """Return the estimated size of the (dense) scalar constraint matrix."""
        num_vars = 0
        num_cons = 0

        for dim in self.variables.values():
            num_vars += dim

        for con, num in self.constraints.items():
            num_cons += con.clstype._cost(con.subtype)*num

        return max(1, num_vars)*max(1, num_cons)

    @property
    def cost(self):
        """A very rough measure on how expensive solving such a problem is.

        This is logarithmic in the estimated size of the constraint matrix.
        """
        return math.log(self.size, 10)


class Specification:
    """Representation of a mathematical class of optimization problems."""

    def __init__(self, objectives=None, variables=None, constraints=None,
            nondefault_options=None):
        """Create a specification from the given features.

        The token :obj:`None` means no requirement.
        """
        self._objs = objectives

        self._tree = RecordTree(chain(
            (("dir", RecordTree.ALL),),  # Not checked.
            (("obj", RecordTree.ALL),),  # Checked manually.

            (("var", RecordTree.ALL),) if variables is None else
            (("var", var, RecordTree.ALL) for var in variables),

            (("con", RecordTree.ALL),) if constraints is None else
            (("con", con, RecordTree.ALL) for con in constraints),

            (("opt", RecordTree.ALL),) if nondefault_options is None else
            (("opt", opt, RecordTree.ALL) for opt in nondefault_options)
        ))

    # TODO: Consider adding a helper method that also gives one string reason
    #       why the footprint does not match the specification.
    def __contains__(self, footprint):
        """Whether a problem footprint matches the specification."""
        if not isinstance(footprint, Footprint):
            raise TypeError("May only check if a footprint matches the "
                "specification, not an object of type {}."
                .format(type(footprint).__name__))

        if self._objs is not None:
            obj = footprint.objective.clstype

            if not any(issubclass(obj, allowed) for allowed in self._objs):
                return False

        if not footprint << self._tree:
            return False

        return True

    def mismatch_reason(self, footprint):
        """Give one string reason why the given footprint does not match."""
        if self._objs is not None:
            obj = footprint.objective.clstype

            if not any(issubclass(obj, allowed) for allowed in self._objs):
                return "Optimizing {}.".format(obj.__name__)

        mismatch = footprint.mismatch(self._tree)

        for path in mismatch.get("var").paths:
            return "Representing {}.".format(path[0].__name__)

        for path in mismatch.get("con").paths:
            return "Obeying {}.".format(path[0].__name__)

        for path in mismatch.get("opt").paths:
            return "Setting {}.".format(path[0].__name__)

        assert footprint << self._tree

        return "No reason."

    @staticmethod
    def _paths_str(paths):
        return ", ".join(
            ":".join(p.__name__ if isclass(p) else str(p) for p in path)
            for path in paths)

    def __str__(self):
        dirStr = "Optimize"

        if self._objs is None:
            objStr = "any objective"
        elif not self._objs:
            objStr = "no objective"
        else:
            objStr = ", ".join(o.__name__ for o in self._objs)

        vars, cons, opts = (self._tree.get(x) for x in ("var", "con", "opt"))

        if not vars:
            varStr = "no variables"
        elif vars is RecordTree.ALL:
            varStr = "any variables"
        else:
            varStr = self._paths_str(vars.paths)

        if not cons:
            conStr = "no constraint"
        elif cons is RecordTree.ALL:
            conStr = "any constraint"
        else:
            conStr = self._paths_str(cons.paths)

        if not opts:
            optStr = "default options"
        elif opts is RecordTree.ALL:
            optStr = "any options"
        else:
            optStr = "nondefault values for " + self._paths_str(opts.paths)

        return "{} {} subject to {} using {} and {}.".format(
            dirStr, objStr, conStr, varStr, optStr)

    def __repr__(self):
        return glyphs.repr2("Specification", str(self))


# --------------------------------------
__all__ = api_end(_API_START, globals())
