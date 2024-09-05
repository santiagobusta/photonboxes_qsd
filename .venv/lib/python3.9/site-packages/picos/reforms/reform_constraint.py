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

"""Reformulations that concern a particular type of constraint.

The reformulations' logic is not found here but defined within the constraint
classes in the form of a :class:`constraint conversion class
<picos.constraints.constraint.ConstraintConversion>`.
"""

import inspect

from .. import constraints
from ..constraints import Constraint, ConstraintConversion
from ..modeling.footprint import Footprint
from .reformulation import Reformulation

# No call to apidoc.api_start: Module defines __all__.
# ----------------------------------------------------


def reformulation_init(self, theObject):
    """Implement :meth:`~.reformulation.Reformulation.__init__`."""
    Reformulation.__init__(self, theObject)

    self.constraintType = self.__class__.CONSTRAINT_TYPE
    self.makeTmpProblem = self.__class__.CONVERSION_TYPE.convert
    self.makeDualValue  = self.__class__.CONVERSION_TYPE.dual


def reformulation_supports(cls, footprint):
    """Implement :meth:`~.reformulation.Reformulation.supports`."""
    return ("con", cls.CONSTRAINT_TYPE) in footprint


def reformulation_predict(cls, footprint):
    """Implement :meth:`~.reformulation.Reformulation.predict`."""
    updates = [("con", cls.CONSTRAINT_TYPE, Footprint.NONE)]

    for subtype, count in footprint[("con", cls.CONSTRAINT_TYPE)].items:
        assert len(subtype) == 1
        subtype = subtype[0]

        for addition in cls.CONVERSION_TYPE.predict(subtype, footprint.options):
            assert isinstance(addition[-1], int)
            updates.append(addition[:-1] + (addition[-1]*count,))

    return footprint.updated(updates)


def reformulation_reform_single(self, constraint, options):
    """Convert a single constraint."""
    assert isinstance(constraint, self.constraintType)

    # Create a temporary problem from the constraint to be replaced.
    tmpProblem = self.makeTmpProblem(constraint, options)

    # Keep track of auxilary vars/cons replacing the constraint.
    self.auxVars[constraint] = {}
    self.auxCons[constraint] = []

    # If the constraint to be transformed is part of the output prolem, remove
    # it. This is the case when forwarding but not when updating.
    if constraint.id in self.output.constraints:
        self.output.remove_constraint(constraint.id)

    # Remember auxiliary variables added so that their value can be removed from
    # the solution in reformulation_backward.
    for tmpVarName, tmpVar in tmpProblem.variables.items():
        self.auxVars[constraint][tmpVarName] = tmpVar

    # Add auxiliary constraints to the output problem.
    # HACK: This only works while Problem.constraints is an OrderedDict as
    #       ConstraintConversion needs a way to identify the constraints added
    #       to the temporary problem without having any pointer to them.
    auxCons = self.output.add_list_of_constraints(
        tmpProblem.constraints.values())
    self.auxCons[constraint].extend(auxCons)


def reformulation_forward(self):
    """Implement :meth:`~.reformulation.Reformulation.forward`."""
    self.output = self.input.clone(copyOptions=False)

    self.auxVars = {}
    self.auxCons = {}

    # TODO: Give Problem quick iterators over constraints of certain type.
    for constraint in self.input.constraints.values():
        if isinstance(constraint, self.constraintType):
            self._reform_single(constraint, self.input.options)


def reformulation_update(self):
    """Implement :meth:`~.reformulation.Reformulation.update`."""
    # Pass changes in the objective function.
    self._pass_updated_objective()

    # Pass all variables as they are.
    self._pass_updated_vars()

    # Pass all unhandled constraints as they are.
    added, removed = self._pass_updated_cons(ignore=self.constraintType)

    # Pass all option changes.
    self._pass_updated_options()

    # Reformulate new relevant constraints.
    for constraint in added:
        self._reform_single(constraint, self.input.options)

    # Remove auxiliary objects added for relevant removed constraints.
    for constraint in removed:
        assert constraint in self.auxVars and constraint in self.auxCons

        for auxCon in self.auxCons[constraint]:
            assert auxCon.id in self.output.constraints
            self.output.remove_constraint(auxCon.id)

        self.auxCons.pop(constraint)
        self.auxVars.pop(constraint)


def reformulation_backward(self, solution):
    """Implement :meth:`~.reformulation.Reformulation.backward`."""
    # TODO: Give Problem quick iterators over constraints of certain type.
    for constraint in self.input.constraints.values():
        if isinstance(constraint, self.constraintType):
            try:
                solution.duals[constraint] = self.makeDualValue(
                    {v: solution.primals.get(v)
                        for v in self.auxVars[constraint]},
                    [solution.duals.get(c) for c in self.auxCons[constraint]],
                    self.input.options)
            except NotImplementedError:
                solution.duals[constraint] = None

            for auxVar in self.auxVars[constraint]:
                if auxVar in solution.primals:
                    solution.primals.pop(auxVar)

            for auxCon in self.auxCons[constraint]:
                if auxCon in solution.duals:
                    solution.duals.pop(auxCon)

    return solution


def make_constraint_reformulation(constraint, conversion):
    """Produce a :class:`Reformulation` from a :class:`ConstraintConversion`.

    A helper that creates a :class:`Reformulation` type (subclass) from a
    :class:`Constraint` and :class:`constraint.ConstraintConversion` types.
    """
    assert constraint.__name__.endswith("Constraint"), \
        "Constraint types must have a name ending in 'Constraint'."

    assert conversion.__name__.endswith("Conversion"), \
        "Constraint conversions must have a name ending in 'Conversion'."

    constraintName = constraint.__name__[:-len("Constraint")]
    conversionName = conversion.__name__[:-len("Conversion")]

    name = "{}{}Reformulation".format(constraintName,
        "To{}".format(conversionName) if conversionName else "")

    docstring = "Reformulation created from :class:`{1} <{0}.{1}>`." \
        .format(conversion.__module__, conversion.__qualname__)

    # TODO: This would need an additional prediction that yields all class
    #       types of constraints that can be converted. Is that needed?
    # NOTE: SumExponentialsConstraint.LSEConversion makes use of return the
    #       constraint as is for certain unsupported subtypes.
    # assert constraint not in conversion.adds_constraint_types(), \
    #     "Constraint conversions may not add the very type being converted."

    body = {
        # Class constants.
        "__module__": make_constraint_reformulation.__module__,  # Defined here.
        "CONSTRAINT_TYPE": constraint,
        "CONVERSION_TYPE": conversion,

        # Class methods.
        "supports": classmethod(reformulation_supports),
        "predict":  classmethod(reformulation_predict),

        # Instance methods.
        "__init__":       reformulation_init,
        "__doc__":        docstring,
        "_reform_single": reformulation_reform_single,
        "forward":        reformulation_forward,
        "update":         reformulation_update,
        "backward":       reformulation_backward
    }

    # TODO: Check if anything like the following is still necessary.
    # # HACK: See QuadConstraint.ConicConversion.predict.
    # if constraint is constraints.QuadConstraint \
    # and conversion is constraints.QuadConstraint.ConicConversion:
    #     body["_verify_prediction"] = lambda self: None

    return type(name, (Reformulation,), body)


# Allow __init__ to import exactly the generated reformulations using asterisk.
__all__ = []

# For every constraint conversion, generate a problem reformulation.
NUM_REFORMS = 0
CONSTRAINT_TO_REFORMS = {}
for constraint in constraints.__dict__.values():
    if not inspect.isclass(constraint):
        continue

    if not issubclass(constraint, Constraint):
        continue

    CONSTRAINT_TO_REFORMS[constraint] = []

    for conversion in constraint.__dict__.values():
        if not inspect.isclass(conversion):
            continue

        if not issubclass(conversion, ConstraintConversion):
            continue

        reformulation = make_constraint_reformulation(constraint, conversion)

        NUM_REFORMS += 1
        CONSTRAINT_TO_REFORMS[constraint].append(reformulation)

        # Export the reformulations as if they were defined at module level.
        globals()[reformulation.__name__] = reformulation
        __all__.append(reformulation.__name__)

# Make the order of __all__ deterministic.
__all__ = sorted(__all__)

# FIXME: Restore a topological sorting.
# TODO: As above, this would need a separate predictor.
TOPOSORTED_REFORMS = [globals()[name] for name in __all__]


# --------------------------------------------------
# No call to apidoc.ape_end: Module defines __all__.
