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

"""Backend for problem reformulation classes."""

# ------------------------------------------------------------------------------
# TODO: Make the solver base class inherit from Reformulation, in particular the
#       update related methods.
# ------------------------------------------------------------------------------

from abc import ABC, abstractmethod

from ..apidoc import api_end, api_start
from ..modeling import Footprint, Problem, Solution

_API_START = api_start(globals())
# -------------------------------


class Reformulation(ABC):
    """Base class for problem reformulations.

    Abstract base class for a reformulation from one (possibly already
    reformulated) problem form to another.
    """

    # --------------------------------------------------------------------------
    # The following section contains the abstract methods, which are exactly the
    # methods that reformulation implementations are supposed to override.
    # --------------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def supports(cls, footprint):
        """Whether the reformulation affects problems with the given footprint.

        The reformulation must support every problem with such a footprint and
        the resulting problem should have a changed footprint.
        """
        pass

    @classmethod
    @abstractmethod
    def predict(cls, footprint):
        """Predict the reformulation's effect on a problem footprint.

        Given a problem footprint, returns another problem footprint that a
        problem with the former one would be reformulated to.

        This is used to predict the effects of a reformulation when planning
        a solution strategy without the cost of actually transforming a problem.
        """
        pass

    @abstractmethod
    def forward(self):
        """Perform the initial problem reformulation.

        Creates a modified copy or clone of the problem in :attr:`input` and
        stores it as :attr:`output`.

        See :meth:`~.problem.Problem.copy` and :meth:`~.problem.Problem.clone`
        for the differences between a copy and a clone.

        Implementations are supposed to do the necessary bookkeeping so that
        :meth:`backward` can transform a solution to the new problem back to a
        solution of the original problem.
        """
        pass

    @abstractmethod
    def update(self):
        """Update a previous problem reformulation.

        Updates :attr:`output` and related bookkeeping information with respect
        to changes in :attr:`input`.

        :raises NotImplementedError:
            If performing an update is not feasible for the reformulation.
        """
        pass

    @abstractmethod
    def backward(self, solution):
        """Translate back a solution from reformulated to original problem.

        Transforms a single :class:`~.solution.Solution` to :attr:`output` to a
        solution of :attr:`input`.

        The method is allowed to modify the solution; it is not necessary to
        work on a copy. In particular, :meth:`~.solution.Solution.attach_to` can
        be used if :meth:`forward` has created a deep copy of the problem.
        """
        pass

    # --------------------------------------------------------------------------
    # The following section contains the non-abstract methods, which are exactly
    # the methods that reformulation implementations are supposed to inherit.
    # --------------------------------------------------------------------------

    def _reset_knowns(self):
        self._knownObjective   = None
        self._knownVariables   = set()
        self._knownConstraints = set()

    def _set_knowns(self):
        self._knownObjective   = self.input.objective
        self._knownVariables   = set(self.input.variables.values())
        self._knownConstraints = set(self.input.constraints.values())

    def __init__(self, theObject):
        """Initialize :class:`Reformulation` instances.

        :param theObject: The input to work on; either an optimization problem
            or the (future) output of another reformulation.
        :type theObject:
            ~picos.Problem or ~picos.reforms.reformulation.Reformulation
        """
        if isinstance(theObject, Problem):
            self.predecessor = None
            self._input      = theObject
        elif isinstance(theObject, Reformulation):
            self.predecessor = theObject
            theObject.successor = self
        elif isinstance(theObject, Footprint):
            raise TypeError("Reformulations cannot be instanciated using "
                "problem footprints. Use the predict classmethod.")
        else:
            raise TypeError("Cannot reformulate an object of type '{}'."
                .format(type(theObject).__name__))

        self.successor = None
        """The next reformulation in the pipeline."""

        self.output = None
        """The output problem."""

        self._reset_knowns()

    def reset(self):
        """Reset the pipeline from this reformulation onward.

        This is done whenever a reformulation does not implement :meth:`update`
        so that succeeding reformulations do not attempt to update a problem
        which was completely rewritten as this may be inefficient.
        """
        assert self.successor, \
            "The reformulation being reset has no successor."

        self.output = None
        self._reset_knowns()

        self.successor.reset()

    input = property(lambda self:
        self.predecessor.output if self.predecessor else self._input,
        doc="The input problem.")

    verbosity = property(lambda self: self.input.options.verbosity,
        doc="Verbosity level of the reformulation; same as for input problem.")

    def _verify_prediction(self):
        if not self.input.options.verify_prediction:
            return

        expectedType = self.predict(Footprint.from_problem(self.input))
        outputType = Footprint.from_problem(self.output)

        if outputType != expectedType:
            raise RuntimeError("{} failed to produce a problem with the "
                "expected footprint:\nEXPECTED: {}\nOUTCOME : {}\n"
                "This is a bug; please report it to the PICOS developers. "
                "You can disable the 'verify_prediction' option to try solving "
                "anyway.".format(type(self).__name__, expectedType, outputType))

    def execute(self):
        """Reformulate the problem and obtain a solution from the result.

        For this to work there needs to be a solver instance at the end of the
        reformulation pipeline, which would implement its own version of this
        method that actually solves the problem and produces the first solution.
        """
        assert self.successor, \
            "The reformulation being executed has no successor."

        verbose = self.verbosity > 0

        # Update the output problem if possible.
        if self.output:
            if verbose:
                print("Updating {}.".format(self.__class__.__name__))

            try:
                self.update()
            except NotImplementedError:
                if verbose:
                    print("Update failed: Not implemented for {}."
                        .format(self.__class__.__name__))

                self.reset()

        # Create the output problem if necessary.
        if not self.output:
            if verbose:
                print("Applying {}.".format(self.__class__.__name__))

            self.forward()
            self._set_knowns()

        # Verify that the output problem is of the expected type.
        self._verify_prediction()

        # Advance one step.
        outputSolution = self.successor.execute()

        # Transform the solution of the output problem for the input problem.
        if isinstance(outputSolution, Solution):
            return self.backward(outputSolution)
        else:
            return [self.backward(solution) for solution in outputSolution]

    def _objective_has_changed(self):
        """Check for an objective function change.

        :returns: Whether the optimization objective has changed since the last
            forward or update.
        """
        assert self._knownObjective is not None, \
            "_objective_has_changed may only be used inside _update_problem."

        objectiveChanged = self._knownObjective != self.input.objective

        if objectiveChanged:
            self._knownObjective = self.input.objective

        return objectiveChanged

    def _new_variables(self):
        """Check for new variables.

        Yields variables that were added to the input problem since the last
        forward or update.

        Note that variables received from this method will also be added to the
        set of known variables, so you can only iterate once within each update.
        """
        for variable in self.input.variables.values():
            if variable not in self._knownVariables:
                self._knownVariables.add(variable)
                yield variable

    def _removed_variables(self):
        """Check for removed variables.

        Yields variables that were removed from the input problem since the last
        forward or update.

        Note that variables received from this method will also be removed from
        the set of known variables, so you can only iterate once within each
        update.
        """
        newVariables = set(self.input.variables.values())
        for variable in self._knownVariables:
            if variable not in newVariables:
                yield variable
        self._knownVariables.intersection_update(newVariables)

    def _new_constraints(self):
        """Check for new constraints.

        Yields constraints that were added to the input problem since the last
        forward or update.

        Note that constraints received from this method will also be added to
        the set of known constraints, so you can only iterate once within each
        update.
        """
        for constraint in self.input.constraints.values():
            if constraint not in self._knownConstraints:
                self._knownConstraints.add(constraint)
                yield constraint

    def _removed_constraints(self):
        """Check for removed constraints.

        Yields constraints that were removed from the input problem since the
        last forward or update.

        Note that constraints received from this method will also be removed
        from the set of known constraints, so you can only iterate once within
        each update.
        """
        newConstraints = set(self.input.constraints.values())
        for constraint in self._knownConstraints:
            if constraint not in newConstraints:
                yield constraint
        self._knownConstraints.intersection_update(newConstraints)

    def _pass_updated_objective(self):
        """Pass changes in the objective function from input to output problem.

        .. warning::
            This method resets the objective-has-changed state.
        """
        if self._objective_has_changed():
            self.output.objective = self.input.objective

    # TODO: Determine if and in which form such a method is needed now that
    #       variables are added to problems implicitly (but still explicitly to
    #       solvers).
    def _pass_updated_vars(self):
        """Pass variable changes from input to output problem.

        Adds all new varibles in the input problem to the output problem, and
        removes all variables removed from the input problem also from the
        output problem.

        .. warning::
            Variables are passed as with :meth:`Problem.clone`, not copied.

        .. warning::
            This method clears the buffers of new and removed variables.
        """
        for variable in self._new_variables():
            pass

        for variable in self._removed_variables():
            pass

    def _pass_updated_cons(self, ignore=type(None)):
        """Pass constraint changes from input to output problem.

        Adds all new constraints in the input problem to the output problem, and
        removes all constraints removed from the input problem also from the
        output problem.

        :param type ignore: Constraints of this type are not handled. Instead,
            the method returns a pair `(added, removed)` that contains the
            respective constraints that were not handled.

        .. warning::
            Constraints are passed as with :meth:`Problem.clone`, not copied.

        .. warning::
            This method clears the buffers of new and removed constraints.
        """
        added, removed = [], []

        for constraint in self._new_constraints():
            assert constraint.id not in self.output.constraints
            if isinstance(constraint, ignore):
                added.append(constraint)
            else:
                self.output.add_constraint(constraint)

        for constraint in self._removed_constraints():
            if isinstance(constraint, ignore):
                # No assertion in this case, because the reformulation would not
                # have added such a constraint.
                removed.append(constraint)
            else:
                assert constraint.id in self.output.constraints
                self.output.remove_constraint(constraint.id)

        if ignore is not type(None):  # noqa: E721
            return added, removed

    def _pass_updated_options(self):
        """Make the output problem use the same options object as the input."""
        self.output.options = self.input.options


# --------------------------------------
__all__ = api_end(_API_START, globals())
