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

"""Implements a helper reformulation to apply temporary options."""

from ..apidoc import api_end, api_start
from .reformulation import Reformulation

_API_START = api_start(globals())
# -------------------------------


class ExtraOptions(Reformulation):
    """Helper reformulation to apply temporary options.

    This reformulation is different from all others in a number of ways:

    - It doesn't change the footprint of any problem.
    - It is automatically the first reformulation in any strategy.
    - It is the only reformulation whose :meth:`execute` accepts a keyword
      argument sequence of additional options to use.
    - It is the only reformulation that can be skipped entirely (by setting
      ``self.output = self.input``).

    The job of this reformulation is to apply temporary options passed to
    :meth:`~.problem.Problem.solve` so that subsequent reformulations can find
    their options stored in their input problem.
    """

    @classmethod
    def supports(cls, footprint):
        """Implement :meth:`~.reformulation.Reformulation.supports`."""
        return True

    @classmethod
    def predict(cls, footprint):
        """Implement :meth:`~.reformulation.Reformulation.predict`."""
        # The additional options are already part of the footprint!
        # (They are applied before the first prediction takes place.)
        return footprint

    def execute(self, **extra_options):
        """Override :meth:`~.reformulation.Reformulation.execute`.

        Adds the ``extra_options`` argument and attempts to perform as little
        reformulation work as possible.
        """
        newOptions = self.input.options.self_or_updated(**extra_options)
        verbose = newOptions.verbosity > 0

        if newOptions == self.input.options:
            # No options need to be applied; shortcut this reformulation so that
            # no forwarding or updating work needs to be done.
            if verbose:
                print("Skipping {}.".format(self.__class__.__name__))

            # Shortcut the whole problem.
            self.output = self.input

            self._reset_knowns()
        elif self.output and self.output is not self.input:
            # We made a problem clone before and we still can't shortcut, so
            # perform a regular update and apply the new options to the copy.
            if verbose:
                print("Updating {}.".format(self.__class__.__name__))

            # Update the options.
            self.output.options = newOptions

            self._pass_updated_objective()
            self._pass_updated_vars()
            self._pass_updated_cons()
        else:
            # We need to apply temporary options and this is either the first
            # execution or the previous execution was a shortcut. Perform a
            # regular forward and apply the new options to the problem clone.
            if verbose:
                print("Applying {}.".format(self.__class__.__name__))

            # Copy the problem and update the options.
            self.output = self.input.clone(copyOptions=False)
            self.output.options = newOptions

            self._set_knowns()

        # Make sure there is a successor to obtain a solution from.
        assert self.successor, \
            "The reformulation being executed has no successor."

        # Advance one step and return any solution as-is.
        return self.successor.execute()

    def forward(self):
        """Dummy-implement :meth:`~.reformulation.Reformulation.forward`."""
        pass

    def update(self):
        """Dummy-implement :meth:`~.reformulation.Reformulation.update`."""
        pass

    def backward(self, solution):
        """Dummy-implement :meth:`~.reformulation.Reformulation.backward`."""
        pass


# --------------------------------------
__all__ = api_end(_API_START, globals())
