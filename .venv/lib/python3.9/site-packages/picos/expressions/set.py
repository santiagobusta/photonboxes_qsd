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

"""Backend for mathematical set type implementations."""

from abc import ABC, abstractmethod

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from .expression import (ExpressionType, convert_operands, refine_operands,
                         validate_prediction)

_API_START = api_start(globals())
# -------------------------------


class SetType(ExpressionType):
    """:class:`~picos.expressions.ExpressionType` for sets."""

    pass


class Set(ABC):
    """Abstract base class for mathematical set expressions."""

    def __init__(self, typeStr, symbStr):
        """Perform basic initialization for :class:`Set` instances.

        :param str typeStr: Short string denoting the set type.
        :param str symbStr: Algebraic string description of the set.
        """
        self._typeStr = typeStr
        self._symbStr = symbStr

    @property
    def string(self):
        """Symbolic string representation of the set."""
        return self._symbStr

    @property
    @abstractmethod
    def Subtype(self):
        """Analog to :meth:`.expression.Expression.Subtype`."""
        pass

    @property
    def type(self):
        """Analog to :meth:`.expression.Expression.type`."""
        return ExpressionType(self.__class__, self._get_subtype())

    @classmethod
    def make_type(cls, *args, **kwargs):
        """Analog to :meth:`.expression.Expression.make_type`."""
        return ExpressionType(cls, cls.Subtype(*args, **kwargs))

    @property
    def subtype(self):
        """Analog to :meth:`.expression.Expression.subtype`."""
        return self._get_subtype()

    @property
    def refined(self):
        """The set itself, as sets do not support refinement.

        This exists for compatibility with expressions.
        """
        return self

    def __repr__(self):
        return str(glyphs.repr2(self._typeStr, self._symbStr))

    def __str__(self):
        return str(self._symbStr)

    def __format__(self, format_spec):
        return self._symbStr.__format__(format_spec)

    @abstractmethod
    def _get_subtype(self):
        """:meth:`picos.expressions.Expression._get_subtype`."""
        pass

    @classmethod
    @abstractmethod
    def _predict(cls, subtype, relation, other):
        """See :meth:`picos.expressions.Expression._predict`."""
        pass

    @abstractmethod
    def _get_mutables(self):
        """Return a Python set of mutables that are involved in the set."""
        pass

    mutables = property(
        lambda self: self._get_mutables(),
        doc=_get_mutables.__doc__)

    @cached_property
    def variables(self):
        """The set of decision variables that are involved in the set."""
        from .variables import BaseVariable

        return frozenset(mutable for mutable in self._get_mutables()
            if isinstance(mutable, BaseVariable))

    @cached_property
    def parameters(self):
        """The set of parameters that are involved in the set."""
        from .variables import BaseVariable

        return frozenset(mutable for mutable in self._get_mutables()
            if not isinstance(mutable, BaseVariable))

    @abstractmethod
    def _replace_mutables(self, mapping):
        """See :meth:`~.expression.Expression._replace_mutables`."""
        pass

    # HACK: Borrow Expression.replace_mutables.
    # TODO: Common base class ExpressionOrSet.
    def replace_mutables(self, new_mutables):
        """See :meth:`~.expression.Expression.replace_mutables`."""
        from .expression import Expression
        return Expression.replace_mutables(self, new_mutables)

    # --------------------------------------------------------------------------
    # Turn __lshift__ and __rshift__ into a single binary relation.
    # This is used for both Loewner order (defining LMIs) and set membership.
    # TODO: Define this in a common base class of Expression and Set.
    # --------------------------------------------------------------------------

    def _lshift_implementation(self, other):
        return NotImplemented

    def _rshift_implementation(self, other):
        return NotImplemented

    @convert_operands()
    @validate_prediction
    @refine_operands()
    def __lshift__(self, other):
        result = self._lshift_implementation(other)

        if result is NotImplemented:
            result = other._rshift_implementation(self)

        return result

    @convert_operands()
    @validate_prediction
    @refine_operands()
    def __rshift__(self, other):
        result = self._rshift_implementation(other)

        if result is NotImplemented:
            result = other._lshift_implementation(self)

        return result


# --------------------------------------
__all__ = api_end(_API_START, globals())
