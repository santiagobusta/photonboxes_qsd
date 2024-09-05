# ------------------------------------------------------------------------------
# Copyright (C) 2019-2020 Maximilian Stahlberg
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

"""Implements the :class:`Mutable` base class for variables and parameters."""

import random
import threading
from abc import ABC, abstractmethod
from copy import copy as pycopy

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from .data import load_data
from .expression import Expression, NotValued

_API_START = api_start(globals())
# -------------------------------


# The mutable IDs start at a random value to prevent a clash if mutables are
# pickled and loaded in another python session. The choice of 31 bits is to
# produce up to 2**31-1 short hexadecimal IDs for use in private mutable names.
_NEXT_MUTABLE_ID = int(random.getrandbits(31))

# A lock for _NEXT_MUTABLE_ID, if the user uses threads to create mutables.
_MUTABLE_ID_LOCK = threading.Lock()


def _make_mutable_id(dim):
    """Create a unique (starting) ID for a new mutable."""
    with _MUTABLE_ID_LOCK:
        global _NEXT_MUTABLE_ID
        id = _NEXT_MUTABLE_ID
        _NEXT_MUTABLE_ID += dim
        return id


# A map used when converting IDs to strings.
_ID_HEX2STR_MAP = {ord(a): ord(b) for a, b in zip(
    "0123456789abcdef", "ABCDEFGHIJKLMNOP")}


def _id_to_str(id):
    """Convert a mutable id to a short string identifier."""
    return "{:08x}".format(id).translate(_ID_HEX2STR_MAP)


class Mutable(ABC):
    """Primary base class for all variable and parameter types.

    Mutables need to inherit this class with priority (first class listed) and
    the affine expression type that they represent without priority.
    """

    def __init__(self, name, vectorization):
        """Perform basic initialization for :class:`Mutable` instances.

        :param str name:
            Name of the mutable. A leading `"__"` denotes a private mutable
            and is replaced by a sequence containing the mutable's unique ID.

        :param vectorization:
            Vectorization format used to store the value.
        :type vectorization:
            ~picos.expressions.vectorizations.BaseVectorization
        """
        if not isinstance(self, Expression):
            raise TypeError("{} may not be initialized directly.".format(
                type(self).__name__))

        self._id    = _make_mutable_id(vectorization.dim)
        self._vec   = vectorization
        self._value = None

        if not name or name.startswith("__"):
            id_str = "__{}_".format(_id_to_str(self._id))
            name = name.replace("__", id_str, 1) if name else id_str + "_"

        self._name = name

    @abstractmethod
    def copy(self, new_name=None):
        """Return an independent copy of the mutable.

        Note that unlike constraints which keep their ID on copy, mutables are
        supposed to receive a new id.
        """
        pass

    @property
    def id(self):
        """The unique (starting) ID of the mutable, assigned at creation."""
        return self._id

    def id_at(self, index):
        """Return the unique ID of a scalar entry, assigned at creation."""
        if index < 0 or index >= self._vec.dim:
            raise IndexError("Bad index {} for the {} dimensional mutable {}."
                .format(index, self._vec.dim, self.name))

        return self._id + index

    @property
    def dim(self):
        """The mutable's dimension on the real field.

        This corresponds to the length of its vectorized value.
        """
        return self._vec.dim

    @property
    def name(self):
        """The name of the mutable."""
        return self._name

    @cached_property
    def long_string(self):
        """A string used to represent the mutable in a problem string."""
        return "{} {} {}".format(glyphs.shape(self.shape),
            self._get_type_string_base().lower(), self._name)

    def _load_vectorized(self, value):
        """Support :meth:`__init__` and ``_set_value``."""
        return self._vec.vectorize(
            load_data(value, self._vec.shape, self._typecode)[0])

    def _check_internal_value(self, value):
        """Support :meth:`_set_value` and :meth:`_set_internal_value`."""
        pass

    # NOTE: This needs to be inherited with priority.
    def _get_value(self):
        return self._vec.devectorize(self._get_internal_value())

    # NOTE: This needs to be inherited with priority.
    def _set_value(self, value):
        if value is None:
            self._value = None
            return

        try:
            value = self._load_vectorized(value)
            self._check_internal_value(value)
        except Exception as error:
            raise type(error)("Failed to assign a value to mutable {}: {}"
                .format(self.string, error)) from None
        else:
            self._value = value

    def _get_internal_value(self):
        if self._value is None:
            raise NotValued("Mutable {} is not valued.".format(self.string))
        else:
            return pycopy(self._value)

    def _set_internal_value(self, value):
        if value is None:
            self._value = None
            return

        try:
            value = load_data(value, (self._vec.dim, 1), "d")[0]
            self._check_internal_value(value)
        except Exception as error:
            raise type(error)(
                "Failed to assign an internal value to mutable {}: {}"
                .format(self.string, error)) from None
        else:
            self._value = value

    internal_value = property(
        lambda self: self._get_internal_value(),
        lambda self, value: self._set_internal_value(value),
        lambda self: self._set_internal_value(None),
        """The internal (special vectorized) value of the mutable.""")

    def _get_clstype(self):
        # Mimic an ordinary (complex) affine expression.
        return self._basetype

    def _get_subtype(self):
        # Mimic an ordinary (complex) affine expression.
        return self._basetype._get_subtype(self)

    @classmethod
    def _predict(cls, subtype, relation, other):
        # Mimic an ordinary (complex) affine expression.
        return cls._get_basetype()._predict(subtype, relation, other)


# --------------------------------------
__all__ = api_end(_API_START, globals())
