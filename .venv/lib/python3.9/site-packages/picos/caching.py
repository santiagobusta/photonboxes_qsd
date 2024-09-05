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

"""Caching helpers."""

import functools
from contextlib import contextmanager

from .apidoc import api_end, api_start

_API_START = api_start(globals())
# -------------------------------


#: The prefix used for storing cached values.
CACHED_PREFIX = "_cached_"

#: An attribute name whose presence unlocks the setter of cached properties.
CACHED_PROP_UNLOCKED_TOKEN = "_CACHED_PROPERTIES_ARE_UNLOCKED"


@contextmanager
def unlocked_cached_properties(obj):
    """Unlock the setters of cached instance attributes.

    Normally, cached attributes are read-only properties. When the user first
    reads them, the cache is populated with the value returned to the user, and
    successive reads will return the cached value.

    The user is allowed to empty the cache by using ``del`` on the variable, but
    they may not assign a value to it. This context allows the programmer to
    manually populate the cache by assigning a value to the property.

    :Example:

    >>> from picos.caching import cached_property, unlocked_cached_properties
    >>> class A:
    ...     @cached_property
    ...     def p(self):
    ...         return 1
    ...
    >>> a = A()
    >>> try:
    ...     a.p = 2
    ... except AttributeError:
    ...     print("Not possible.")
    ...
    Not possible.
    >>> with unlocked_cached_properties(a):
    ...     a.p = 2  # Populate the cache of a.p.
    ...
    >>> a.p
    2
    """
    assert not hasattr(obj, CACHED_PROP_UNLOCKED_TOKEN)
    setattr(obj, CACHED_PROP_UNLOCKED_TOKEN, None)

    try:
        yield
    finally:
        delattr(obj, CACHED_PROP_UNLOCKED_TOKEN)


class cached_property(property):
    """A read-only property whose result is cached."""

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):  # noqa
        if fget is None:
            raise NotImplementedError("Unlike normal properties, cached "
                "properties must be initialized with a getter.")
        elif fget.__name__ == (lambda: None).__name__:
            raise NotImplementedError("Cached properties cannot be used with a "
                "lambda getter as lambdas do not have a unique name to identify"
                " the cached value.")

        if fset is not None:
            raise AttributeError("Cannot have a custom setter on a cached "
                "property as __set__ is reserved for cache population.")

        if fdel is not None:
            raise AttributeError("Cannot have a custom deleter on a cached "
                "property as __delete__ is reserved for cache clearing.")

        self._cache_name = CACHED_PREFIX + fget.__name__

        property.__init__(self, fget, fset, fdel, doc)

        self.__module__ = fget.__module__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        if self.fget is None:
            raise AttributeError("unreadable attribute")  # Mimic Python.

        if not hasattr(obj, self._cache_name):
            # Compute and cache the value.
            setattr(obj, self._cache_name, self.fget(obj))

        return getattr(obj, self._cache_name)

    def __set__(self, obj, value):
        if not hasattr(obj, CACHED_PROP_UNLOCKED_TOKEN):
            raise AttributeError("can't set attribute")  # Mimic Python.

        # Populate the cache.
        setattr(obj, self._cache_name, value)

    def __delete__(self, obj):
        # Empty the cache.
        if hasattr(obj, self._cache_name):
            delattr(obj, self._cache_name)

    def getter(self, fget):  # noqa
        raise NotImplementedError(
            "Cannot change the getter on a cached property.")

    def setter(self, fset):  # noqa
        raise AttributeError("Cannot add a setter to a cached property.")

    def deleter(self, fdel):  # noqa
        return AttributeError("Cannot add a deleter to a cached property.")


class cached_selfinverse_property(cached_property):
    """A read-only, self-inverse property whose result is cached."""

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        if self.fget is None:
            raise AttributeError("unreadable attribute")  # Mimic Python.

        if not hasattr(obj, self._cache_name):
            # Compute and cache the value, populate the value's cache.
            value = self.fget(obj)
            setattr(obj, self._cache_name, value)
            setattr(value, self._cache_name, obj)

        return getattr(obj, self._cache_name)


def cached_unary_operator(operator):
    """Make a unary operator method cache its result.

    This is supposed to be used for property-like special methods such as
    ``__neg__`` where :func:`cached_property` can't be used.
    """
    cacheName = CACHED_PREFIX + operator.__name__

    @functools.wraps(operator)
    def wrapper(self):
        if not hasattr(self, cacheName):
            setattr(self, cacheName, operator(self))
        return getattr(self, cacheName)
    return wrapper


def cached_selfinverse_unary_operator(operator):
    """Make a self-inverse unary operator method cache its result.

    This is supposed to be used for property-like special methods such as
    ``__neg__`` where :func:`cached_property` can't be used.

    .. warning::
        The result returned by the wrapped operator must be a fresh object as
        it will be modified.
    """
    cacheName = CACHED_PREFIX + operator.__name__

    @functools.wraps(operator)
    def wrapper(self):
        if not hasattr(self, cacheName):
            value = operator(self)
            setattr(self, cacheName, value)
            setattr(value, cacheName, self)
        return getattr(self, cacheName)
    return wrapper


def empty_cache(obj):
    """Clear all cached values of an object."""
    for name in dir(obj):
        if name.startswith(CACHED_PREFIX):
            delattr(obj, name)


def borrow_cache(target, source, names):
    """Copy cached values from one object to another.

    :param target: The object to populate the cache of.
    :param source: The object to take cached values from.
    :param names: Names of cached properties or functions to borrow.
    """
    for name in names:
        cacheName = CACHED_PREFIX + name
        if hasattr(source, cacheName):
            setattr(target, cacheName, getattr(source, cacheName))


# --------------------------------------
__all__ = api_end(_API_START, globals())
