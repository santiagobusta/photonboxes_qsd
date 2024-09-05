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

"""Domain-specific container types."""

from collections import OrderedDict
from collections.abc import MutableSet
from itertools import chain
from types import MappingProxyType

from .apidoc import api_end, api_start
from .caching import cached_property

_API_START = api_start(globals())
# -------------------------------


class OrderedSet(MutableSet):
    """A set that remembers its insertion order.

    >>> from picos.containers import OrderedSet as oset
    >>> o = oset([4, 3, 2, 1]); o
    OrderedSet([4, 3, 2, 1])
    >>> 3 in o
    True
    >>> o.update([5, 4, 3]); o
    OrderedSet([4, 3, 2, 1, 5])
    >>> list(o)
    [4, 3, 2, 1, 5]
    """

    def __init__(self, iterable=()):
        """Intialize the ordered set.

        :param iterable:
            Iterable to take initial elements from.
        """
        self._dict = OrderedDict((element, None) for element in iterable)

    # --------------------------------------------------------------------------
    # Special methods not implemented by MutableSet.
    # --------------------------------------------------------------------------

    def __str__(self):
        return "{{{}}}".format(", ".join(str(element) for element in self))

    def __repr__(self):
        return "OrderedSet([{}])".format(
            ", ".join(str(element) for element in self))

    # --------------------------------------------------------------------------
    # Abstract method implementations.
    # --------------------------------------------------------------------------

    def __contains__(self, key):
        return key in self._dict

    def __iter__(self):
        return iter(self._dict.keys())

    def __len__(self):
        return len(self._dict)

    def add(self, element):
        """Add an element to the set."""
        self._dict[element] = None

    def discard(self, element):
        """Discard an element from the set.

        If the element is not contained, do nothing.
        """
        self.pop(element, None)

    # --------------------------------------------------------------------------
    # Overridingns to improve performance over MutableSet's implementation.
    # --------------------------------------------------------------------------

    def clear(self):
        """Clear the set."""
        self._dict.clear()

    # --------------------------------------------------------------------------
    # Methods provided by set but not by MutableSet.
    # --------------------------------------------------------------------------

    def update(self, *iterables):
        """Update the set with elements from a number of iterables."""
        for iterable in iterables:
            for element in iterable:
                self._dict[element] = None

    difference = property(
        lambda self: self.__sub__,
        doc=set.difference.__doc__)

    difference_update = property(
        lambda self: self.__isub__,
        doc=set.difference_update.__doc__)

    intersection = property(
        lambda self: self.__and__,
        doc=set.intersection.__doc__)

    intersection_update = property(
        lambda self: self.__iand__,
        doc=set.intersection_update.__doc__)

    issubset = property(
        lambda self: self.__le__,
        doc=set.issubset.__doc__)

    issuperset = property(
        lambda self: self.__ge__,
        doc=set.issuperset.__doc__)

    symmetric_difference = property(
        lambda self: self.__xor__,
        doc=set.symmetric_difference.__doc__)

    symmetric_difference_update = property(
        lambda self: self.__ixor__,
        doc=set.symmetric_difference_update.__doc__)

    union = property(
        lambda self: self.__or__,
        doc=set.union.__doc__)


class frozendict(dict):
    """An immutable, hashable dictionary."""

    @classmethod
    def fromkeys(cls, iterable, value=None):
        """Overwrite :meth:`dict.fromkeys`."""
        return cls(dict.fromkeys(iterable, value))

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash(tuple(sorted(self.items())))

        return self._hash

    def __str__(self):
        return dict.__repr__(self)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self)

    @property
    def _modify(self, *args, **kwargs):
        raise AttributeError(
            "Cannot modify a {}.".format(self.__class__.__name__))

    __delitem__ = _modify
    __setitem__ = _modify
    clear = _modify
    pop = _modify
    popitem = _modify
    setdefault = _modify
    update = _modify

    def copy(self):
        """Since :class:`frozendict` are immutable, returns self."""
        return self


class DetailedType:
    """Container for a pair of Python class and subtype.

    A container for a pair of a python class type and some logical subtype data
    structure, together called a detailed type.

    A detailed type is used when the mathematical type of an object must be
    distinguished more precisely than at the level of the python classes used to
    represent such mathematical objects. For instance, a single python class
    would be used for a type of expressions of varying dimensionality and
    subtypes would be used to distinguish further based on dimension.

    Instances of this class are treated exceptionally when used as a label of
    a :class:`RecordTree`: They are expanded into the class and the subtype as
    two seperate labels, making it convenient to store detailed types in trees.
    """

    def __init__(self, theClass, subtype):
        """Construct a :class:`DetailedType`.

        :param type theClass: The Python class part of the detailed type.
        :param object subtype: Additional type information.
        """
        if not hasattr(subtype, "_asdict"):
            raise TypeError("The given subtype of {} is not a namedtuple "
                "instance.".format(subtype))

        self.clstype = theClass
        self.subtype = subtype

    def __iter__(self):
        yield self.clstype
        yield self.subtype

    def __hash__(self):
        return hash((self.clstype, self.subtype))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def equals(self, other):
        """Whether two detailed types are the same."""
        return hash(self) == hash(other)

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, str(self))

    def __str__(self):
        subtypeArgsStr = "|".join("{}={}".format(key, val)
            for key, val in self.subtype._asdict().items())

        return "{}[{}]".format(self.clstype.__name__, subtypeArgsStr)

    def __add__(self, other):
        if isinstance(other, tuple):
            return tuple(self) + other
        elif isinstance(other, list):
            return list(self) + other
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, tuple):
            return other + tuple(self)
        elif isinstance(other, list):
            return other + list(self)
        else:
            return NotImplemented


class RecordTreeToken:
    """Base class for special :class:`RecordTree` value tokens."""

    def __init__(self):
        """Raise a :exc:`TypeError` on instanciation."""
        raise TypeError("{} may not be initialized.".format(
            self.__class__.__name__))


class RecordTree():
    """Labeled tree for storing records.

    An immutable labeled tree with values at the leaf nodes, where labels and
    values are arbitrary hashable python objects.

    An n-tuple whose first (n-1) elements are labels and whose last element is a
    value is called a record. Thus, every path from the root node to a leaf node
    represents one record.

    :class:`DetailedType` labels are treated exceptionally: They are expanded
    into the class and the subtype as two seperate labels.
    """

    class _NodeDict(dict):
        pass

    class NONE(RecordTreeToken):
        """Special :class:`RecordTree` value: No subtrees.

        If inserted at some (inner) node of the tree, the whole subtree starting
        at that node is deleted. If that node's parent node has no other
        children, then the parent node is deleted as well. This process is
        repeated iteratively up to the root node, which is never deleted.

        This is the only value that may be inserted at an inner node.

        This value cannot itself be stored in the tree as its insertion is
        always read as a deletion.
        """

        pass

    class ALL(RecordTreeToken):
        """Special :class:`RecordTree` value: Any subtrees.

        A special value that behaves as an arbitrary subtree during subtree
        checks.
        """

        pass

    @classmethod
    def _flatten(cls, path):
        for index, thing in enumerate(path):
            if isinstance(thing, DetailedType):
                return cls._flatten(path[:index] + thing + path[index+1:])
        return path

    @classmethod
    def _freeze(cls, value):
        """Make a label or value hashable."""
        if isinstance(value, list):
            newValue = tuple(value)
        elif isinstance(value, set):
            newValue = frozenset(value)
        elif isinstance(value, dict):
            newValue = frozendict(value)
        else:
            newValue = value

        try:
            hash(newValue)
        except Exception as error:
            raise TypeError("Failed to freeze {} to a hashable type."
                .format(value)) from error

        return newValue

    @staticmethod
    def _keyval_iterator(recordsOrDict):
        if isinstance(recordsOrDict, dict):
            return recordsOrDict.items()
        else:
            return ((rec[:-1], rec[-1]) for rec in recordsOrDict)

    @staticmethod
    def _record_iterator(recordsOrDict):
        if isinstance(recordsOrDict, dict):
            return ((key + (val,)) for key, val in recordsOrDict.items())
        else:
            return recordsOrDict

    def __init__(self, recordsOrDict=(), addValues=False, freeze=True):
        """Construct a :class:`RecordTree`.

        :param recordsOrDict:
            Data stored in the tree.
        :type recordsOrDict:
            dict or list(tuple)

        :param addValues:
            Add the (numeric) values of records stored in the same place in the
            tree, instead of replacing the value. If this is exactly a list of
            path tuples (precise types are required), then add values only for
            records below any of these paths instead. In either case, resulting
            values of zero are not stored in the tree.
        :type addValues:
            bool or list(tuple)

        :param bool freeze:
            Whether to transform mutable labels and values into hashable ones.
        """
        self._tree = self._NodeDict()

        if isinstance(addValues, list):
            addValues = [self._flatten(path) for path in addValues]

        def _add_values_at(path):
            if isinstance(addValues, list):
                return any(path[:end] in addValues for end in range(len(path)))
            else:
                return bool(addValues)

        for path, value in self._keyval_iterator(recordsOrDict):
            path = self._flatten(path)
            node = self._tree

            if freeze:
                path = tuple(self._freeze(thing) for thing in path)
                value = self._freeze(value)

            if value is not self.NONE and _add_values_at(path):
                if value == 0:
                    # Do not add a value equal to zero.
                    continue
                elif path in self:
                    oldValue = self[path]

                    if isinstance(oldValue, RecordTree):
                        raise LookupError("Can't add value '{}' at '{}': Path "
                            "leads to an inner node.".format(value, path))

                    value = oldValue + value

                    # If the sum is zero, delete the record instead.
                    if value == 0:
                        value = self.NONE

            if value is self.NONE:
                # Handle deletion of a subtree.
                clearNodes  = [node]
                clearLabels = []
                for label in path:
                    if label in node:
                        node = node[label]
                        clearNodes.insert(0, node)
                        clearLabels.insert(0, label)
                    else:
                        clearNodes = None
                        break

                if not clearNodes:
                    continue

                clearNodes.pop(0)
                for childLabel, node in zip(clearLabels, clearNodes):
                    node.pop(childLabel)
                    if node:
                        break
            else:
                # Handle insertion of a leaf (may replace a subtree).
                for label in path[:-1]:
                    node.setdefault(label, self._NodeDict())
                    node = node[label]

                    if not isinstance(node, self._NodeDict):
                        raise LookupError("Can't set value '{}' at '{}': Path "
                            "already contains a leaf.".format(value, path))

                node[path[-1]] = value

        self._hash = hash(self.set)

    @classmethod
    def _traverse(cls, node):
        if not isinstance(node, cls._NodeDict):
            # Not a node but a value.
            yield (node,)
            return
        elif not node:
            # Empty tree.
            return

        for label, child in node.items():
            for labels in cls._traverse(child):
                yield (label,) + labels

    @property
    def records(self):
        """Return an iterator over tuples, each representing one record."""
        return self._traverse(self._tree)

    @property
    def items(self):
        """Return an iterator over path/value pairs representing records."""
        return ((path[:-1], path[-1]) for path in self.records)

    @property
    def paths(self):
        """Return an iterator over paths, each representing one record."""
        return (path[:-1] for path in self.records)

    @cached_property
    def dict(self):
        """Return the tree as a read-only, tuple-indexed dictionary view.

        Every key/value pair of the returned dictionary is a record.
        """
        return MappingProxyType({path[:-1]: path[-1] for path in self.records})

    @cached_property
    def set(self):
        """Return a frozen set of tuples, each representing one record."""
        return frozenset(self.records)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __bool__(self):
        return bool(self._tree)

    def __len__(self):
        # TODO: Compute this on initialization.
        return len(list(self.records))

    def __contains__(self, path):
        if not isinstance(path, tuple):
            raise TypeError("{} indices must be tuples.".format(
                self.__class__.__name__))

        node = self._tree
        for label in path:
            if not isinstance(node, self._NodeDict) or label not in node:
                return False
            node = node[label]

        return True

    def _get(self, path, errorOnBadPath):
        # try:
        #     iter(path)
        # except TypeError:
        #     path = (path,)
        if not isinstance(path, tuple) and not isinstance(path, list):
            path = (path,)

        node = self._tree
        for label in path:
            if not isinstance(node, self._NodeDict) or label not in node:
                if errorOnBadPath:
                    raise LookupError(str(path))
                else:
                    return RecordTree()
            node = node[label]

        if isinstance(node, self._NodeDict):
            return RecordTree(
                {path[:-1]: path[-1] for path in self._traverse(node)})
        else:
            return node

    def __getitem__(self, path):
        return self._get(path, True)

    def get(self, path):
        """Return an empty :class:`RecordTree` if the path does not exist."""
        return self._get(path, False)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.dict)

    def __str__(self):
        return str(self.dict)

    def __le__(self, other):
        """Perform entrywise lower-or-equal-than comparison.

        Each left hand side path must be present on the right hand side, and the
        associated left hand side value must compare lower-or-equal-than the
        right hand side value.
        """
        if type(self) != type(other):
            return NotImplemented

        for path, value in self.items:
            if path not in other:
                return False

            if not value <= other[path]:
                return False

        return True

    def __ge__(self, other):
        """Perform entrywise greater-or-equal-than comparison.

        Each left hand side path must be present on the right hand side, and the
        associated left hand side value must compare greater-or-equal-than the
        right hand side value.
        """
        if type(self) != type(other):
            return NotImplemented

        for path, value in self.items:
            if path not in other:
                return False

            if not value >= other[path]:
                return False

        return True

    def __lshift__(self, other):
        """Perform subtree comparison.

        Each left hand side path must be present on the right hand side. If the
        special :class:`ALL` type is found as a value in the right hand side
        tree, it is treated as "all possible subtrees". All other values are not
        considered.
        """
        if not isinstance(other, RecordTree):
            return NotImplemented

        for path in self.paths:
            lhsNode = self._tree
            rhsNode = other._tree

            for label in path:
                if rhsNode is self.ALL:
                    break

                if label not in rhsNode:
                    return False

                assert label in lhsNode

                lhsNode = lhsNode[label]
                rhsNode = rhsNode[label]

        return True

    def mismatch(self, other):
        """A subtree of ``self`` that renders ``self << other`` :obj:`False`.

        :returns RecordTree:
            The smallest subtree ``T`` of ``self`` such that ``self`` without
            the records in ``T`` is a subtree of ``other``. The returned tree is
            a direct instance of the :class:`RecordTree` base class.
        """
        if not isinstance(other, RecordTree):
            raise TypeError("The argument must be another record tree.")

        records = []

        for record in self.records:
            lhsNode = self._tree
            rhsNode = other._tree

            for label in record[:-1]:
                if rhsNode is self.ALL:
                    break

                if label not in rhsNode:
                    records.append(record)
                    break

                assert label in lhsNode

                lhsNode = lhsNode[label]
                rhsNode = rhsNode[label]

        return RecordTree(records)

    @staticmethod
    def _str(thing):
        return thing.__name__ if hasattr(thing, "__name__") else str(thing)

    @property
    def text(self):
        """Return the full tree as a multiline string."""
        keys, vals = [], []
        for path in self._traverse(self._tree):
            keys.append("/".join(self._str(label) for label in path[:-1]))
            vals.append(self._str(path[-1]))
        if not keys:
            return "Empty {} instance.".format(self.__class__.__name__)
        keyLen = max(len(key) for key in keys)
        valLen = max(len(val) for val in vals)
        return "\n".join(sorted(
            "{{:{}}} = {{:{}}}".format(keyLen, valLen).format(key, val)
            for key, val in zip(keys, vals)))

    def copy(self):
        """Create a shallow copy; the tree is copied, the values are not."""
        return self.__class__(self.records)

    def updated(self, recordsOrDict, addValues=False):
        """Create a shallow copy with modified records.

        :Example:

        >>> from picos.modeling.footprint import RecordTree as T
        >>> t = T({(1, 1, 1): 3, (1, 1, 2): 4, (1, 2, 1): 5}); t
        RecordTree({(1, 1, 1): 3, (1, 1, 2): 4, (1, 2, 1): 5})
        >>> t.updated({(1, 1, 1): "a", (2, 2): "b"}) # Change or add a record.
        RecordTree({(1, 1, 1): 'a', (1, 1, 2): 4, (1, 2, 1): 5, (2, 2): 'b'})
        >>> t.updated({(1,1,1): T.NONE}) # Delete a single record.
        RecordTree({(1, 1, 2): 4, (1, 2, 1): 5})
        >>> t.updated({(1,1): T.NONE}) # Delete multiple records.
        RecordTree({(1, 2, 1): 5})
        >>> t.updated([(1, 1, 1, T.NONE), (1, 1, 1, 1, 6)]) # Delete, then add.
        RecordTree({(1, 1, 2): 4, (1, 1, 1, 1): 6, (1, 2, 1): 5})
        >>> try: # Not possible to implicitly turn a leaf into an inner node.
        ...     t.updated([(1, 1, 1, 1, 6)])
        ... except LookupError as error:
        ...     print(error)
        Can't set value '6' at '(1, 1, 1, 1)': Path already contains a leaf.
        """
        return self.__class__(
            chain(self.records, self._record_iterator(recordsOrDict)),
            addValues)


# --------------------------------------
__all__ = api_end(_API_START, globals())
