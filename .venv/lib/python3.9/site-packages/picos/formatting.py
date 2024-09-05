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

"""Console output helpers related to formatting."""

import re
import sys
from contextlib import contextmanager
from math import ceil, floor

from . import glyphs
from .apidoc import api_end, api_start

_API_START = api_start(globals())
# -------------------------------


HEADER_WIDTH = 35  #: Character length of headers and footers printed by PICOS.


def print_header(title, subtitle=None, symbol="-", width=HEADER_WIDTH):
    """Print a header line."""
    w = "{:d}".format(width)

    print(("{0}\n{1:^"+w+"}\n{2}{0}").format(symbol * width, title,
        ("{:^"+w+"}\n").format("{}".format(subtitle)) if subtitle else ""))
    sys.stdout.flush()


def print_footer(caption, symbol="=", width=HEADER_WIDTH):
    """Print a footer line."""
    middle = "[ {} ]".format(caption)

    if width < len(middle) + 2:
        footer = symbol * width
    else:
        s = (width - len(middle))
        l = int(floor(s / 2.0))
        r = int(ceil(s / 2.0))
        footer = symbol * l + middle + symbol * r

    print(footer)
    sys.stdout.flush()


@contextmanager
def box(title, caption, subtitle=None, symbol="-", width=HEADER_WIDTH,
        show=True):
    """Print both a header above and a footer below the context.

    :param str title: The (long) main title printed at the top.
    :param str caption: The (short) caption printed at the bottom.
    :param str subtitle: The (long) subtitle printed at the top.
    :param str symbol: A single character used to draw lines.
    :param int width: The width of the box.
    :param bool show: Whether anything should be printed at all.
    """
    if show:
        print_header(title, subtitle, symbol, width)
        yield
        print_footer(caption, symbol, width)
    else:
        yield


@contextmanager
def picos_box(show=True):
    """Print a PICOS header above and a PICOS footer below the context."""
    from .__init__ import __version__ as picosVer
    with box("PICOS {}".format(picosVer), "PICOS", symbol="=", show=show):
        yield


@contextmanager
def solver_box(longName, shortName, subSolver=None, show=True):
    """Print a solver header above and a solver footer below the context."""
    subtitle = "via {}".format(subSolver) if subSolver else None
    with box(longName, shortName, subtitle, show=show):
        yield


def doc_cat(docstring, append):
    """Append to a docstring."""
    if not docstring:
        raise ValueError("Empty base docstring.")

    if not append:
        return docstring

    lines = [line for line in docstring.splitlines() if line.strip()]
    i = len(lines[1]) - len(lines[1].lstrip()) if len(lines) > 1 else 0

    append = "\n".join(" "*i + line for line in append.splitlines())

    return docstring.rstrip() + "\n\n" + append


def detect_range(sequence, asQuadruple=False, asStringTemplate=False,
        shortString=False):
    """Return a Python range mirroring the given integer sequence.

    :param sequence: An integer sequence that can be mirrored by a Python range.
    :param bool asQuadruple: Whether to return a quadruple with factor, inner
        shift, outer shift, and length, formally ``(a, i, o, n)`` such that
        ``[a*(x+i)+o for x in range(n)]`` mirrors the input sequence.
    :param bool asStringTemplate: Whether to return a format string that, if
        instanciated with numbers from ``0`` to ``len(sequence) - 1``, yields
        math expression strings that describe the input sequence members.
    :param bool shortString: Whether to return condensed string templates that
        are designed to be instanciated with an index character string. Requires
        asStringTemplate to be ``True``.
    :raises TypeError: If the input is not an integer sequence.
    :raises ValueError: If the input cannot be mirrored by a Python range.
    :returns: A range object, a quadruple of numbers, or a format string.

    :Example:

    >>> from picos.formatting import detect_range as dr
    >>> R = range(7,30,5)
    >>> S = list(R)
    >>> S
    [7, 12, 17, 22, 27]
    >>> # By default, returns a matching range object:
    >>> dr(S)
    range(7, 28, 5)
    >>> dr(S) == R
    True
    >>> # Sequence elements can also be decomposed w.r.t. range(len(S)):
    >>> a, i, o, n = dr(S, asQuadruple=True)
    >>> [a*(x+i)+o for x in range(n)] == S
    True
    >>> # The same decomposition can be returned in a string representation:
    >>> dr(S, asStringTemplate=True)
    '5·({} + 1) + 2'
    >>> # Short string representations are designed to accept index names:
    >>> dr(S, asStringTemplate=True, shortString=True).format("i")
    '5(i+1)+2'
    >>> dr(range(0,100,5), asStringTemplate=True, shortString=True).format("i")
    '5i'
    >>> dr(range(10,100), asStringTemplate=True, shortString=True).format("i")
    'i+10'

    :Example:

    >>> # This works with decreasing ranges as well.
    >>> R2 = range(10,4,-2)
    >>> S2 = list(R2)
    >>> S2
    [10, 8, 6]
    >>> dr(S2)
    range(10, 5, -2)
    >>> dr(S2) == R2
    True
    >>> a, i, o, n = dr(S2, asQuadruple=True)
    >>> [a*(x+i)+o for x in range(n)] == S2
    True
    >>> T = dr(S2, asStringTemplate=True, shortString=True)
    >>> [T.format(i) for i in range(len(S2))]
    ['-2(0-5)', '-2(1-5)', '-2(2-5)']
    """
    if asQuadruple and asStringTemplate:
        raise ValueError(
            "Can only return a quadruple or a string template, not both.")

    if shortString and not asStringTemplate:
        raise ValueError("Enabling 'shortString' requires 'asStringTemplate'.")

    if len(sequence) == 0:
        if asQuadruple:
            return 0, 0, 0, 0
        elif asStringTemplate:
            return ""
        else:
            return range(0)

    first  = sequence[0]
    last   = sequence[-1]
    next   = last + (1 if first <= last else -1)
    length = len(sequence)

    if not isinstance(first, int) or not isinstance(last, int):
        raise TypeError("Not an integer container.")

    # Determine potential integer step size.
    if length > 1:
        step = (last - first) / (length - 1)
    else:
        step = 1
    if int(step) != step:
        raise ValueError("Cannot be mirrored by a Python range.")
    step = int(step)

    # Determine potential range.
    range_ = range(first, next, step)

    if len(range_) != len(sequence):
        raise ValueError("Cannot be mirrored by a Python range.")

    for position, number in enumerate(range_):
        if sequence[position] != number:
            raise ValueError("Cannot be mirrored by a Python range.")

    if asQuadruple or asStringTemplate:
        # Compute inner and outer shift.
        innerShift = first // step
        outerShift = first  % step

        # Verify our finding.
        assert last // step + 1 - innerShift                 == length
        assert step*(0 + innerShift) + outerShift            == first
        assert step*((length - 1) + innerShift) + outerShift == last

        if asQuadruple:
            return step, innerShift, outerShift, length
        elif shortString:
            string = "{{}}{:+d}".format(innerShift) if innerShift else "{}"
            if step != 1 and innerShift:
                string = "{}({})".format("-" if step == -1 else step, string)
            elif step != 1:
                string = "{}{}".format("-" if step == -1 else step, string)
            string = "{}{:+d}".format(string, outerShift) \
                if outerShift else string
            # TODO: Something like the following is needed in case replacement
            #       happens for a factor in a multiplication.
            # if (innerShift and step == 1) or outerShift:
            #     string = "({})".format(string)

            return string
        else:
            glyph = glyphs.add("{}", innerShift) if innerShift else "{}"
            glyph = glyphs.mul(step, glyph) if step != 1 else glyph
            glyph = glyphs.add(glyph, outerShift) if outerShift else glyph

            return glyph
    else:
        return range_


def natsorted(strings, key=None):
    """Sort the given list of strings naturally with respect to numbers."""
    def natsplit(string):
        parts = re.split(r"([+-]?\d*\.?\d+)", string)
        split = []
        for part in parts:
            try:
                split.append(float(part))
            except ValueError:
                split.append(part)
        return split

    if key:
        return sorted(strings, key=lambda x: natsplit(key(x)))
    else:
        return sorted(strings, key=natsplit)


def parameterized_string(
        strings, replace=r"-?\d+", context=r"\W", placeholders="ijklpqr",
        fallback="?"):
    """Find a string template for the given (algebraic) strings.

    Given a list of strings with similar structure, finds a single string with
    placeholders and an expression that denotes how to instantiate the
    placeholders in order to obtain each string in the list.

    The function is designed to take a number of symbolic string representations
    of math expressions that differ only with respect to indices.

    :param list strings:
        The list of strings to compare.
    :param str replace:
        A regular expression describing the bits to replace with placeholders.
    :param str context:
        A regular expression describing context characters that need to be
        present next to the bits to be replaced with placeholders.
    :param placeholders:
        An iterable of placeholder strings. Usually a string, so that each of
        its characters becomes a placeholder.
    :param str fallback:
        A fallback placeholder string, if the given placeholders are not
        sufficient.

    :returns:
        A tuple of two strings, the first being the template string and the
        second being a description of the placeholders used.

    :Example:

    >>> from picos.formatting import parameterized_string as ps
    >>> ps(["A[{}]".format(i) for i in range(5, 31)])
    ('A[i+5]', 'i ∈ [0…25]')
    >>> ps(["A[{}]".format(i) for i in range(5, 31, 5)])
    ('A[5(i+1)]', 'i ∈ [0…5]')
    >>> S=["A[0]·B[2]·C[3]·D[5]·F[0]",
    ...    "A[1]·B[1]·C[6]·D[6]·F[0]",
    ...    "A[2]·B[0]·C[9]·D[9]·F[0]"]
    >>> ps(S)
    ('A[i]·B[-(i-2)]·C[3(i+1)]·D[j]·F[0]', '(i,j) ∈ zip([0…2],[5,6,9])')
    """
    if len(strings) == 0:
        return "", ""
    elif len(strings) == 1:
        return strings[0], ""

    for string in strings:
        if not isinstance(string, str):
            raise TypeError("First argument must be a list of strings.")

    def split_with_context(string, match, context):
        pattern = r"(?<={0})?{1}(?={0})?".format(context, match)
        return re.split(pattern, string)

    def findall_with_context(string, match, context):
        pattern = r"(?<={0})?{1}(?={0})?".format(context, match)
        return re.findall(pattern, string)

    # The skeleton of a string is the part not matched by 'replace' and
    # surrounded by 'context', and it must be the same for all strings.
    skeleton = split_with_context(strings[0], replace, context)
    for string in strings[1:]:
        if skeleton != split_with_context(string, replace, context):
            raise ValueError("Strings do not have a common skeleton.")

    # The slots are the parts that are matched by 'replace' and surrounded by
    # 'context' and should be filled  with the placeholders.
    slotToValues = []
    for string in strings:
        slotToValues.append(findall_with_context(string, replace, context))
    slotToValues = list(zip(*slotToValues))

    # Verify that slots are always surrounded by (empty) skeleton strings.
    assert len(skeleton) == len(slotToValues) + 1

    # Find slots with the same value in each string; add them to the skeleton.
    for slot in range(len(slotToValues)):
        if len(set(slotToValues[slot])) == 1:
            skeleton[slot + 1] =\
                skeleton[slot] + slotToValues[slot][0] + skeleton[slot + 1]
            skeleton[slot] = None
            slotToValues[slot] = None
    skeleton     = [s for s in skeleton     if s is not None]
    slotToValues = [v for v in slotToValues if v is not None]

    # We next build a mapping from slots to (few) placeholder indices.
    slotToIndex = {}
    nextIndex = 0

    # Find slots whose values form a range, and build string templates that lift
    # a placeholder to a formula denoting sequence elements (e.g. "i" → "2i+1").
    # All such slots share the first placeholder (with index 0).
    slotToTemplate = {}
    for slot, values in enumerate(slotToValues):
        try:
            slotToTemplate[slot] = detect_range([int(v) for v in values],
                asStringTemplate=True, shortString=True)
        except ValueError:
            pass
        else:
            slotToIndex[slot] = 0
            nextIndex = 1

    # Find slots with identical value in each string and assign them the same
    # placeholder.
    valsToIndex = {}
    for slot, values in enumerate(slotToValues):
        if slot in slotToIndex:
            # The slot holds a range.
            continue

        if values in valsToIndex:
            slotToIndex[slot] = valsToIndex[values]
        else:
            slotToIndex[slot]   = nextIndex
            valsToIndex[values] = nextIndex
            nextIndex += 1

    # Define a function that maps slots to their placeholder symbols.
    def placeholder(slot):
        index = slotToIndex[slot]
        return placeholders[index] if index < len(placeholders) else fallback

    # Assemble the string template (with values replaced by placeholders).
    template = ""
    for slot in range(len(slotToIndex)):
        if slot in slotToTemplate:
            ph = slotToTemplate[slot].format(placeholder(slot))
        else:
            ph = placeholder(slot)

        template += skeleton[slot] + ph
    template += skeleton[-1]

    # Collect the placeholdes that were used, and their domains.
    usedPHs, domains = [], []
    indices = set()
    for slot, index in slotToIndex.items():
        values = slotToValues[slot]

        if index in indices:
            continue
        else:
            indices.add(index)

        usedPHs.append(placeholder(slot))

        if slot in slotToTemplate:
            domains.append(glyphs.intrange(0, len(values) - 1))
        elif len(values) > 4:
            domains.append(glyphs.intrange(
                ",".join(values[:2]) + ",", "," + ",".join(values[-2:])))
        else:
            domains.append(glyphs.interval(",".join(values)))

    # Make sure used placeholders and their domains match in number.
    assert len(usedPHs) == len(domains)

    # Assemble occuring placeholders and ther joint domain (the data).
    if len(domains) == 0:
        data = ""
    else:
        if len(domains) == 1:
            usedPHs = usedPHs[0]
            domain  = domains[0]
        else:
            usedPHs = "({})".format(",".join(usedPHs))
            domain  = "zip({})".format(",".join(domains))

        data = glyphs.element(usedPHs, domain)

    return template, data


def arguments(strings, sep=", ", empty=""):
    """A wrapper around :func:`parameterized_string` for argument lists.

    :param list(str) strings:
        String descriptions of the arguments.

    :param str sep:
        Separator.

    :param str empty:
        Representation of an empty argument list.
    """
    if len(strings) == 0:
        return empty
    elif len(strings) == 1:
        return strings[0]
    elif len(strings) == 2:
        return strings[0] + sep + strings[1]

    try:
        template, data = parameterized_string(strings)
    except ValueError:
        pass
    else:
        if data:
            return glyphs.sep(template, data)

    return glyphs.fromto(strings[0] + sep, sep + strings[-1])


# --------------------------------------
__all__ = api_end(_API_START, globals())
