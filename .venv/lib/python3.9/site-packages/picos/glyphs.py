# ------------------------------------------------------------------------------
# Copyright (C) 2018 Maximilian Stahlberg
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

"""String templates used to print (algebraic) expressions.

PICOS internally uses this module to produce string representations for the
algebraic expressions that you create.
The function-like objects that are used to build such strings are called
"glyphs" and are instanciated by this module following the
`singleton pattern <https://en.wikipedia.org/wiki/Singleton_pattern>`_.
As a result, you can modify the glyph objects listed below to influence how
PICOS will format future strings, for example to disable use of unicode symbols
that your console does not suppport or to adapt PICOS' output to the rest of
your application.

Here's an example of first swapping the entire character set to display
expressions using only `Latin-1 <https://en.wikipedia.org/wiki/ISO/IEC_8859-1>`_
characters, and then modifying a single glyph to our liking:

  >>> import picos
  >>> X = picos.Problem().add_variable("X", (2,2), "symmetric")
  >>> print(X >> 0)
  X ≽ 0
  >>> picos.glyphs.latin1()
  >>> print(X >> 0)
  X » 0
  >>> picos.glyphs.psdge.template = "{} - {} is psd"
  >>> print(X >> 0)
  X - 0 is psd

Note that glyphs understand some algebraic rules such as operator precedence
and associativity. This is possible because strings produced by glyphs remember
how they were created.

  >>> one_plus_two = picos.glyphs.add(1, 2)
  >>> one_plus_two
  '1 + 2'
  >>> one_plus_two.glyph.template, one_plus_two.operands
  ('{} + {}', (1, 2))
  >>> picos.glyphs.add(one_plus_two, 3)
  '1 + 2 + 3'
  >>> picos.glyphs.sub(0, one_plus_two)
  '0 - (1 + 2)'

The positive semidefinite glyph above does not yet know how to properly handle
arguments with respect to the ``-`` symbol involved, but we can modify it
further:

  >>> print(X + X >> X + X)
  X + X - X + X is psd
  >>> # Use the same operator binding strength as regular substraction.
  >>> picos.glyphs.psdge.order = picos.glyphs.sub.order
  >>> print(X + X >> X + X)
  X + X - (X + X) is psd

You can reset all glyphs to their initial state as follows:

  >>> picos.glyphs.default()
"""

import functools
import sys

from . import settings
from .apidoc import api_end, api_start

# Allow functions to modify this module directly.
glyphs = sys.modules[__name__]

_API_START = api_start(globals())
# -------------------------------


# Definitions of glyph classes and the rich strings they create.
# --------------------------------------------------------------

class GlStr(str):
    """A string created from a :class:`glyph <Gl>`.

    It has an additional :attr:`glyph` field pointing to the glyph that created
    it, and a :attr:`operands` field containing the values used to create it.
    """

    def __new__(cls, string, glyph, operands):
        """Create a regular Python string."""
        return str.__new__(cls, string)

    def __init__(self, string, glyph, operands):
        """Augment the Python string with metadata on its origin."""
        self.glyph = glyph
        """The glyph used to create the string."""

        self.operands = operands
        """The operands used to create the string."""

    def __copy__(self):
        return self.__class__(str(self), self.glyph, self.operands)

    def reglyphed(self, replace={}):
        """Returns a rebuilt version of the string using current glyphs.

        :param dict replace:
            Replace leaf-node (non :class:`GlStr`) strings with new strings.
            This can be used, for instance, to change the names of varaibles.
        """
        return self.glyph(*(op.reglyphed(replace) if isinstance(op, GlStr) else
            (replace[op] if op in replace else op) for op in self.operands))


class Gl:
    """The basic "glyph", an (algebraic) string formatting template.

    Sublcasses are supposed to extend formatting routines, going beyond of what
    Python string formatting is capabale of. In particular, glyphs can be used
    to craft unambiguous algebraic expressions with the minimum amount of
    parenthesis.
    """

    def __init__(self, glyph):
        """Construct a glyph.

        :param str glyph: The glyph's format string template.
        """
        self.template = glyph
        self.initial  = glyph

    def reset(self):
        """Reset the glyph to its initial formatting template."""
        self.template = self.initial

    def update(self, new):
        """Change the glyph's formatting template."""
        self.template = new.template

    def rebuild(self):
        """If the template was created using other glyphs, rebuild it.

        :returns: True if the template has changed.
        """
        if isinstance(self.template, GlStr):
            oldTemplate = self.template
            self.template = self.template.reglyphed()
            return self.template != oldTemplate
        else:
            return False

    def __call__(self, *args):
        """Format the arguments as a :class:`GlStr`."""
        return GlStr(self.template.format(*args), self, args)


class OpStr(GlStr):
    """A string created from a math operator glyph."""

    pass


class Op(Gl):
    """The basic math operator glyph."""

    def __init__(self, glyph, order, assoc=False, closed=False):
        """Construct a math operator glyph.

        :param str glyph: The glyph's format string template.
        :param int order: The operator's position in the binding strength
            hierarchy. Operators with lower numbersbind more strongly.
        :param bool assoc: If this is :obj:`True`, then the operator is
            associative, so that parenthesis are always omitted around operands
            with an equal outer operator. Otherwise, (1) parenthesis are used
            around the right hand side operand of a binary operation of same
            binding strength and (2) around all operands of non-binary
            operations of same binding strength.
        :param closed: If :obj:`True`, the operator already encloses the
            operands in some sort of brackets, so that no additional parenthesis
            are needed. For glyphs where only some operands are enclosed, this
            can be specified per operand in the form of a list.
        :type closed: bool or list(bool)
        """
        self.initial = (glyph, order, assoc, closed)
        self.reset()

    def reset(self):
        """Reset the glyph to its initial behavior."""
        self.template, self.order, self.assoc, self.closed = self.initial

    def update(self, new):
        """Change the glyph's behavior."""
        self.template = new.template
        self.order    = new.order
        self.assoc    = new.assoc
        self.closed   = new.closed

    def __call__(self, *operands):
        """Format the arguments as an :class:`OpStr`."""
        if self.closed is True:
            return OpStr(self.template.format(*operands), self, operands)

        placeholders = []
        for i, operand in enumerate(operands):
            if isinstance(self.closed, list) and i < len(self.closed) \
            and self.closed[i]:
                parenthesis = False
            elif not isinstance(operand, OpStr):
                parenthesis = False
            elif operand.glyph.order < self.order:
                parenthesis = False
            elif operand.glyph.order == self.order:
                if len(operands) == 2 and i == 0:
                    # By default, bind from left to right.
                    parenthesis = False
                elif self.assoc in (None, False):
                    parenthesis = True
                else:
                    parenthesis = operand.glyph is not self
            else:
                parenthesis = True

            if type(operand) is float:
                # If no format specifier was given, then integral floats would
                # be formatted with a trailing '.0', which we don't want. Note
                # that for complex numbers the default behavior is already as we
                # want it, while 'g' would omit the parenthesis that we need.
                placeholder = "{:g}"
            else:
                placeholder = "{}"

            if parenthesis:
                placeholders.append(glyphs.parenth(placeholder))
            else:
                placeholders.append(placeholder)

        return OpStr(self.template.format(*placeholders).format(*operands),
            self, operands)


class Am(Op):
    """A math atom glyph."""

    def __init__(self, glyph):
        """Construct an :class:`Am` glyph.

        :param str glyph: The glyph's format string template.
        """
        Op.__init__(self, glyph, 0)


class Br(Op):
    """A math operator glyph with enclosing brackets."""

    def __init__(self, glyph):
        """Construct a :class:`Br` glyph.

        :param str glyph: The glyph's format string template.
        """
        Op.__init__(self, glyph, 0, closed=True)


class Fn(Op):
    """A math operator glyph in function form."""

    def __init__(self, glyph):
        """Construct a :class:`Fn` glyph.

        :param str glyph: The glyph's format string template.
        """
        Op.__init__(self, glyph, 0, closed=True)


class Tr(Op):
    """A math glyph in superscript/trailer form."""

    def __init__(self, glyph):
        """Construct a :class:`Tr` glyph.

        :param str glyph: The glyph's format string template.
        """
        Op.__init__(self, glyph, 1)


class Rl(Op):
    """A math relation glyph."""

    def __init__(self, glyph):
        """Construct a :class:`Rl` glyph.

        :param str glyph: The glyph's format string template.
        """
        Op.__init__(self, glyph, 5, assoc=True)


# Functions that show, reset or rebuild the glyph objects.
# --------------------------------------------------------

def show(*args):
    """Show output from all glyphs.

    :param list(str) args: Strings to use as glyph operands.
    """
    args = list(args) + ["{}"]*4

    print("{:8} | {:3} | {:5} | {}\n{}+{}+{}+{}".format(
        "Glyph", "Pri", "Asso", "Value", "-"*9, "-"*5, "-"*7, "-"*10))

    for name in sorted(list(glyphs.__dict__.keys())):
        glyph = getattr(glyphs, name)
        if isinstance(glyph, Gl):
            order = glyph.order if hasattr(glyph, "order") else ""
            assoc = str(glyph.assoc) if hasattr(glyph, "order") else ""
            print("{:8} | {:3} | {:5} | {}".format(
                name, order, assoc, glyph(*args)))


def rebuild():
    """Update glyphs that are based upon other glyphs."""
    for i in range(100):
        if not any(glyph.rebuild() for glyph in glyphs.__dict__.values()
                if isinstance(glyph, Gl)):
            return

    raise Exception("Maximum recursion depth for glyph rebuilding reached. "
        "There is likely a cyclic dependence between them.")


def ascii():
    """Let PICOS create future strings using only ASCII characters."""
    for glyph in glyphs.__dict__.values():
        if isinstance(glyph, Gl):
            glyph.reset()


# Initial glyph definitions and functions that update them.
# ---------------------------------------------------------

# Non-operator glyphs.
repr1    = Gl("<{}>"); """Representation glyph."""
repr2    = Gl(glyphs.repr1("{}: {}")); """Long representation glyph."""
parenth  = Gl("({})"); """Parenthesis glyph."""
sep      = Gl("{} : {}"); """Seperator glyph."""
compsep  = Gl("{}:{}"); """Compact seperator glyph."""
comma    = Gl("{}, {}"); """Seperated by comma glyph."""
size     = Gl("{}x{}"); """Matrix size/shape glyph."""
compose  = Gl("{}.{}"); """Function composition glyph."""
set      = Gl("{{{}}}"); """Set glyph."""
closure  = Fn("cl{}"); """Set closure glyph."""
interval = Gl("[{}]"); """Interval glyph."""
fromto   = Gl("{}..{}"); """Range glyph."""
intrange = Gl(glyphs.interval(glyphs.fromto("{}", "{}")))
"""Integer range glyph."""
shortint = Gl(glyphs.interval(glyphs.fromto("{},", ",{}")))
"""Shortened interval glyph."""
forall   = Gl("{} f.a. {}"); """Universal quantification glyph."""
leadsto  = Gl("{} -> {}"); """Successorship glyph."""
and_     = Gl("{} and {}"); """Logical and glyph."""
or_      = Gl("{} or {}"); """Logical or glyph."""

# Atomic glyphs.
infty    = Am("inf"); """Infinity glyph."""
idmatrix = Am("I"); """Identity matrix glyph."""
lambda_  = Am("lambda"); """Lambda symbol glyph."""

# Bracketed glyphs.
matrix   = Br("[{}]"); """Matrix glyph."""
dotp     = Br("<{}, {}>"); """Scalar product glyph."""
abs      = Br("|{}|"); """Absolute value glyph."""
norm     = Br("||{}||"); """Norm glyph."""

# Special norms.
pnorm    = Op(Gl("{}_{}")(glyphs.norm("{}"), "{}"), 1, closed=[True, False])
"""p-Norm glyph."""
pqnorm   = Op(Gl("{}_{},{}")(glyphs.norm("{}"), "{}", "{}"), 1,
    closed=[True, False, False]); """pq-Norm glyph."""
spnorm   = Op(Gl("{}_{}")(glyphs.norm("{}"), "2"), 1, closed=True)
"""Spectral Norm glyph."""
ncnorm   = Op(Gl("{}_{}")(glyphs.norm("{}"), "*"), 1, closed=True)
"""Nuclear Norm glyph."""

# Function glyphs.
sum      = Fn("sum({})");   """Summation glyph."""
prod     = Fn("prod({})");   """Product glyph."""
max      = Fn("max({})");   """Maximum glyph."""
min      = Fn("min({})");   """Minimum glyph."""
exp      = Fn("exp({})");   """Exponentiation glyph."""
log      = Fn("log({})");   """Logarithm glyph."""
vec      = Fn("vec({})");   """Vectorization glyph."""
trilvec  = Fn("trilvec({})");  """Lower triangular vectorization glyph."""
triuvec  = Fn("triuvec({})");  """Upper triangular vectorization glyph."""
svec     = Fn("svec({})");  """Symmetric vectorization glyph."""
desvec   = Fn("desvec({})");   """Symmetric de-vectorization glyph."""
trace    = Fn("tr({})");    """Matrix trace glyph."""
diag     = Fn("diag({})");  """Diagonal matrix glyph."""
maindiag = Fn("maindiag({})"); """Main diagonal glyph."""
det      = Fn("det({})");   """Determinant glyph."""
real     = Fn("Re({})");    """Real part glyph."""
imag     = Fn("Im({})");    """Imaginary part glyph."""
conj     = Fn("conj({})");  """Complex conugate glyph."""
quadpart = Fn("quad({})");  """Quadratic part glyph."""
affpart  = Fn("aff({})");   """Affine part glyph."""
linpart  = Fn("lin({})");   """Linear part glyph."""
blinpart = Fn("bilin({})"); """Bilinear part glyph."""
ncstpart = Fn("noncst({})"); """Nonconstant part glyph."""
cstpart  = Fn("cst({})");   """Constant part glyph."""
frozen   = Fn("[{}]");      """Frozen mutables glyph."""
reshaped = Fn("reshaped({}, {})"); """Column-major (Fortran) reshaped glyph."""
reshaprm = Fn("reshaped({}, {}, C)"); """Row-major (C-order) reshaped glyph."""
bcasted  = Fn("bcasted({}, {})"); """Broadcasted glyph."""
sqrt     = Fn("sqrt({})");  """Square root glyph."""
shuffled = Fn("shuffled({})"); """Matrix reshuffling glyph."""
probdist = Fn("pd({})");    """Probability distribution glyph."""
expected = Fn("E[{}]");     """Epected value glyph."""

# Semi-closed glyphs.
ptrace   = Op("trace_{}({})", 0, closed=[False, True])
"""Matrix p-Trace glyph."""
slice    = Op("{}[{}]", 0, closed=[False, True])
"""Expression slicing glyph."""
ptransp_ = Op("{}.{{{{{}}}}}", 1, closed=[False, True])
"""Matrix partial transposition glyph."""  # TODO: Replace ptransp.
ptrace_  = Op("{}.{{{{{}}}}}", 1, closed=[False, True])
"""Matrix partial trace glyph."""  # TODO: Replace ptrace_.
exparg   = Op("E_{}({})", 0, closed=[False, True])
"""Expected value glyph."""

# Basic algebraic glyphs.
add      = Op("{} + {}", 3, assoc=True);  """Addition glyph."""
sub      = Op("{} - {}", 3, assoc=False); """Substraction glyph."""
hadamard = Op("{}(o){}", 2, assoc=True);  """Hadamard product glyph."""
kron     = Op("{}(x){}", 2, assoc=True);  """Kronecker product glyph."""
mul      = Op("{}*{}",   2, assoc=True);  """Multiplication glyph."""
div      = Op("{}/{}",   2, assoc=False); """Division glyph."""
neg      = Op("-{}",     2.5);              """Negation glyph."""
plsmns   = Op("[+/-]{}", 2.5);              """Plus/Minus glyph."""

# Trailer glyphs.
power    = Tr("{}^{}"); """Power glyph."""
cubed    = Tr(glyphs.power("{}", "3")); """Cubed value glyph."""
squared  = Tr(glyphs.power("{}", "2")); """Squared value glyph."""
inverse  = Tr(glyphs.power("{}", glyphs.neg(1))); """Matrix inverse glyph."""
transp   = Tr("{}.T"); """Matrix transposition glyph."""
ptransp  = Tr("{}.Tx"); """Matrix partial transposition glyph."""
htransp  = Tr("{}.H"); """Matrix hermitian transposition glyph."""
index    = Tr("{}_{}"); """Index glyph."""

# Concatenation glyphs.
horicat  = Op("{}, {}", 4, assoc=True); """Horizontal concatenation glyph."""
vertcat  = Op("{}; {}", 4.5, assoc=True); """Vertical concatenation glyph."""

# Relation glyphs.
element  = Rl("{} in {}"); """Set element glyph."""
eq       = Rl("{} = {}");  """Equality glyph."""
ge       = Rl("{} >= {}"); """Greater or equal glyph."""
gt       = Rl("{} > {}");  """Greater than glyph."""
le       = Rl("{} <= {}"); """Lesser or equal glyph."""
lt       = Rl("{} < {}");  """Lesser than glyph."""
psdge    = Rl("{} >> {}"); """Lesser or equal w.r.t. the p.s.d. cone glyph."""
psdle    = Rl("{} << {}"); """Greater or equal w.r.t. the p.s.d. cone glyph."""

# Bracket-less function glyphs.
maxarg   = Op("max_{} {}", 3.5, closed=False)
minarg   = Op("min_{} {}", 3.5, closed=False)


def latin1(rebuildDerivedGlyphs=True):
    """Let PICOS create future strings using only ISO 8859-1 characters."""
    # Reset to ASCII first.
    ascii()

    # Update glyphs with only template changes.
    glyphs.compose.template  = "{}°{}"
    glyphs.cubed.template    = "{}³"
    glyphs.hadamard.template = "{}(·){}"
    glyphs.kron.template     = "{}(×){}"
    glyphs.leadsto.template  = "{} » {}"
    glyphs.mul.template      = "{}·{}"
    glyphs.squared.template  = "{}²"
    glyphs.plsmns.template   = "±{}"
    glyphs.psdge.template    = "{} » {}"
    glyphs.psdle.template    = "{} « {}"
    glyphs.size.template     = "{}×{}"

    # Update all derived glyphs.
    if rebuildDerivedGlyphs:
        rebuild()


def unicode(rebuildDerivedGlyphs=True):
    """Let PICOS create future strings using only unicode characters."""
    # Reset to LATIN-1 first.
    latin1(rebuildDerivedGlyphs=False)

    # Update glyphs with only template changes.
    glyphs.and_.template     = "{} ∧ {}"
    glyphs.compose.template  = "{}∘{}"
    glyphs.dotp.template     = "⟨{}, {}⟩"
    glyphs.element.template  = "{} ∈ {}"
    glyphs.forall.template   = "{} ∀ {}"
    glyphs.fromto.template   = "{}…{}"
    glyphs.ge.template       = "{} ≥ {}"
    glyphs.hadamard.template = "{}⊙{}"
    glyphs.htransp.template  = "{}ᴴ"
    glyphs.infty.template    = "∞"
    glyphs.kron.template     = "{}⊗{}"
    glyphs.lambda_.template  = "λ"
    glyphs.le.template       = "{} ≤ {}"
    glyphs.leadsto.template  = "{} → {}"
    glyphs.norm.template     = "‖{}‖"
    glyphs.or_.template      = "{} ∨ {}"
    glyphs.prod.template     = "∏({})"
    glyphs.psdge.template    = "{} ≽ {}"
    glyphs.psdle.template    = "{} ≼ {}"
    glyphs.sum.template      = "∑({})"
    glyphs.transp.template   = "{}ᵀ"

    # Update all derived glyphs.
    if rebuildDerivedGlyphs:
        rebuild()


# Set and use the default charset.
if settings.DEFAULT_CHARSET == "unicode":
    default = unicode
elif settings.DEFAULT_CHARSET == "latin1":
    default = latin1
elif settings.DEFAULT_CHARSET == "ascii":
    default = ascii
else:
    raise ValueError("PICOS doesn't have a charset named '{}'.".format(
        settings.DEFAULT_CHARSET))
default()


# Helper functions that mimic or create additional glyphs.
# --------------------------------------------------------

def scalar(value):
    """Format a scalar value.

    This function mimics an operator glyph, but it returns a normal string (as
    opposed to an :class:`OpStr`) for nonnegative numbers.

    This is not realized as an atomic operator glyph to not increase the
    recursion depth of :func:`is_negated` and :func:`unnegate` unnecessarily.

    **Example**

    >>> from picos.glyphs import scalar
    >>> str(1.0)
    '1.0'
    >>> scalar(1.0)
    '1'
    """
    if not isinstance(value, complex) and value < 0:
        value = -value
        negated = True
    else:
        negated = False

    string = ("{:g}" if type(value) is float else "{}").format(value)

    if negated:
        return glyphs.neg(string)
    else:
        return string


def shape(theShape):
    """Describe a matrix shape that can contain wildcards.

    A wrapper around :obj:`size` that takes just one argument (the shape) that
    may contain wildcards which are printed as ``'?'``.
    """
    newShape = (
        "?" if theShape[0] is None else theShape[0],
        "?" if theShape[1] is None else theShape[1])
    return glyphs.size(*newShape)


def make_function(*names):
    """Create an ad-hoc composite function glyphs.

    **Example**

    >>> from picos.glyphs import make_function
    >>> make_function("log", "sum", "exp")("x")
    'log∘sum∘exp(x)'
    """
    return Fn("{}({{}})".format(functools.reduce(glyphs.compose, names)))


# Helper functions that make context-sensitive use of glyphs.
# -----------------------------------------------------------

def from_glyph(string, theGlyph):
    """Whether the given string was created by the given glyph."""
    return isinstance(string, GlStr) and string.glyph is theGlyph


CAN_FACTOR_OUT_NEGATION = (
    glyphs.matrix,
    glyphs.sum,
    glyphs.trace,
    glyphs.vec,
    glyphs.diag,
    glyphs.maindiag,
    glyphs.real
)  #: Operator glyphs for which negation may be factored out.


def is_negated(value):
    """Check if a value can be unnegated by :func:`unnegate`."""
    if isinstance(value, OpStr) and value.glyph in CAN_FACTOR_OUT_NEGATION:
        return is_negated(value.operands[0])
    elif from_glyph(value, glyphs.neg):
        return True
    elif type(value) is str:
        try:
            return float(value) < 0
        except ValueError:
            return False
    elif type(value) in (int, float):
        return value < 0
    else:
        return False


def unnegate(value):
    """Unnegate a value, usually a glyph-created string, in a sensible way.

    Unnegates a :class:`operator glyph created string <OpStr>` or other value in
    a sensible way, more precisely by recursing through a sequence of glyphs
    used to create the value and for which we can factor out negation, and
    negating the underlaying (numeric or string) value.

    :raises ValueError: When :meth:`is_negated` returns :obj:`False`.
    """
    if isinstance(value, OpStr) and value.glyph in CAN_FACTOR_OUT_NEGATION:
        return value.glyph(unnegate(value.operands[0]))
    elif from_glyph(value, glyphs.neg):
        return value.operands[0]
    elif type(value) is str:
        # We raise any conversion error, because is_negated returns False.
        return "{:g}".format(-float(value))
    elif type(value) in (int, float):
        return -value
    else:
        raise ValueError("The value to recursively unnegate is not negated in a"
            "supported manner.")


def clever_neg(value):
    """Describe the negation of a value in a clever way.

    A wrapper around :attr:`neg` that resorts to unnegating an already negated
    value.

    **Example**

    >>> from picos.glyphs import neg, clever_neg, matrix
    >>> neg("x")
    '-x'
    >>> neg(neg("x"))
    '-(-x)'
    >>> clever_neg(neg("x"))
    'x'
    >>> neg(matrix(-1))
    '-[-1]'
    >>> clever_neg(matrix(-1))
    '[1]'
    """
    if is_negated(value):
        return unnegate(value)
    else:
        return glyphs.neg(value)


def clever_add(left, right):
    """Describe the addition of two values in a clever way.

    A wrapper around :attr:`add` that resorts to :attr:`sub` if the second
    operand was created by :attr:`neg` or is a negative number (string). In both
    cases the second operand is adjusted accordingly.

    **Example**

    >>> from picos.glyphs import neg, add, clever_add, matrix
    >>> add("x", neg("y"))
    'x + -y'
    >>> clever_add("x", neg("y"))
    'x - y'
    >>> add("X", matrix(neg("y")))
    'X + [-y]'
    >>> clever_add("X", matrix(neg("y")))
    'X - [y]'
    >>> clever_add("X", matrix(-1.5))
    'X - [1.5]'
    """
    if left in (0, "0"):
        return right

    if right in (0, "0"):
        return left

    if is_negated(right):
        return glyphs.sub(left, unnegate(right))
    else:
        return glyphs.add(left, right)


def clever_sub(left, right):
    """Describe the substraction of a value from another in a clever way.

    A wrapper around :attr:`sub` that resorts to :attr:`add` if the second
    operand was created by :attr:`neg` or is a negative number(string). In both
    cases the second operand is adjusted accordingly.

    **Example**

    >>> from picos.glyphs import neg, sub, clever_sub, matrix
    >>> sub("x", neg("y"))
    'x - -y'
    >>> clever_sub("x", neg("y"))
    'x + y'
    >>> sub("X", matrix(neg("y")))
    'X - [-y]'
    >>> clever_sub("X", matrix(neg("y")))
    'X + [y]'
    >>> clever_sub("X", matrix(-1.5))
    'X + [1.5]'
    """
    if left in (0, "0"):
        return clever_neg(right)

    if right in (0, "0"):
        return left

    if is_negated(right):
        return glyphs.add(left, unnegate(right))
    else:
        return glyphs.sub(left, right)


def clever_mul(left, right):
    """Describe the multiplocation of two values in a clever way.

    A wrapper around :attr:`mul` that factors out identity factors.
    """
    # Apply a factor of zero on the left side.
    if left in (0, "0"):
        return left

    # Apply a factor of zero on the right side.
    if right in (0, "0"):
        return right

    # Factor out a scalar one on the left side.
    if left in (1, "1"):
        return right

    # Factor out a scalar one on the right side.
    if right in (1, "1"):
        return left

    # Detect quadratics.
    if left == right:
        return glyphs.squared(left)

    # Factor out negation.
    ln, rn = is_negated(left), is_negated(right)
    if ln and rn:
        return glyphs.clever_mul(unnegate(left), unnegate(right))
    elif ln:
        return glyphs.neg(glyphs.clever_mul(unnegate(left), right))
    elif rn:
        return glyphs.neg(glyphs.clever_mul(left, unnegate(right)))

    return glyphs.mul(left, right)


def clever_div(left, right):
    """Describe the division of one value by another in a clever way.

    A wrapper around :attr:`div` that factors out identity factors.
    """
    # Apply a factor of zero on the left side.
    if left in (0, "0"):
        return left

    # Factor out a scalar one on the right side.
    if right in (1, "1"):
        return left

    # Factor out negation.
    ln, rn = is_negated(left), is_negated(right)
    if ln and rn:
        return glyphs.clever_div(unnegate(left), unnegate(right))
    elif ln:
        return glyphs.neg(glyphs.clever_div(unnegate(left), right))
    elif rn:
        return glyphs.neg(glyphs.clever_div(left, unnegate(right)))

    return glyphs.div(left, right)


def clever_dotp(left, right, complexRHS, scalar=False):
    """Describe an inner product in a clever way.

    :param bool complexRHS: Whether the right hand side is complex.
    """
    riCo = glyphs.conj(right) if complexRHS else right

    if scalar:
        return glyphs.clever_mul(left, right)

    if   from_glyph(left, glyphs.idmatrix):
        return glyphs.trace(riCo)
    elif from_glyph(riCo, glyphs.idmatrix):
        return glyphs.trace(left)
    elif from_glyph(left, glyphs.matrix) and left.operands[0] in (1, "1"):
        return glyphs.sum(riCo)
    elif from_glyph(riCo, glyphs.matrix) and riCo.operands[0] in (1, "1"):
        return glyphs.sum(left)
    elif left == right:
        return glyphs.squared(glyphs.norm(left))
    else:
        return glyphs.dotp(left, right)


def matrix_cat(left, right, horizontal=True):
    """Describe matrix concatenation in a clever way.

    A wrapper around :attr:`matrix`, :attr:`horicat` and :attr:`vertcat`.

    **Example**

    >>> from picos.glyphs import matrix_cat
    >>> Z = matrix_cat("X", "Y")
    >>> Z
    '[X, Y]'
    >>> matrix_cat(Z, Z)
    '[X, Y, X, Y]'
    >>> matrix_cat(Z, Z, horizontal = False)
    '[X, Y; X, Y]'
    """
    if isinstance(left, OpStr) and left.glyph is glyphs.matrix:
        left = left.operands[0]

    if isinstance(right, OpStr) and right.glyph is glyphs.matrix:
        right = right.operands[0]

    catGlyph = glyphs.horicat if horizontal else glyphs.vertcat

    return glyphs.matrix(catGlyph(left, right))


def row_vectorize(*entries):
    """Describe a row vector with the given symbolic entries."""
    return functools.reduce(matrix_cat, entries)


def col_vectorize(*entries):
    """Describe a column vector with the given symbolic entries."""
    return functools.reduce(lambda l, r: matrix_cat(l, r, False), entries)


def free_var_name(string):
    """Return a variable name not present in the given string."""
    names = "xyzpqrstuvwabcdefghijklmno"
    for name in names:
        if name not in string:
            return name
    return "?"


# --------------------------------------
__all__ = api_end(_API_START, globals())
