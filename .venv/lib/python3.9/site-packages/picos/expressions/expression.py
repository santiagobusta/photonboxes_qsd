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

"""Backend for expression type implementations."""

import functools
import operator
import threading
import warnings
from abc import abstractmethod
from contextlib import contextmanager

from .. import glyphs
from ..apidoc import api_end, api_start
from ..caching import cached_property
from ..constraints import ConstraintType
from ..containers import DetailedType
from ..legacy import deprecated
from ..valuable import NotValued, Valuable
from .data import convert_operands

_API_START = api_start(globals())
# -------------------------------


def validate_prediction(the_operator):
    """Validate that the constraint outcome matches the predicted outcome."""
    @functools.wraps(the_operator)
    def wrapper(lhs, rhs, *args, **kwargs):
        from .set import Set

        def what():
            return "({}).{}({})".format(
                lhs._symbStr, the_operator.__name__, rhs._symbStr)

        assert isinstance(lhs, (Expression, Set)) \
            and isinstance(rhs, (Expression, Set)), \
            "validate_prediction must occur below convert_operands."

        lhs_type = lhs.type
        rhs_type = rhs.type

        try:
            abstract_operator = getattr(operator, the_operator.__name__)
        except AttributeError as error:
            raise AssertionError("validate_prediction may only decorate "
                "standard operator implementations.") from error

        try:
            predictedType = lhs_type.predict(abstract_operator, rhs_type)
        except NotImplementedError:
            predictedType = None  # No prediction was made.
        except PredictedFailure:
            predictedType = NotImplemented  # Prediction is "not possible".

        try:
            outcome = the_operator(lhs, rhs, *args, **kwargs)
        except Exception as error:
            # Case where the prediction is positive and the outcome is negative.
            if predictedType not in (None, NotImplemented):
                warnings.warn(
                    "Outcome for {} was predicted {} but the operation raised "
                    "an error: \"{}\" This a noncritical error (false positive)"
                    " in PICOS' constraint outcome prediction."
                    .format(what(), predictedType, error),
                    category=RuntimeWarning, stacklevel=3)
                raise
            else:
                raise

        # Case where the prediction is negative and the outcome is positive.
        if predictedType is NotImplemented and outcome is not NotImplemented:
            raise AssertionError(
                "The operation {} was predicted to fail but it produced "
                "an output of {}.".format(what(), outcome.type))

        # Case where no prediction was made.
        if not predictedType:
            return outcome

        # Case where the outcome is try-to-reverse-the-operation.
        if outcome is NotImplemented:
            return outcome

        # Case where the prediction and the outcome are positive but differ.
        outcomeType = outcome.type
        if not predictedType.equals(outcomeType):
            raise AssertionError("Outcome for {} was predicted {} but is {}."
                .format(what(), predictedType, outcomeType))

        return outcome
    return wrapper


def refine_operands(stop_at_affine=False):
    """Cast :meth:`~Expression.refined` on both operands.

    If the left hand side operand (i.e. ``self``) is refined to an instance of a
    different type, then, instead of the decorated method, the method with the
    same name on the refined type is invoked with the (refined) right hand side
    operand as its argument.

    This decorator is supposed to be used on all constraint creating binary
    operator methods so that degenerated instances (e.g. a complex affine
    expression with an imaginary part of zero) can occur but are not used in
    constraints. This speeds up many computations involving expressions as these
    degenerate cases do not need to be detected. Note that
    :attr:`Expression.type` also refers to the refined version of an expression.

    :param bool stop_at_affine: Do not refine any affine expressions, in
        particular do not refine complex affine expressions to real ones.
    """
    def decorator(the_operator):
        @functools.wraps(the_operator)
        def wrapper(lhs, rhs, *args, **kwargs):
            from .exp_affine import ComplexAffineExpression
            from .set import Set

            assert isinstance(lhs, (Expression, Set)) \
                and isinstance(rhs, (Expression, Set)), \
                "refine_operands must occur below convert_operands."

            if stop_at_affine and isinstance(lhs, ComplexAffineExpression):
                lhs_refined = lhs
            else:
                lhs_refined = lhs.refined

            if type(lhs_refined) is not type(lhs):
                assert hasattr(lhs_refined, the_operator.__name__), \
                    "refine_operand transformed 'self' to another type that " \
                    "does not define an operator with the same name as the " \
                    "decorated one."

                refined_operation = getattr(lhs_refined, the_operator.__name__)

                return refined_operation(rhs, *args, **kwargs)

            if stop_at_affine and isinstance(rhs, ComplexAffineExpression):
                rhs_refined = rhs
            else:
                rhs_refined = rhs.refined

            return the_operator(lhs_refined, rhs_refined, *args, **kwargs)
        return wrapper
    return decorator


# TODO: Once PICOS requires Python >= 3.7, use a ContextVar instead.
class _Refinement(threading.local):
    allowed = True


_REFINEMENT = _Refinement()


@contextmanager
def no_refinement():
    """Context manager that disables the effect of :meth:`Expression.refined`.

    This can be necessary to ensure that the outcome of a constraint coversion
    is as predicted, in particular when PICOS uses overridden comparison
    operators for constraint creation internally.
    """
    _REFINEMENT.allowed = False

    try:
        yield
    finally:
        _REFINEMENT.allowed = True


class PredictedFailure(TypeError):
    """Denotes that comparing two expressions will not form a constraint."""

    pass


class ExpressionType(DetailedType):
    """The detailed type of an expression for predicting constraint outcomes.

    This is suffcient to predict the detailed type of any constraint that can be
    created by comparing with another expression.
    """

    @staticmethod
    def _relation_str(relation):
        if relation is operator.__eq__:
            return "=="
        elif relation is operator.__le__:
            return "<="
        elif relation is operator.__ge__:
            return ">="
        elif relation is operator.__lshift__:
            return "<<"
        elif relation is operator.__rshift__:
            return ">>"
        else:
            return "??"

    @staticmethod
    def _swap_relation(relation):
        if relation is operator.__eq__:
            return operator.__eq__
        elif relation is operator.__le__:
            return operator.__ge__
        elif relation is operator.__ge__:
            return operator.__le__
        elif relation is operator.__lshift__:
            return operator.__rshift__
        elif relation is operator.__rshift__:
            return operator.__lshift__
        else:
            return None

    def predict(self, relation, other):
        """Predict the constraint outcome of comparing expressions.

        :param relation:
            An object from the :mod:`operator` namespace representing the
            operation being predicted.

        :param other:
            Another expression type representing the right hand side operand.
        :type other:
            ~picos.expressions.expression.ExpressionType

        :Example:

        >>> import operator, picos
        >>> a = picos.RealVariable("x") + 1
        >>> b = picos.RealVariable("y") + 2
        >>> (a <= b).type == a.type.predict(operator.__le__, b.type)
        True
        """
        if not isinstance(other, ExpressionType):
            raise TypeError("The 'other' argument must be another {} instance."
                .format(self.__class__.__name__))

        # Perform the forward prediction.
        result = self.clstype._predict(self.subtype, relation, other)

        # Fall back to the backward prediction.
        if result is NotImplemented:
            reverse = self._swap_relation(relation)
            result  = other.clstype._predict(other.subtype, reverse, self)

        # If both fail, the prediction is "not possible".
        if result is NotImplemented:
            raise PredictedFailure(
                "The statement {} {} {} is predicted to error."
                .format(self, self._relation_str(relation), other))
        else:
            assert isinstance(result, ConstraintType)
            return result


class Expression(Valuable):
    """Abstract base class for mathematical expressions, including mutables.

    For mutables, this is the secondary base class, with
    :class:`~.mutable.Mutable` or a subclass thereof being the primary one.
    """

    def __init__(self, typeStr, symbStr):
        """Perform basic initialization for :class:`Expression` instances.

        :param str typeStr: Short string denoting the expression type.
        :param str symbStr: Algebraic string description of the expression.
        """
        self._typeStr = typeStr
        """A string describing the expression type."""

        self._symbStr = symbStr
        """A symbolic string representation of the expression. It is always used
        by __descr__, and it is equivalent to the value returned by __str__ when
        the expression is not fully valued."""

    @property
    def string(self):
        """Symbolic string representation of the expression.

        Use this over Python's :class:`str` if you want to output the symbolic
        representation even when the expression is valued.
        """
        return self._symbStr

    # --------------------------------------------------------------------------
    # Abstract method implementations for the Valuable base class.
    # NOTE: _get_value and possibly _set_value are implemented by subclasses.
    # --------------------------------------------------------------------------

    def _get_valuable_string(self):
        return "expression {}".format(self.string)

    # --------------------------------------------------------------------------
    # Abstract and default-implementation methods.
    # --------------------------------------------------------------------------

    def _get_refined(self):
        """See :attr:`refined`."""
        return self

    def _get_clstype(self):
        """Return the Python class part of the expression's detailed type."""
        return self.__class__

    @property
    @abstractmethod
    def Subtype(self):
        """The class of which :attr:`subtype` returns an instance.

        Instances must be hashable. By convention a
        :func:`namedtuple <collections.namedtuple>` class.

        .. warning::
            This should be declared in the class body as e.g.
            `Subtype = namedtuple(…)` and not as a property so that it's static.
        """
        pass

    @abstractmethod
    def _get_subtype(self):
        """See :attr:`subtype`."""
        pass

    @classmethod
    @abstractmethod
    def _predict(cls, subtype, relation, other):
        """Predict the constraint outcome of a comparison.

        :param object subtype: An object returned by the :meth:`_get_subtype`
            instance method of :class:`cls`.
        :param method-wrapper relation: A function from the :mod:`operator`
            namespace, such as :func:`operator.__le__`. See
            :class:`ExpressionType` for what operators are defined.
        :param ExpressionType other: The detailed type of another expression.
        :returns: Either the :obj:`NotImplemented` token or a
            :class:`ConstraintType` object such that an instance of :class:`cls`
            with the given subtype, when compared with another expression with
            the given expression type, returns a constraint with that constraint
            type.
        """
        pass

    def _get_shape(self):
        """Return the algebraic shape of the expression."""
        return (1, 1)

    @abstractmethod
    def _get_mutables(self):
        """Return the set of mutables that are involved in the expression."""
        pass

    @abstractmethod
    def _is_convex(self):
        """Whether the expression is convex in its :attr:`variables`.

        Method implementations may assume that the expression is refined. Thus,
        degenerate cases affected by refinement do not need to be considered.

        For uncertain expressions, this assumes the perturbation as constant.
        """
        pass

    @abstractmethod
    def _is_concave(self):
        """Whether the expression is concave in its :attr:`variables`.

        Method implementations may assume that the expression is refined. Thus,
        degenerate cases affected by refinement do not need to be considered.

        For uncertain expressions, this assumes the perturbation as constant.
        """
        pass

    @abstractmethod
    def _replace_mutables(self, mapping):
        """Return a copy of the expression concerning different mutables.

        This is the fast internal-use counterpart to :meth:`replace_mutables`.

        The returned expression should be of the same type as ``self`` (no
        refinement) so that it can be substituted in composite expressions.

        :param dict mapping:
            A mutable replacement map. The caller must ensure the following
            properties:

            1. This must be a complete map from existing mutables to the same
               mutable, another mutable, or a real-valued affine expression
               (completeness).
            2. The shape and vectorization format of each replacement must match
               the existing mutable. Replacing with affine expressions is only
               allowed when the existing mutable uses the trivial
               :class:`~vectorizations.FullVectorization` (soudness).
            3. Mutables that appear in a replacement may be the same as the
               mutable being replaced but may otherwise not appear in the
               expression (freshness).
            4. Mutables may appear at most once anywhere in the image of the map
               (uniqueness).

            If any property is not fulfilled, the implementation does not need
            to raise a proper exception but may fail arbitrarily.
        """
        pass

    @abstractmethod
    def _freeze_mutables(self, subset):
        """Return a copy with some mutables frozen to their current value.

        This is the fast internal-use counterpart to :meth:`frozen`.

        The returned expression should be of the same type as ``self`` (no
        refinement) so that it can be substituted in composite expressions.

        :param dict subset:
            An iterable of valued :class:`mutables <.mutable.Mutable>` that
            should be frozen. May include mutables that are not present in the
            expression, but may not include mutables without a value.
        """
        pass

    # --------------------------------------------------------------------------
    # An interface to the abstract and default-implementation methods above.
    # --------------------------------------------------------------------------

    @property
    def refined(self):
        """A refined version of the expression.

        The refined expression can be an instance of a different
        :class:`Expression` subclass than the original expression, if that type
        is better suited for the mathematical object in question.

        The refined expression is automatically used instead of the original one
        whenever a constraint is created, and in some other places.

        The idea behind refined expressions is that operations that produce new
        expressions can be executed quickly without checking for exceptionnel
        cases. For instance, the sum of two
        :class:`~.exp_affine.ComplexAffineExpression` instances could have the
        complex part eliminated so that storing the result as an
        :class:`~.exp_affine.AffineExpression` would be prefered, but checking
        for this case on every addition would be too slow. Refinement is used
        sparingly to detect such cases at times where it makes the most sense.

        Refinement may be disallowed within a context with the
        :func:`no_refinement` context manager. In this case, this property
        returns the expression as is.
        """
        if not _REFINEMENT.allowed:
            return self

        fine = self._get_refined()

        if fine is not self:
            # Recursively refine until the expression doesn't change further.
            return fine.refined
        else:
            return fine

    @property
    def subtype(self):
        """The subtype part of the expression's detailed type.

        Returns a hashable object that, together with the Python class part of
        the expression's type, is sufficient to predict the constraint outcome
        (constraint class and subtype) of any comparison operation with any
        other expression.

        By convention the object returned is a
        :func:`namedtuple <collections.namedtuple>` instance.
        """
        return self._get_subtype()

    @property
    def type(self):
        """The expression's detailed type for constraint prediction.

        The returned value is suffcient to predict the detailed type of any
        constraint that can be created by comparing with another expression.

        Since constraints are created from
        :attr:`~.expression.Expression.refined` expressions only, the Python
        class part of the detailed type may differ from the type of the
        expression whose :attr:`type` is queried.
        """
        refined = self.refined
        return ExpressionType(refined._get_clstype(), refined._get_subtype())

    @classmethod
    def make_type(cls, *args, **kwargs):
        """Create a detailed expression type from subtype parameters."""
        return ExpressionType(cls, cls.Subtype(*args, **kwargs))

    shape = property(
        lambda self: self._get_shape(),
        doc=_get_shape.__doc__)

    size = property(
        lambda self: self._get_shape(),
        doc="""The same as :attr:`shape`.""")

    @property
    def scalar(self):
        """Whether the expression is scalar."""
        return self._get_shape() == (1, 1)

    @property
    def square(self):
        """Whether the expression is a square matrix."""
        shape = self._get_shape()
        return shape[0] == shape[1]

    mutables = property(
        lambda self: self._get_mutables(),
        doc=_get_mutables.__doc__)

    @property
    def constant(self):
        """Whether the expression involves no mutables."""
        return not self._get_mutables()

    @cached_property
    def variables(self):
        """The set of decision variables that are involved in the expression."""
        from .variables import BaseVariable

        return frozenset(mutable for mutable in self._get_mutables()
            if isinstance(mutable, BaseVariable))

    @cached_property
    def parameters(self):
        """The set of parameters that are involved in the expression."""
        from .variables import BaseVariable

        return frozenset(mutable for mutable in self._get_mutables()
            if not isinstance(mutable, BaseVariable))

    @property
    def convex(self):
        """Whether the expression is convex."""
        return self.refined._is_convex()

    @property
    def concave(self):
        """Whether the expression is concave."""
        return self.refined._is_concave()

    def replace_mutables(self, replacement):
        """Return a copy of the expression concerning different mutables.

        New mutables must have the same shape and vectorization format as the
        mutables that they replace. This means in particular that
        :class:`~.variables.RealVariable`, :class:`~.variables.IntegerVariable`
        and :class:`~.variables.BinaryVariable` of same shape are
        interchangeable.

        If the mutables to be replaced do not appear in the expression, then
        the expression is not copied but returned as is.

        :param replacement:
            Either a map from mutables or mutable names to new mutables or an
            iterable of new mutables to replace existing mutables of same name
            with. See the section on advanced usage for additional options.
        :type replacement:
            tuple or list or dict

        :returns Expression:
            The new expression, refined to a more suitable type if possible.

        :Advanced replacement:

        It is also possible to replace mutables with real affine expressions
        concerning pairwise disjoint sets of fresh mutables. This works only on
        real-valued mutables that have a trivial internal vectorization format
        (i.e. :class:`~.vectorizations.FullVectorization`). The shape of the
        replacing expression must match the variable's. Additional limitations
        depending on the type of expression that the replacement is invoked on
        are possible. The ``replacement`` argument must be a dictionary.

        :Example:

        >>> import picos
        >>> x = picos.RealVariable("x"); x.value = 1
        >>> y = picos.RealVariable("y"); y.value = 10
        >>> z = picos.RealVariable("z"); z.value = 100
        >>> c = picos.Constant("c", 1000)
        >>> a = x + 2*y; a
        <1×1 Real Linear Expression: x + 2·y>
        >>> a.value
        21.0
        >>> b = a.replace_mutables({y: z}); b  # Replace y with z.
        <1×1 Real Linear Expression: x + 2·z>
        >>> b.value
        201.0
        >>> d = a.replace_mutables({x: 2*x + z, y: c}); d  # Advanced use.
        <1×1 Real Affine Expression: 2·x + z + 2·c>
        >>> d.value
        2102.0
        """
        from .exp_biaffine import BiaffineExpression
        from .mutable import Mutable
        from .vectorizations import FullVectorization

        # Change an iterable of mutables to a map from names to mutables.
        if not isinstance(replacement, dict):
            if not all(isinstance(new, Mutable) for new in replacement):
                raise TypeError("If 'replacement' is a non-dictionary iterable,"
                    " then it may only contain mutables.")

            new_replacement = {new.name: new for new in replacement}

            if len(new_replacement) != len(replacement):
                raise TypeError("If 'replacement' is a non-dictionary iterable,"
                    " then the mutables within must have unique names.")

            replacement = new_replacement

        # Change a map from names to a map from existing mutables.
        # Names that reference non-existing mutables are dropped.
        old_mtbs_by_name = {mtb.name: mtb for mtb in self.mutables}
        replacing_by_name = False
        new_replacement = {}
        for old, new in replacement.items():
            if isinstance(old, Mutable):
                new_replacement[old] = new
            elif not isinstance(old, str):
                raise TypeError(
                    "Keys of 'replacement' must be mutables or names thereof.")
            else:
                replacing_by_name = True
                if old in old_mtbs_by_name:
                    new_replacement[old_mtbs_by_name[old]] = new
        replacement = new_replacement

        # Check unique naming of existing mutables if it matters.
        if replacing_by_name and len(old_mtbs_by_name) != len(self.mutables):
            raise RuntimeError("Cannot replace mutables by name in {} as "
                "its mutables are not uniquely named.".format(self.string))

        # Remove non-existing sources and identities.
        assert all(isinstance(old, Mutable) for old in replacement)
        replacement = {old: new for old, new in replacement.items()
            if old is not new and old in self.mutables}

        # Do nothing if there is nothing to replace.
        if not replacement:
            return self

        # Validate individual replacement requirements.
        for old, new in replacement.items():
            # Replacement must be a mutable or biaffine expression.
            if not isinstance(new, BiaffineExpression):
                raise TypeError("Can only replace mutables with other mutables "
                    "or affine expressions thereof.")

            # Shapes must match.
            if old.shape != new.shape:
                raise TypeError(
                    "Cannot replace {} with {} in {}: Differing shape."
                    .format(old.name, new.name, self.string))

            # Special requirements when replacing with mutables or expressions.
            if isinstance(new, Mutable):
                # Vectorization formats must match.
                if type(old._vec) != type(new._vec):  # noqa: E721
                    raise TypeError("Cannot replace {} with {} in {}: "
                        "Differing vectorization."
                        .format(old.name, new.name, self.string))
            else:
                # Replaced mutable must use a trivial vectorization.
                if not isinstance(old._vec, FullVectorization):
                    raise TypeError("Can only replace mutables using a trivial "
                        "vectorization format with affine expressions.")

                # Replacing expression must be real-valued and affine.
                if new._bilinear_coefs or new.complex:
                    raise TypeError("Can only replace mutables with real-valued"
                        " affine expressions.")

        old_mtbs_set = set(replacement)
        new_mtbs_lst = [mtb  # Excludes each mutable being replaced.
            for old, new in replacement.items()
            for mtb in new.mutables.difference((old,))]
        new_mtbs_set = set(new_mtbs_lst)

        # New mutables must be fresh.
        # It is OK to replace a mutable with itself or an affine expression of
        # itself and other fresh mutables, though.
        if old_mtbs_set.intersection(new_mtbs_set):
            raise ValueError("Can only replace mutables with fresh mutables "
                "or affine expressions of all fresh mutables (the old mutable "
                "may appear in the expression).")

        # New mutables must be unique.
        if len(new_mtbs_lst) != len(new_mtbs_set):
            raise ValueError("Can only replace multiple mutables at once if "
                "the replacing mutables (and/or the mutables in replacing "
                "expressions) are all unique.")

        # Turn the replacement map into a complete map.
        mapping = {mtb: mtb for mtb in self.mutables}
        mapping.update(replacement)

        # Replace recursively and refine the result.
        return self._replace_mutables(mapping).refined

    def frozen(self, subset=None):
        """The expression with valued mutables frozen to their current value.

        If all mutables of the expression are valued (and in the subset unless
        ``subset=None``), this is the same as the inversion operation ``~``.

        If the mutables to be frozen do not appear in the expression, then the
        expression is not copied but returned as is.

        :param subset:
            An iterable of valued :class:`mutables <.mutable.Mutable>` or names
            thereof that should be frozen. If :obj:`None`, then all valued
            mutables are frozen to their current value. May include mutables
            that are not present in the expression, but may not include mutables
            without a value.

        :returns Expression:
            The frozen expression, refined to a more suitable type if possible.

        :Example:

        >>> from picos import RealVariable
        >>> x, y = RealVariable("x"), RealVariable("y")
        >>> f = x + y; f
        <1×1 Real Linear Expression: x + y>
        >>> sorted(f.mutables, key=lambda mtb: mtb.name)
        [<1×1 Real Variable: x>, <1×1 Real Variable: y>]
        >>> x.value = 5
        >>> g = f.frozen(); g  # g is f with x frozen at its current value of 5.
        <1×1 Real Affine Expression: [x] + y>
        >>> sorted(g.mutables, key=lambda mtb: mtb.name)
        [<1×1 Real Variable: y>]
        >>> x.value, y.value = 10, 10
        >>> f.value  # x takes its new value in f.
        20.0
        >>> g.value  # x remains frozen at [x] = 5 in g.
        15.0
        >>> # If an expression is frozen to a constant, this is reversable:
        >>> f.frozen().equals(~f) and ~f.frozen() is f
        True
        """
        from .mutable import Mutable

        # Collect mutables to be frozen in the expression.
        if subset is None:
            freeze = set(mtb for mtb in self.mutables if mtb.valued)
        else:
            if not all(isinstance(mtb, (str, Mutable)) for mtb in subset):
                raise TypeError("Some element of the subset of mutables to "
                    "freeze is neither a mutable nor a string.")

            subset_mtbs = set(m for m in subset if isinstance(m, Mutable))
            subset_name = set(n for n in subset if isinstance(n, str))

            freeze = set()
            if subset_mtbs:
                freeze.update(m for m in subset_mtbs if m in self.mutables)
            if subset_name:
                freeze.update(m for m in self.mutables if m.name in subset_name)

            if not all(mtb.valued for mtb in freeze):
                raise NotValued(
                    "Not all mutables in the selected subset are valued.")

        if not freeze:
            return self

        if freeze == self.mutables:
            return ~self  # Allow ~self.frozen() to return self.

        return self._freeze_mutables(freeze).refined

    @property
    def certain(self):
        """Always :obj:`True` for certain expression types.

        This can be :obj:`False` for Expression types that inherit from
        :class:`~.uexpression.UncertainExpression` (with priority).
        """
        return True

    @property
    def uncertain(self):
        """Always :obj:`False` for certain expression types.

        This can be :obj:`True` for Expression types that inherit from
        :class:`~.uexpression.UncertainExpression` (with priority).
        """
        return False

    # --------------------------------------------------------------------------
    # Python special method implementations.
    # --------------------------------------------------------------------------

    def __len__(self):
        """Report the number of entries of the (multidimensional) expression."""
        return self.shape[0] * self.shape[1]

    def __le__(self, other):
        """Return a constraint that the expression is upper-bounded."""
        # Try to refine self and see if the operation is then supported.
        # This allows e.g. a <= 0 if a is a real-valued complex expression.
        refined = self.refined
        if type(refined) != type(self):
            return refined.__le__(other)

        return NotImplemented

    def __ge__(self, other):
        """Return a constraint that the expression is lower-bounded."""
        # Try to refine self and see if the operation is then supported.
        # This allows e.g. a >= 0 if a is a real-valued complex expression.
        refined = self.refined
        if type(refined) != type(self):
            return refined.__ge__(other)

        return NotImplemented

    def __invert__(self):
        """Convert between a valued expression and its value.

        The value is returned as a constant affine expression whose conversion
        returns the original expression.
        """
        if hasattr(self, "_origin"):
            return self._origin
        elif self.constant:
            return self

        from .exp_affine import Constant

        A = Constant(
            glyphs.frozen(self.string), self.safe_value_as_matrix, self.shape)
        A._origin = self
        return A

    def __contains__(self, mutable):
        """Report whether the expression concerns the given mutable."""
        return mutable in self.mutables

    def __eq__(self, exp):
        """Return an equality constraint concerning the expression."""
        raise NotImplementedError("PICOS supports equality comparison only "
            "between affine expressions, as otherwise the problem would "
            "become non-convex. Choose either <= or >= if possible.")

    def __repr__(self):
        """Return a bracketed string description of the expression.

        The description contains both the mathematical type and a symbolic
        description of the expression.
        """
        return str(glyphs.repr2(self._typeStr, self._symbStr))

    def __str__(self):
        """Return a dynamic string description of the expression.

        The description is based on whether the expression is valued. If it is
        valued, then a string representation of the value is returned.
        Otherwise, the symbolic description of the expression is returned.
        """
        value = self.value

        if value is None:
            return str(self._symbStr)
        else:
            return str(value).strip()

    def __format__(self, format_spec):
        """Format either the value or the symbolic string of the expression.

        If the expression is valued, then its value is formatted, otherwise its
        symbolic string description.
        """
        value = self.value

        if value is None:
            return self._symbStr.__format__(format_spec)
        else:
            return value.__format__(format_spec)

    # Since we define __eq__, __hash__ is not inherited. Do this manually.
    __hash__ = object.__hash__

    # --------------------------------------------------------------------------
    # Fallback algebraic operations: Try again with converted RHS, refined LHS.
    # NOTE: The forward operations call the backward operations manually
    #       (instead of returning NotImplemented) so that they can be performed
    #       on a converted operand, which is always a PICOS type. The backward
    #       operations then use WeightedSum as a last fallback where applicable.
    # --------------------------------------------------------------------------

    def _wsum_fallback(self, summands, weights, opstring):
        """Try to represent the result as a weighted sum."""
        from .exp_wsum import WeightedSum

        # NOTE: WeightedSum with an opstring set will act as a final fallback
        #       and raise a proper exception if the result can't be represented.
        #       This is handled there and not here so that also operations on
        #       existing WeightedSum instances can produce such exceptions, as
        #       they cannot fallback to Expression like other operations do.
        return WeightedSum(summands, weights, opstring)

    def _scalar_mult_fallback(self, lhs, rhs):
        """Try to express scalar by scalar multiplication as a weighted sum."""
        assert isinstance(lhs, Expression) and isinstance(rhs, Expression)

        opstring = "a product between {} and {}".format(repr(lhs), repr(rhs))

        if lhs.scalar and lhs.constant:
            return self._wsum_fallback((rhs,), lhs.safe_value, opstring)
        elif rhs.scalar and rhs.constant:
            return self._wsum_fallback((lhs,), rhs.safe_value, opstring)
        else:
            # NOTE: Constant scalars are also AffineExpression but otherwise
            #       raising the default Python TypeError (stating that the two
            #       types are fully operation-incompatible) makes sense here.
            return NotImplemented

    @convert_operands(sameShape=True)
    def __add__(self, other):
        """Denote addition with another expression on the right-hand side."""
        if type(self.refined) != type(self):
            return self.refined.__add__(other)
        else:
            return other.__radd__(self)

    @convert_operands(sameShape=True)
    def __radd__(self, other):
        """Denote addition with another expression on the left-hand side."""
        if type(self.refined) != type(self):
            return self.refined.__radd__(other)
        else:
            opstring = "{} plus {}".format(repr(other), repr(self))
            return self._wsum_fallback((other, self), (1, 1), opstring)

    @convert_operands(sameShape=True)
    def __sub__(self, other):
        """Denote subtraction of another expression from the expression."""
        if type(self.refined) != type(self):
            return self.refined.__sub__(other)
        else:
            return other.__rsub__(self)

    @convert_operands(sameShape=True)
    def __rsub__(self, other):
        """Denote subtraction of the expression from another expression."""
        if type(self.refined) != type(self):
            return self.refined.__rsub__(other)
        else:
            opstring = "{} minus {}".format(repr(other), repr(self))
            return self._wsum_fallback((other, self), (1, -1), opstring)

    @convert_operands(sameShape=True)
    def __or__(self, other):
        r"""Denote the scalar product with another expression on the right.

        For (complex) vectors :math:`a` and :math:`b` this is the dot product

        .. math::
            (a \mid b)
            &= \langle a, b \rangle \\
            &= a \cdot b \\
            &= b^H a.

        For (complex) matrices :math:`A` and :math:`B` this is the Frobenius
        inner product

        .. math::
            (A \mid B)
            &= \langle A, B \rangle_F \\
            &= A : B \\
            &= \operatorname{tr}(B^H A) \\
            &= \operatorname{vec}(B)^H \operatorname{vec}(\overline{A})

        .. note::
            Write ``(A|B)`` instead of ``A|B`` for the scalar product of ``A``
            and ``B`` to obtain correct operator binding within a larger
            expression context.
        """
        if type(self.refined) != type(self):
            return self.refined.__or__(other)
        else:
            return other.__ror__(self)

    @convert_operands(sameShape=True)
    def __ror__(self, other):
        """Denote the scalar product with another expression on the left.

        See :meth:`__or__` for details on this operation.
        """
        if type(self.refined) != type(self):
            return self.refined.__ror__(other)
        else:
            return self._scalar_mult_fallback(other, self)

    @convert_operands(rMatMul=True)
    def __mul__(self, other):
        """Denote multiplication with another expression on the right."""
        if type(self.refined) != type(self):
            return self.refined.__mul__(other)
        else:
            return other.__rmul__(self)

    @convert_operands(lMatMul=True)
    def __rmul__(self, other):
        """Denote multiplication with another expression on the left."""
        if type(self.refined) != type(self):
            return self.refined.__rmul__(other)
        else:
            return self._scalar_mult_fallback(other, self)

    @convert_operands(sameShape=True)
    def __xor__(self, other):
        """Denote the entrywise product with another expression on the right."""
        if type(self.refined) != type(self):
            return self.refined.__xor__(other)
        else:
            return other.__rxor__(self)

    @convert_operands(sameShape=True)
    def __rxor__(self, other):
        """Denote the entrywise product with another expression on the left."""
        if type(self.refined) != type(self):
            return self.refined.__rxor__(other)
        else:
            return self._scalar_mult_fallback(other, self)

    @convert_operands()
    def __matmul__(self, other):
        """Denote the Kronecker product with another expression on the right."""
        if type(self.refined) != type(self):
            return self.refined.__matmul__(other)
        else:
            return other.__rmatmul__(self)

    @convert_operands()
    def __rmatmul__(self, other):
        """Denote the Kronecker product with another expression on the left."""
        if type(self.refined) != type(self):
            return self.refined.__rmatmul__(other)
        else:
            return self._scalar_mult_fallback(other, self)

    @convert_operands(scalarRHS=True)
    def __truediv__(self, other):
        """Denote division by another, scalar expression."""
        if type(self.refined) != type(self):
            return self.refined.__truediv__(other)
        else:
            return other.__rtruediv__(self)

    @convert_operands(scalarLHS=True)
    def __rtruediv__(self, other):
        """Denote scalar division of another expression."""
        if type(self.refined) != type(self):
            return self.refined.__rtruediv__(other)
        else:
            if self.constant and not self.is0:
                try:
                    return other.__mul__(1 / self.safe_value)
                except TypeError:
                    assert False, "Multiplication of {} by a nonzero constant" \
                        " has unexpectedly failed; it should have produced a " \
                        "weighted sum.".format(repr(other))
            else:
                reason = "nonconstant" if not self.constant else "zero"
                raise TypeError("Cannot divide {} by {}: The denominator is {}."
                    .format(repr(other), repr(self), reason))

    @convert_operands(scalarRHS=True)
    def __pow__(self, other):
        """Denote exponentiation with another, scalar expression."""
        if type(self.refined) != type(self):
            return self.refined.__pow__(other)
        else:
            return other.__rpow__(self)

    @convert_operands(scalarLHS=True)
    def __rpow__(self, other):
        """Denote taking another expression to the power of the expression."""
        if type(self.refined) != type(self):
            return self.refined.__rpow__(other)
        else:
            return NotImplemented

    @convert_operands(horiCat=True)
    def __and__(self, other):
        """Denote horizontal stacking with another expression on the right."""
        if type(self.refined) != type(self):
            return self.refined.__and__(other)
        else:
            return other.__rand__(self)

    @convert_operands(horiCat=True)
    def __rand__(self, other):
        """Denote horizontal stacking with another expression on the left."""
        if type(self.refined) != type(self):
            return self.refined.__rand__(other)
        else:
            return NotImplemented

    @convert_operands(vertCat=True)
    def __floordiv__(self, other):
        """Denote vertical stacking with another expression below."""
        if type(self.refined) != type(self):
            return self.refined.__floordiv__(other)
        else:
            return other.__rfloordiv__(self)

    @convert_operands(vertCat=True)
    def __rfloordiv__(self, other):
        """Denote vertical stacking with another expression above."""
        if type(self.refined) != type(self):
            return self.refined.__rfloordiv__(other)
        else:
            return NotImplemented

    def __pos__(self):
        """Return the expression as-is."""
        return self

    def __neg__(self):
        """Denote the negation of the expression."""
        if type(self.refined) != type(self):
            return self.refined.__neg__()
        else:
            opstring = "the negation of {}".format(repr(self))
            return self._wsum_fallback((self,), -1, opstring)

    def __abs__(self):
        """Denote the default norm of the expression.

        The norm used depends on the expression's domain. It is

            1. the absolute value of a real scalar,
            2. the modulus of a complex scalar,
            3. the Euclidean norm of a vector, and
            4. the Frobenius norm of a matrix.
        """
        if type(self.refined) != type(self):
            return self.refined.__abs__()
        else:
            return NotImplemented

    # --------------------------------------------------------------------------
    # Turn __lshift__ and __rshift__ into a single binary relation.
    # This is used for both Loewner order (defining LMIs) and set membership.
    # --------------------------------------------------------------------------

    def _lshift_implementation(self, other):
        return NotImplemented

    def _rshift_implementation(self, other):
        return NotImplemented

    @convert_operands(diagBroadcast=True)
    @validate_prediction
    @refine_operands()
    def __lshift__(self, other):
        """Denote either set membership or a linear matrix inequality.

        If the other operand is a set, then this denotes that the expression
        shall be constrained to that set. Otherwise, it is expected that both
        expressions are square matrices of same shape and this denotes that the
        expression is upper-bounded by the other expression with respect to the
        Loewner order (i.e. ``other - self`` is positive semidefinite).
        """
        result = self._lshift_implementation(other)

        if result is NotImplemented:
            result = other._rshift_implementation(self)

        return result

    @convert_operands(diagBroadcast=True)
    @validate_prediction
    @refine_operands()
    def __rshift__(self, other):
        """Denote that the expression is lower-bounded in the Lowener order.

        In other words, return a constraint that ``self - other`` is positive
        semidefinite.
        """
        result = self._rshift_implementation(other)

        if result is NotImplemented:
            result = other._lshift_implementation(self)

        return result

    # --------------------------------------------------------------------------
    # Backwards compatibility methods.
    # --------------------------------------------------------------------------

    @deprecated("2.0", useInstead="~picos.valuable.Valuable.valued")
    def is_valued(self):
        """Whether the expression is valued."""
        return self.valued

    @deprecated("2.0", useInstead="~picos.valuable.Valuable.value")
    def set_value(self, value):
        """Set the value of an expression."""
        self.value = value

    @deprecated("2.0", "PICOS treats all inequalities as non-strict. Using the "
        "strict inequality comparison operators may lead to unexpected results "
        "when dealing with integer problems.")
    def __lt__(self, exp):
        return self.__le__(exp)

    @deprecated("2.0", "PICOS treats all inequalities as non-strict. Using the "
        "strict inequality comparison operators may lead to unexpected results "
        "when dealing with integer problems.")
    def __gt__(self, exp):
        return self.__ge__(exp)


# --------------------------------------
__all__ = api_end(_API_START, globals())
