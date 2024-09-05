# ------------------------------------------------------------------------------
# Copyright (C) 2020 Maximilian Stahlberg
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

"""Implements :class:`MomentAmbiguitySet`."""

from collections import namedtuple

from ... import glyphs
from ...apidoc import api_end, api_start
from ..cone_trivial import TheField
from ..data import cvxopt_hpd, cvxopt_inverse, load_shape
from ..exp_affine import AffineExpression
from ..expression import Expression
from ..set_ellipsoid import Ellipsoid
from .perturbation import Perturbation, PerturbationUniverse

_API_START = api_start(globals())
# -------------------------------


class MomentAmbiguitySet(PerturbationUniverse):
    r"""A moment-uncertain description of a random perturbation parameter.

    :Model of uncertainty:

    As a distributional ambiguity set, an instance :math:`\mathcal{P}` of this
    class

    1. represents a safety region for a partially known (ambiguous) probability
       distribution :math:`\Xi \in \mathcal{P}` and
    2. provides a random, ambiguously distributed perturbation parameter
       :math:`\xi \sim \Xi` that can be used to define worst-case-expectation
       expressions of the form

       .. math::

           \mathop{(\max\;\textit{or}\;\min)}_{\Xi \in \mathcal{P}}
           \mathbb{E}_\Xi[f(x, \xi)]

       for a selection of functions :math:`f` and a decision variable :math:`x`.

    :Definition:

    Formally, this class can describe ambiguity sets of the form

    .. math::

        \mathcal{P} = \left\{
            \Xi \in \mathcal{M} ~\middle|~
            \begin{aligned}
                \mathbb{P}(\xi \in \mathcal{S}) &= 1, \\
                \left( \mathbb{E}[\xi] - \mu \right)^T \Sigma^{-1}
                    \left( \mathbb{E}[\xi] - \mu \right) &\leq \alpha, \\
                \mathbb{E}[(\xi - \mu)(\xi - \mu)^T] &\preceq \beta \Sigma
            \end{aligned}
        \right\}

    where

    1. :math:`\mathcal{M}` is the set of all Borel probability measures on
       :math:`\mathbb{R}^n`for some :math:`n \in \mathbb{Z}_{\geq 1}`,
    2. the sample space :math:`\mathcal{S} \subseteq \mathbb{R}^n` bounds the
       support of :math:`\Xi` and may be given as either :math:`\mathbb{R}^n`
       or as an :math:`n`-dimensional ellipsoid,
    3. :math:`\mu \in \mathbb{R}^n` and
       :math:`\Sigma \in \mathbb{S}^n` with :math:`\Sigma \succ 0` are the
       **nominal** mean and covariance matrix of :math:`\Xi`, respectively, and
    4. :math:`\alpha \geq 0` and :math:`\beta \geq 1` are meta-parameters
       bounding the uncertainty concerning the mean and the covariance matrix,
       respectively.

    Unless :math:`\mathcal{S} = \mathbb{R}^n`, this class can also represent the
    limit cases of :math:`\alpha \to \infty` and :math:`\beta \to \infty`
    denoting an unbounded mean and covariance matrix, respectively.

    .. note::

        While :math:`\alpha = 0` denotes that the nominal mean :math:`\mu` is
        certain, there is a subtle difference between setting :math:`\beta = 1`
        on the one hand and assuming a certain form for the covariance matrix on
        the other hand: In the former case, the worst case covariance matrix may
        be Lowener smaller than the nominal one. Setting a lower bound on the
        covarianve matrix is computationally difficult and not supported.

    :Supported functions:

    1. A squared norm :math:`f(x, \xi) = \lVert A(x, \xi) \rVert_F^2` where
       :math:`A` is biaffine in :math:`x` and :math:`\xi`. This can be written
       as ``abs(A)**2`` in Python.
    2. A convex piecewise linear function :math:`f(x, \xi) = max_{i=1}^k a_i(x,
       \xi)` where :math:`a` is biaffine in :math:`x` and :math:`\xi` for all
       :math:`i \in [k]`. This can be written as ``picos.max([a_1, ..., a_k])``
       in Python.
    3. A concave piecewise linear function :math:`f(x, \xi) = min_{i=1}^k a_i(x,
       \xi)` where :math:`a` is biaffine in :math:`x` and :math:`\xi` for all
       :math:`i \in [k]`. This can be written as ``picos.min([a_1, ..., a_k])``
       in Python.

    :Example:

    We show that for unbounded mean and support (i.e. :math:`\alpha \to \infty`
    and :math:`\beta \to \infty`) and for a finite samples space :math:`S`, this
    distributional ambiguity set can be used in the context of classical
    (non-distributional) robust optimization applied to least squares problems.

    >>> from picos import Constant, diag, Ellipsoid, Problem, RealVariable
    >>> from picos.uncertain import ConicPerturbationSet, MomentAmbiguitySet
    >>> import numpy
    >>> numpy.random.seed(1)
    >>> # Setup data and variables of the nominal least squares problem.
    >>> n = 3
    >>> A = Constant("A", numpy.random.random((n, n)))
    >>> b = Constant("b", numpy.random.random(n))
    >>> x = RealVariable("x", n)
    >>> # Setup an ellipsoid S bounding the uncertainty in both models.
    >>> N = n**2
    >>> S = Ellipsoid(N, diag(range(1, N + 1)), range(N))
    >>> # Define and solve both robust models.
    >>> U1 = ConicPerturbationSet.from_constraints(
    ...     "Y", RealVariable("Y", N) << S)
    >>> U2 = MomentAmbiguitySet("Z", N, alpha=None, beta=None, sample_space=S)
    >>> Y = U1.parameter.reshaped((n, n))
    >>> Z = U2.parameter.reshaped((n, n))
    >>> P1, P2 = Problem(), Problem()
    >>> P1.objective = "min", abs((A + Y)*x - b)
    >>> P2.objective = "min", abs((A + Z)*x - b)**2
    >>> _ = P1.solve(solver="cvxopt")
    >>> x1 = ~x  # Save current value of x as a constant PICOS expression x1.
    >>> _ = P2.solve(solver="cvxopt")
    >>> x2 = ~x
    >>> # Verify that both models yield the same robust regression vector.
    >>> x1.equals(x2, relTol=1e-4)
    True
    """

    def __init__(self, parameter_name, shape, nominal_mean=0,
            nominal_covariance="I", alpha=0, beta=1, sample_space=None):
        r"""Create a :class:`MomentAmbiguitySet`.

        :param str parameter_name:
            Name of the random parameter :math:`\xi`.

        :param shape:
            Shape of :math:`\xi`. Must characterize a column vector. If
            :obj:`None`, then the shape is inferred from the nominal mean.
        :type shape:
            anything recognized by :func:`~picos.expressions.data.load_shape`

        :param nominal_mean:
            The nominal mean :math:`\mu` of the ambiguous distribution
            :math:`\Xi`.
        :type nominal_mean:
            anything recognized by :func:`~picos.expressions.data.load_data`

        :param nominal_covariance:
            The nominal covariance matrix :math:`\Sigma` of the ambiguous
            distribution :math:`\Xi`.
        :type nominal_covariance:
            anything recognized by :func:`~picos.expressions.data.load_data`

        :param float alpha:
            The parameter :math:`\alpha \geq 0` bounding the uncertain mean.
            The values :obj:`None` and ``float("inf")`` denote an unbounded
            mean.

        :param float beta:
            The parameter :math:`\beta \geq 1` bounding the uncertain covariance
            matrix. The values :obj:`None` and ``float("inf")`` denote unbounded
            covariances.

        :param sample_space:
            The sample space :math:`\mathcal{S}`. If this is :obj:`None` or an
            instance of :class:`~picos.TheField` (i.e. :math:`\mathbb{R}^n`),
            then the support of :math:`\Xi` is unbounded. If this is an
            :math:`n`-dimensional instance of :class:`~picos.Ellipsoid`, then
            the support of :math:`\Xi` is a subset of that ellipsoid.
        :type sample_space:
            :obj:`None` or :class:`~picos.TheField` or :class:`~picos.Ellipsoid`
        """
        # Load the dimensionality.
        shape = load_shape(shape)
        if shape[1] != 1:
            raise ValueError(
                "The perturbation parameter must be a column vector; a shape of"
                " {} is not supported.".format(glyphs.shape(shape)))
        self._dim = n = shape[0]

        # Load the nominal mean.
        if not isinstance(nominal_mean, Expression):
            nominal_mean = AffineExpression.from_constant(
                nominal_mean, shape=shape)
        else:
            nominal_mean = nominal_mean.refined

            if not isinstance(nominal_mean, AffineExpression) \
            or not nominal_mean.constant:
                raise TypeError("The nominal mean must be a real constant.")

            if nominal_mean.shape != shape:
                raise TypeError("The nominal mean must be a {}-dimensional "
                    "column vector, got {} instead."
                    .format(n, glyphs.shape(nominal_mean.shape)))

        self._mean = nominal_mean

        # Load the nominal covariance matrix.
        if not isinstance(nominal_covariance, Expression):
            nominal_covariance = AffineExpression.from_constant(
                nominal_covariance, shape=(n, n))
        else:
            nominal_covariance = nominal_covariance.refined

            if not isinstance(nominal_covariance, AffineExpression) \
            or not nominal_covariance.constant:
                raise TypeError(
                    "The nominal covarance matrix must be a real constant.")

            if nominal_covariance.shape != (n, n):
                raise TypeError("The nominal covariance matrix must be a {} "
                    "matrix, got {} instead.".format(glyphs.shape((n, n)),
                    glyphs.shape(nominal_covariance.shape)))

        cov_value = nominal_covariance.safe_value_as_matrix

        if not cvxopt_hpd(cov_value):
            raise ValueError("The nominal covariance matrix is not symmetric "
                "positive definite.")

        self._cov = nominal_covariance

        # Compute the inverse of the nominal covariance matrix.
        self._cov_inv = AffineExpression.from_constant(
            cvxopt_inverse(cov_value), shape=(n, n))

        # Load alpha.
        if alpha in (None, float("inf")):
            self._alpha = None
        else:
            self._alpha = float(alpha)
            if self._alpha < 0:
                raise ValueError("The parameter alpha must be nonnegative.")

        # Load beta.
        if beta in (None, float("inf")):
            self._beta = None
        else:
            self._beta = float(beta)
            if self._beta < 1:
                raise ValueError("The parameter beta must be at least one.")

        # Load the sample space.
        if sample_space is None:
            self._ss = None
        elif isinstance(sample_space, TheField):
            if sample_space.dim is not None and sample_space.dim != self._dim:
                raise TypeError("Invalid sample space dimension.")

            self._ss = None
        elif isinstance(sample_space, Ellipsoid):
            if sample_space.dim != self._dim:
                raise TypeError("Invalid sample space dimension.")

            self._ss = sample_space
        else:
            raise TypeError("Invalid sample space type.")

        # Limit the permissible combinations of unboundedness.
        if self._ss is None and (self._alpha is None or self._beta is None):
            raise ValueError("If the support is unbounded, both the mean and "
                "the covariance matrix must be bounded.")

        # Create the perturbation parameter.
        self._parameter = Perturbation(self, parameter_name, shape)

    @property
    def dim(self):
        """The dimension :math:`n` of the sample space."""
        return self._dim

    @property
    def nominal_mean(self):
        r"""The nominal mean :math:`\mu` of the ambiguous distribution."""
        return self._mean

    @property
    def nominal_covariance(self):
        r"""The nominal covariance matrix :math:`\Sigma`."""
        return self._cov

    @property
    def alpha(self):
        r"""The parameter :math:`\alpha`.

        A value of :obj:`None` denotes :math:`\alpha \to \infty`.
        """
        return self._alpha

    @property
    def beta(self):
        r"""The parameter :math:`\beta`.

        A value of :obj:`None` denotes :math:`\beta \to \infty`.
        """
        return self._beta

    @property
    def sample_space(self):
        r"""The sample space (bound on support) :math:`\mathcal{S}`.

        A value of :obj:`None` means :math:`\mathcal{S} = \mathbb{R}^n`.
        """
        return self._ss

    Subtype = namedtuple("Subtype", (
        "dim",
        "bounded_mean",
        "bounded_covariance",
        "bounded_support"))

    def _subtype(self):
        return self.Subtype(
            self._dim,
            self._alpha is not None,
            self._beta is not None,
            self._ss is not None)

    def __str__(self):
        return "MAS(mu={}, cov={}, a={}, b={}, S={})".format(
            self._mean.string,
            self._cov.string,
            self._alpha,
            self._beta,
            self._ss.string if self._ss is not None else None)

    @classmethod
    def _get_type_string_base(cls):
        return "Moment Ambiguity Set"

    def __repr__(self):
        return glyphs.repr2("{} {}".format(glyphs.shape(self._parameter.shape),
            self._get_type_string_base()), self.__str__())

    @property
    def distributional(self):
        """Implement for :class:`~.perturbation.PerturbationUniverse`."""
        return True

    @property
    def parameter(self):
        r"""The random perturbation parameter :math:`\xi`."""
        return self._parameter


# --------------------------------------
__all__ = api_end(_API_START, globals())
