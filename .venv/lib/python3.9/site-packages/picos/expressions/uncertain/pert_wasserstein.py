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

"""Implements :class:`WassersteinAmbiguitySet`."""

from collections import namedtuple

import numpy

from ... import glyphs
from ...apidoc import api_end, api_start
from ..data import cvx2np
from ..exp_affine import AffineExpression
from ..samples import Samples
from .perturbation import Perturbation, PerturbationUniverse

_API_START = api_start(globals())
# -------------------------------


class WassersteinAmbiguitySet(PerturbationUniverse):
    r"""A wasserstein ambiguity set centered at a discrete distribution.

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

    Formally, this class can describe discrepancy-based ambiguity sets of the
    form

    .. math::

        \mathcal{P} = \left\{
            \Xi \in \mathcal{M} ~\middle|~
            \operatorname{W}_p(\Xi, \Xi_\text{N}) \leq \epsilon
        \right\}

    where discrepancy from the discrete nominal distribution

    .. math::

        \Xi_\text{N} = \sum_{i = 1}^N w_i \delta_{\xi_{(i)}} \in \mathcal{M}

    is measured with respect to the Wasserstein distance of order
    :math:`p \geq 1`,

    .. math::

        \operatorname{W}_p(\Xi, \Xi') =
            {\left(
                \inf_{\Phi \in \Pi(\Xi, \Xi')}
                \int_{\mathbb{R}^m \times \mathbb{R}^m}
                \lVert \phi - \phi' \rVert^p \;
                \Phi(
                    \mathop{}\!\mathrm{d} \phi
                    \times
                    \mathop{}\!\mathrm{d} \phi')
            \right)}^{\frac{1}{p}},

    where

    1. :math:`\mathcal{M}` is the set of all Borel probability measures on
       :math:`\mathbb{R}^n` for some :math:`n \in \mathbb{Z}_{\geq 1}`,
    2. :math:`\Pi(\Xi, \Xi')` denotes the set of all couplings of :math:`\Xi`
       and :math:`\Xi'`,
    3. :math:`\xi_{(i)} \in \mathbb{R}^n` for all :math:`i \in [N]` are the
       :math:`N \in \mathbb{Z}_{\geq 1}` *samples* comprising the support of
       :math:`\Xi_\text{N}`,
    4. :math:`w_i \in \mathbb{R}_{\geq 0}` are *weights* denoting the nominal
       probabilitiy mass at :math:`\xi_{(i)}` for all :math:`i \in [N]`,
    5. :math:`\delta_{\xi_{(i)}}` denotes the Dirac delta function with unit
       mass at :math:`\xi_{(i)}` for all :math:`i \in [N]` and where
    6. :math:`\epsilon \in \mathbb{R}_{\geq 0}` controls the radius of the
       ambiguity set.

    :Supported functions:

    For :math:`p = 1`:

    1. A convex piecewise linear function :math:`f(x, \xi) = max_{i=1}^k a_i(x,
       \xi)` where :math:`a` is biaffine in :math:`x` and :math:`\xi` for all
       :math:`i \in [k]`. This can be written as ``picos.max([a_1, ..., a_k])``
       in Python.
    2. A concave piecewise linear function :math:`f(x, \xi) = min_{i=1}^k a_i(x,
       \xi)` where :math:`a` is biaffine in :math:`x` and :math:`\xi` for all
       :math:`i \in [k]`. This can be written as ``picos.min([a_1, ..., a_k])``
       in Python.

    For :math:`p = 2`:

    1. A squared norm :math:`f(x, \xi) = \lVert A(x, \xi) \rVert_F^2` where
       :math:`A` is biaffine in :math:`x` and :math:`\xi`. This can be written
       as ``abs(A)**2`` in Python.
    """

    def __init__(self, parameter_name, p, eps, samples, weights=1):
        r"""Create a :class:`WassersteinAmbiguitySet`.

        :param str parameter_name:
            Name of the random parameter :math:`\xi`.

        :param float p:
            The Wasserstein type/order parameter :math:`p`.

        :param float eps:
            The Wasserstein ball radius :math:`\epsilon`.

        :param samples:
            The support of the discrete distribution :math:`\Xi_\text{D}` given
            as the *samples* :math:`\xi_{(i)}`. The original shape of the
            samples determines the shape of :math:`\xi`.
        :type samples:
            aynthing recognized by :class:`~.samples.Samples`

        :param weights:
            A vector denoting the nonnegative weight (e.g. frequency or
            probability) of each sample. Its length must match the number of
            samples provided. The argument will be normalized such that its
            entries sum to one. Entries of zero will be dropped alongside their
            associated sample. The default value of ``1`` denotes the empirical
            distribution on the samples.

        .. warning::

            Duplicate samples are not detected and can impact performance. If
            duplicate samples are likely, make sure to detect them and encode
            their frequency in the weight vector.
        """
        # Load p.
        self._p = float(p)

        if self._p < 1:
            raise ValueError("The Wasserstein parameter p must be >= 1.")

        supported_p = (1, 2)
        if self._p not in supported_p:
            raise NotImplementedError("Currently, Wasserstein DRO is only "
                "supported for p in {}.".format(set(supported_p)))

        # Load epsilon.
        self._eps = float(eps)

        if self._eps < 0:
            raise ValueError("The Wasserstein ball radius must be nonnegative.")

        # Load the samples.
        self._samples = Samples(samples)

        # Load the normalized weights.
        w = AffineExpression.from_constant(weights, (len(self._samples), 1))
        w_np = numpy.ravel(cvx2np(w.value_as_matrix))

        if any(w_np < 0):
            raise ValueError(
                "The weight vector must be nonnegative everywhere.")

        if any(w_np == 0):
            if all(w_np == 0):
                raise ValueError("The weight vector must be nonzero.")

            nonzero = numpy.where(w_np != 0)[0].tolist()
            w = w[nonzero]
            self._samples = self._samples.select(nonzero)

        self._weights = (w / (w | 1)).renamed("w")

        assert len(self._samples) == len(self._weights)

        # Create the perturbation parameter.
        self._parameter = Perturbation(
            self, parameter_name, self._samples.original_shape)

    @property
    def p(self):
        """The Wasserstein order :math:`p`."""
        return self._p

    @property
    def eps(self):
        r"""The Wasserstein ball radius :math:`\epsilon`."""
        return self._eps

    @property
    def samples(self):
        """The registered samples as a :class:`~.samples.Samples` object."""
        return self._samples

    @property
    def weights(self):
        """The sample weights a constant PICOS vector."""
        return self._weights

    Subtype = namedtuple("Subtype", ("sample_dim", "sample_num", "p"))

    def _subtype(self):
        return self.Subtype(self._samples.dim, self._samples.num, self._p)

    def __str__(self):
        return "WAS(p={}, eps={}, N={})".format(
            self._p, self._eps, self._samples.num)

    @classmethod
    def _get_type_string_base(cls):
        return "Wasserstein Ambiguity Set"

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
