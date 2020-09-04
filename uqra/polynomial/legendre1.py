#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
import itertools, math
import scipy.stats as stats
from ._polybase import PolyBase

class Legendre1(PolyBase):
    """
    Legendre polynomial

    Orthogonality:
    domain: [-1,1]
    \int Pm(x) Pn(x) dx = 2/(2n+1) 1{mn} 
    """

    def __init__(self, d=None, deg=None, coef=None, domain=None, window=None, multi_index='total'):
        super().__init__(d=d, deg=deg, coef=coef, domain=domain, window=window, multi_index=multi_index)
        self.name       = 'Legendre'
        self.nickname   = 'Leg'
        self.dist_name  = 'Uniform'
        self.dist_u     = None if self.ndim is None else [stats.uniform(-1,2), ] * self.ndim 
        self._update_basis()

    def weight(self, x):
        """
        Weight function of the Jacobi polynomials.
        The weight function is :math: (1-x)^a * (1+x)^b and the interval of integration is
        :math:`[-1, 1]`. The Jacobi polynomials are orthogonal, but not
        normalized, with respect to this weight function.
        Parameters
        ----------
        x : array_like
           Values at which the weight function will be computed.
        Returns
        -------
        w : ndarray
           The weight function at `x`.
        Notes
        -----
        """
        w = x *0.0 + 1.0 
        return w

    def weight_normalization(self):
        """
        Normalization value to make the weight function being a Beta distribution as defined in scipy.stats.beta
        if x ~ Beta(alpha+1, beta+1)
        1-2x ~ w(x, alpha, beta) * C(alpha, beta)
        """
        return 0.5

    def orthogonal_normalization(self, n):
        """
        Calculate the orthogonality value for Jacobi polynoial, C
        int Pm(x) * Pn(x) * w(x) dx = C* delta_{ij} on [-1,1]
        """
        c = 2.0/(2 * n + 1)
        return c

    def orthopoly1d(self, deg):
        r"""Jacobi polynomial.
        Defined to be the solution of
        .. math::
            (1 - x^2)\frac{d^2}{dx^2}P_n^{(\alpha, \beta)}
              + (\beta - \alpha - (\alpha + \beta + 2)x)
                \frac{d}{dx}P_n^{(\alpha, \beta)}
              + n(n + \alpha + \beta + 1)P_n^{(\alpha, \beta)} = 0
        for :math:`\alpha, \beta > -1`; :math:`P_n^{(\alpha, \beta)}` is alpha
        polynomial of degree :math:`n`.
        Parameters
        ----------
        deg  : int
            Degree of the polynomial.
        alpha: float
            Parameter, must be greater than -1.
        beta : float
            Parameter, must be greater than -1.
        monic : bool, optional
            If `True`, scale the leading coefficient to be 1. Default is
            `False`.
        Returns
        -------
        P : orthopoly1d
            Jacobi polynomial.
        Notes
        -----
        For fixed :math:`\alpha, \beta`, the polynomials
        :math:`P_n^{(\alpha, \beta)}` are orthogonal over :math:`[-1, 1]`
        with weight function :math:`(1 - x)^\alpha(1 + x)^\beta`.
        """
        orthpoly = np.polynomial.legendre.Legendre.basis(deg)
        return orthpoly

    def vander1d(self, x, deg):
        """Pseudo-Vandermonde matrix of given degree.
        Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
        `x`. The pseudo-Vandermonde matrix is defined by
        .. math:: V[..., i] = L_i(x)
        where `0 <= i <= deg`. The leading indices of `V` index the elements of
        `x` and the last index is the degree of the Legendre polynomial.
        If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
        array ``V = legvander(x, n)``, then ``np.dot(V, c)`` and
        ``legval(x, c)`` are the same up to roundoff. This equivalence is
        useful both for least squares fitting and for the evaluation of a large
        number of Legendre series of the same degree and sample points.
        Parameters
        ----------
        x : array_like
            Array of points. The dtype is converted to float64 or complex128
            depending on whether any of the elements are complex. If `x` is
            scalar it is converted to a 1-D array.
        deg : int
            Degree of the resulting matrix.
        Returns
        -------
        vander : ndarray
            The pseudo-Vandermonde matrix. The shape of the returned matrix is
            ``x.shape + (deg + 1,)``, where The last index is the degree of the
            corresponding Legendre polynomial.  The dtype will be the same as
            the converted `x`.
        """
        ideg = int(deg)
        if ideg != deg:
            raise ValueError("deg must be integer")
        if ideg < 0:
            raise ValueError("deg must be non-negative")

        x = np.array(x, copy=False, ndmin=1) + 0.0
        dims = (ideg + 1,) + x.shape
        dtyp = x.dtype
        v = np.empty(dims, dtype=dtyp)
        # Use forward recursion to generate the entries. This is not as accurate
        # as reverse recursion in this application but it is more efficient.
        v[0] = x*0 + 1
        if ideg > 0:
            v[1] = x
            for i in range(2, ideg + 1):
                v[i] = (v[i-1]*x*(2*i - 1) - v[i-2]*(i - 1))/i
        return np.moveaxis(v, 0, -1)

    def vandermonde(self, x, normed=True):
        """
            Pseudo-Vandermonde matrix of given degree.
        Arguments:
            x, ndarray of shape(ndim, nsamples)
            normed: boolean. If true, each column is normalized such that \int_-1,1 Pm(x) Pm(x) f(x)dx = 1 w.r.t distribution f(x)
        """
        x        = np.array(x, copy=0, ndmin=2) + 0.0
        vander   = np.ones((x.shape[1], self.num_basis), x.dtype)
        vander1d = [self.vander1d(ix, self.deg) for ix in x]

        ### basis_degree, list of tuples containing degree component for each basis function. i.e. (3,0,2) -> x1**3 + x2**0 + x3**2
        if self.basis_degree is None:
            self._update_basis()
        for i, ibasis_degree in enumerate(self.basis_degree):
            ### ith polynomial, it is composed of ibasis_degree = (l,m,n)
            for idim, ideg in enumerate(ibasis_degree):
                vander1d_idim = vander1d[idim]
                vander[:,i] = vander[:,i] * vander1d_idim[:,ideg]
        if normed:
            vander = vander / np.sqrt(self.basis_norms)
        return vander

    def gauss_quadrature(self, n, loc=[], scale=[]):
        """
        Gauss-Legendre quadrature.
        Computes the sample points and weights for Gauss-HermiteE quadrature. 
        These sample points and weights will correctly integrate polynomials of degree 2*deg - 1 or less over the interval [-1, 1] with the weight function f(x) = 1 

        Parameters:	
        deg : int
        Number of sample points and weights. It must be >= 1.

        Returns:	
        x : ndarray
        1-D ndarray containing the sample points.

        y : ndarray
        1-D ndarray containing the weights.

        """
        super().gauss_quadrature(n)
        ## for unspecified distribution parameters, default (loc, scale) = (-1,2)
        ## tradition from scipy.stats
        for _ in range(len(loc), self.ndim):
            loc.append(-1)
            scale.append(2)

        coords = []
        weight = []
        for iloc, iscale in zip(loc, scale):
            x, w = np.polynomial.legendre.leggauss(self.n_gauss) 
            x = iloc + iscale/2.0*(x+1)
            w = iscale/2.0 * w
            coords.append(x)
            weight.append(w)

        x = np.array(list(itertools.product(*coords))).T
        x = x.reshape(self.ndim, -1)
        w = np.prod(np.array(list(itertools.product(*weight))).T, axis=0)
        w = np.squeeze(w)
        return x, w

    def vandermonde(self, x, normed=True):
        """
            Pseudo-Vandermonde matrix of given degree.
        Arguments:
            x, ndarray of shape(ndim, nsamples)
            normed: boolean. If true, each column is normalized such that \int_-1,1 Pm(x) Pm(x) dx = 1
        """
        x    = np.array(x, copy=0, ndmin=2) + 0.0
        d, n = x.shape
        assert (d == self.ndim), 'Input dimension is {:d}, given {:d}'.format(self.ndim, d)
        vander      = np.ones((n, self.num_basis), x.dtype)
        vander_ind  = np.array([np.polynomial.legendre.legvander(ix, self.deg) for ix in x])

        ### basis_degree, list of tuples containing degree component for each basis function. i.e. (3,0,2) -> x1**3 + x2**0 + x3**2
        if self.basis_degree is None:
            self._update_basis()
        for i, ibasis_degree in enumerate(self.basis_degree):
            ### ith polynomial, it is composed of ibasis_degree = (l,m,n)
            for idim, ideg in enumerate(ibasis_degree):
                vander[:,i] = vander[:,i] * vander_ind[idim,:,ideg]
        if normed:
            vander = vander / np.sqrt(self.basis_norms)
        return vander

    def set_ndim(self, ndim):
        """
        set the dimension of polynomial
        """
        self.ndim = super()._check_int(ndim)
        self._update_basis()

    def set_degree(self, deg):
        """
        set polynomial degree order
        """
        self.deg = super()._check_int(deg)
        self._update_basis()

    def set_coef(self, coef):
        """
        set coefficients for orthoNORMAL basis
        """
        self._update_basis()
        self.coef = coef 
        self.coef_normalized = coef * np.sqrt(self.basis_norms)

    def _update_basis(self):
        """
        Return a list of polynomial basis function with specified degree and multi_index rule
        """
        ### get self.basis_degree and self.num_basis
        super()._update_basis()
        if self.basis_degree is None:
            self.basis       = None
            self.basis_norms = None
        else:
            norms_1d    = np.array([1/(2*i + 1) for i in range(self.deg+1)])
            basis       = []
            basis_norms = [] 
            for ibasis_degree in self.basis_degree:
                ibasis = 1.0
                inorms = 1.0
                for ideg in ibasis_degree:
                    ibasis = ibasis * self.orthopoly1d(ideg) 
                    inorms = inorms * self.orthogonal_normalization(ideg) * self.weight_normalization()
                basis.append(ibasis)
                basis_norms.append(inorms)
            self.basis = basis
            self.basis_norms = np.array(basis_norms)

        return self.basis, self.basis_norms

    def __call__(self, x):
        """
        Evaluate polynomials at given values x
        Arguments:
            x, ndarray of shape (ndim, nsamples)
        """
        x = np.array(x, ndmin=2, copy=False)
        y = np.zeros(x.shape[1]) + 0.0
        for icoef, ibasis_degree in zip(self.coef, self.basis_degree):
            ### e.g. ibasis_degree = (2,0,4)
            ibasis_y = 1.0
            for ix, ideg in zip(x, ibasis_degree):
                orthpoly_ = self.orthopoly1d(ideg)
                ibasis_y *= orthpoly_(ix)
            y += ibasis_y * icoef
        return y

    def __str__(self):
        self._update_basis()
        if self.coef is None:
            self.coef = np.ones((self.num_basis,))
        return str(sum([ibasis * icoef for ibasis, icoef in zip(self.basis, self.coef)]))
        # raise NotImplementedError

