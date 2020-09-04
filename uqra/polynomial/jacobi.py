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
import scipy
import scipy.stats as stats
from ._polybase import PolyBase
import warnings

class Jacobi(PolyBase):
    """
    Jacobi polynomials corresponds to Beta(a,b) on domain [-1,1]
    Orthogonality: 
    \int Pm(x) Pn(x) w(x) dx = C 1{mn} 
    w(x) = (1-x)**alpha * (1+x)**beta

    For fixed Beta(a, b), the polynomials are orthogonal over[-1,1] with weight function (1-x)**alpha * (1+x)**beta
    alpha = a-1, beta = b-1
    domain: [-1,1]

    """

    def __init__(self, a=None, b=None, d=None, deg=None, coef=None, multi_index='total'):
        """
        a ,b: shape parameters of Beta distributions. Note that the corresponding weight function is w(a-1,b-1)

        """
        super().__init__(d=d, deg=deg, coef=coef, multi_index=multi_index)
        self.name       = 'Jacobi'
        self.nickname   = 'Jac'
        self.dist_name  = 'Beta'
        if a is None or b is None:
            pass
        else:
            self.a, self.b  = self._update_beta_parameters(a,b)
            ## weight function (1-x)**alpha * (1+x)**beta
            self.dist_u = [stats.beta(ia, ib, loc=-1, scale=2) for ia, ib in zip(self.a, self.b)]
            self.alpha  = self.b - 1
            self.beta   = self.a - 1 
            self._update_basis()
            self.coef   = coef if coef is not None else np.ones(self.num_basis)


    def weight(self, x):
        """
        Weight function of the Jacobi polynomials.
        The weight function is :math: (1-x)^alpha * (1+x)^beta and the interval of integration is
        :math:`[-1, 1]`. The Jacobi polynomials are orthogonal, but not
        normalized, with respect to this weight function.
        Parameters
        ----------
        x : array_like of shape (ndim, nsamples)
           Values at which the weight function will be computed.
        Returns
        -------
        w : ndarray
           The weight function at `x`.
        Notes
        -----
        """
        w = np.array([(1-ix)**ialpha * (1+ix)**ibeta + 0.0 for ix, ialpha, ibeta in zip(x, self.alpha, self.beta)])
        return w

    def weight_normalization(self, alpha, beta):
        """
        Normalization value to make the weight function being a Beta distribution as defined in scipy.stats.beta
        if x ~ Beta(alpha+1, beta+1)
        1-2x ~ w(x, alpha, beta) * C(alpha, beta)
        """
        B = lambda a, b : scipy.special.gamma(a) * scipy.special.gamma(b) / scipy.special.gamma(a+b)
        c = 1.0/ (2**(alpha+beta+1)*B(beta+1,alpha+1))
        return c

    def orthogonal_normalization(self, n, alpha, beta):
        """
        Calculate the orthogonality value for Jacobi polynoial, C
        int Pm(1-2x) * Pn(1-2x) * w(x) dx = C* delta_{ij} on [0,1]
        """
        c1 = 2**(alpha+beta+1)
        c2 = (2 * n + alpha + beta + 1)
        c3 = scipy.special.gamma(n+alpha+1)*scipy.special.gamma(n+beta+1)
        c4 = scipy.special.gamma(n+alpha+beta+1)*scipy.special.factorial(n)
        c  = c1 * c3 / c2 / c4
        return c

    def orthopoly1d(self, deg, alpha, beta, monic=False):
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

        Orthogonality:
        For fixed (a, b), the polynomials are orthogonal over[-1,1] with weight function (1-x)^a(1+x)^b
        domain: [-1,1]
        \int Pm(x) Pn(x) w(x) dx =  1{mn} 
        """
        return scipy.special.jacobi(deg, alpha, beta, monic=monic)

    def vander1d(self, x, deg, alpha, beta):
        """Pseudo-Vandermonde matrix of given degree.
        Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
        `x`. The pseudo-Vandermonde matrix is defined by
        .. math:: V[..., i] = L_i(x)
        where `0 <= i <= deg`. The leading indices of `V` index the elements of
        `x` and the last index is the degree of the Jacobi polynomial.
        If `c` is alpha 1-D array of coefficients of length `n + 1` and `V` is the
        array ``V = vander1d(x, n)``, then ``np.dot(V, c)`` and
        ``legval(x, c)`` are the same up to roundoff. This equivalence is
        useful both for least squares fitting and for the evaluation of alpha large
        number of Jacobi series of the same degree and sample points.
        Parameters
        ----------
        x : array_like
            Array of points. The dtype is converted to float64 or complex128
            depending on whether any of the elements are complex. If `x` is
            scalar it is converted to alpha 1-D array.
        deg : int
            Degree of the resulting matrix.
        Returns
        -------
        vander : ndarray
            The pseudo-Vandermonde matrix. The shape of the returned matrix is
            ``x.shape + (deg + 1,)``, where The last index is the degree of the
            corresponding Jacobi polynomial.  The dtype will be the same as
            the converted `x`.
        """
        ideg = int(deg)
        if ideg != deg:
            raise ValueError("deg must be integer")
        if ideg < 0:
            raise ValueError("deg must be non-negative")

        x = np.array(x, copy=0, ndmin=1) + 0.0

        dims = (ideg + 1,) + x.shape
        dtyp = x.dtype
        v = np.empty(dims, dtype=dtyp)
        # Use forward recursion to generate the entries. This is not as accurate
        # as reverse recursion in this application but it is more efficient.
        v[0] = x*0 + 1
        if ideg > 0:
            v[1] = x
            for i in range(2, ideg + 1):
                c_i, c_i_1, c_i_2 = self._three_terms_recursive_coefs(x, i, alpha, beta)
                v[i] = (v[i-1]*c_i_1 + v[i-2]*c_i_2)/c_i
        return np.rollaxis(v, 0, v.ndim)

    def vandermonde(self, x, normed=True):
        """
            Pseudo-Vandermonde matrix of given degree.
        Arguments:
            x, ndarray of shape(ndim, nsamples)
            normed: boolean. If true, each column is normalized such that \int_-1,1 Pm(x) Pm(x) f(x)dx = 1 w.r.t distribution f(x)
        """
        x        = np.array(x, copy=0, ndmin=2) + 0.0
        vander   = np.ones((x.shape[1], self.num_basis), x.dtype)
        vander1d = [self.vander1d(ix, self.deg, ialpha, ibeta) for ix, ialpha, ibeta in zip(x, self.alpha, self.beta)]

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
        update coefficients for polynomial basis
        
        coef: coef values are for orthogonal but not normalized basis corresponding to weight function
        coef_normalized: coef values are for each orthnormal basis corresponding to distribution function
        """
        self._update_basis()
        self.coef = coef 
        self.coef_normalized = coef * np.sqrt(self.basis_norms)

    def gauss_quadrature(self, n, loc=[], scale=[]):
        """
        Gauss-Jacobi quadrature.
        Computes the sample points and weights for Gauss-Jacobi quadrature. 
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
        raise NotImplementedError
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

    def _three_terms_recursive_coefs(self, x, deg, alpha, beta):
        """
        Three-term recursive formula:
        c_n * P_n(x) = c_{n-1} * P_{n-1}(x) + c_{n-2} * P_{n-2}

        """
        c_n   =  2*deg * (deg + alpha +beta) * (2*deg + alpha + beta -2)
        c_n_1 = (2*deg + alpha + beta -1) * ((2*deg +alpha + beta)*(2*deg +alpha +beta-2) * x+ alpha**2 - beta**2)
        c_n_2 =-2*(deg + alpha -1)*(deg + beta - 1)*(2*deg + alpha + beta)

        return c_n, c_n_1, c_n_2



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
            basis       = []
            basis_norms = [] ## normalization const normalizes the orthogonal polynomials, w.r.t. DISTRIBUTION functions 
            for ibasis_degree in self.basis_degree:
                ### e.g. ibasis_degree = (2,0,4)
                ibasis = 1.0
                inorms = 1.0
                for ialpha, ibeta, ideg in zip(self.alpha, self.beta, ibasis_degree):
                    with warnings.catch_warnings():
                            # FutureWarning: In the future extra properties will not be 
                            # copied across when constructing one poly1d from another
                            warnings.filterwarnings("ignore")
                            ibasis = ibasis * self.orthopoly1d(ideg, ialpha, ibeta) 
                    inorms = inorms * self.orthogonal_normalization(ideg, ialpha, ibeta) \
                            *self.weight_normalization(ialpha, ibeta)
                basis.append(ibasis)
                basis_norms.append(inorms)
            self.basis = basis
            self.basis_norms = np.array(basis_norms)

        return self.basis, self.basis_norms

    def _update_beta_parameters(self, a, b):
        """
        making a, b iterable 
        """
        a = np.asarray(a)
        b = np.asarray(b)
        if np.ndim(a) == 0:
            a = a * np.ones(self.ndim) 
        elif np.ndim(a) == 1:
            a = np.asarray(a)
            if a.size != self.ndim:
                raise ValueError(' {:d} a-parameters are expected for Beta(a,b) distributions \
                        but {:d} were given'.format(self.ndim, a.size))
        else:
            raise ValueError

        if np.ndim(b) == 0:
            b = b * np.ones(self.ndim) 
        elif np.ndim(b) == 1:
            if b.size != self.ndim:
                raise ValueError(' {:d} b-parameters are expected for Beta(a,b) distributions \
                        but {:d} were given'.format(self.ndim, b.size))
        else:
            raise ValueError

        return a, b



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
            for ix, ialpha, ibeta, ideg in zip(x, self.alpha, self.beta, ibasis_degree):
                orthpoly_ = self.orthopoly1d(ideg, ialpha, ibeta)
                ibasis_y *= orthpoly_(ix)
            y += ibasis_y * icoef
        return y

    def __str__(self):


        try:
            self._update_basis()
            if self.coef is None:
                self.coef = np.ones((self.num_basis,))
            basis = 0.0
            for ibasis, icoef in zip(self.basis, self.coef):
                basis += ibasis * icoef
            jacobi_str = '\n'+str(sum([ibasis * icoef for ibasis, icoef in zip(self.basis, self.coef)]))
        except TypeError:
            jacobi_str = 'Jacobi()'
            
        return jacobi_str 

