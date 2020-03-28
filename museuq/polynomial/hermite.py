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
from ._polybase import PolyBase

class Hermite(PolyBase):
    """
    Probabilists Hermite polynomial

    Orthoganality:
    probabilists: \int Hm(x) Hn(x) exp(-x^2/2) dx = sqrt(2pi) n! 1{mn}
    physicists  : \int Hm(x) Hn(x) exp(-x^2  ) dx = sqrt(pi)  2**n  n! 1{mn}

    """

    def __init__(self, d=None, deg=None, coef=None, domain=None, window=None, multi_index='total', hem_type='probabilists'):
        super().__init__(d=d, deg=deg, coef=coef, domain=domain, window=window, multi_index=multi_index)
        self.name = 'Hermite'
        self.hem_type = hem_type
        self.nickname = 'Heme' if hem_type.lower() == 'probabilists' else 'Hem'
        self._update_basis()

    def gauss_quadrature(self, n, loc=[], scale=[]):
        """
        Gauss-HermiteE quadrature.
        Computes the sample points and weights for Gauss-HermiteE quadrature. 
        These sample points and weights will correctly integrate polynomials of degree 2*deg - 1 or less over the interval [-\inf, \inf] with the weight function f(x) = \exp(-x^2/2) for probabilists and weight function f(x) = \exp(-x^2) for physicists

        Parameters:	
        deg : int
        Number of sample points and weights. It must be >= 1.

        Returns:	
        x : ndarray
        1-D ndarray containing the sample points.

        y : ndarray
        1-D ndarray containing the weights.

        """
        super().gauss_quadrature(n) ## check n and assign self.n_gauss

        ## for unspecified distribution parameters, default (loc, scale) = (0,1)
        for _ in range(len(loc), self.ndim):
            loc.append(0)
            scale.append(1)

        coords = []
        weight = []

        if self.hem_type == 'probabilists':
            for iloc, iscale in zip(loc, scale):
                x, w = np.polynomial.hermite_e.hermegauss(self.n_gauss) 
                x = iloc + iscale* x
                w = iscale * w
                coords.append(x)
                weight.append(w)
        elif self.hem_type == 'physicists':
            for iloc, iscale in zip(loc, scale):
                x, w = np.polynomial.hermite.hermgauss(self.n_gauss) 
                x = iloc + iscale* x
                w = iscale * w
                coords.append(x)
                weight.append(w)
        else:
            raise ValueError('hem_type is either probabilists or physicists')

        x = np.array(list(itertools.product(*coords))).T
        x = x.reshape(self.ndim, -1)
        w = np.prod(np.array(list(itertools.product(*weight))).T, axis=0)
        w = np.squeeze(w)
        return x, w

    def vandermonde(self, x, normed=True):
        """
        Pseudo-Vandermonde matrix of given degree.
        Arguments:
            x: ndarray of shape(ndim, nsamples)
            normed: boolean
            
        return:
            vandermonde matrix of shape(nsampels, deg)
            
        """
        x    = np.array(x, copy=0, ndmin=2) + 0.0
        d, n = x.shape
        assert (d == self.ndim), 'Expected input dimension {:d}, but {:d} given '.format(self.ndim, d)
        vander      = np.ones((n, self.num_basis), x.dtype)
        if self.hem_type == 'probabilists':
            vander_ind  = np.array([np.polynomial.hermite_e.hermevander(ix, self.deg) for ix in x])
        elif self.hem_type == 'physicists':
            vander_ind  = np.array([np.polynomial.hermite.hermvander(ix, self.deg) for ix in x])
        else:
            raise ValueError('hem_type is either probabilists or physicists')

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
        self._update_basis()
        if len(coef) != self.num_basis:
            raise TypeError('Expected coefficients has length {}, but {} is given'.format(self.num_basis, len(coef)))
        self.coef = coef

    def _update_basis(self):
        """
        Return a list of polynomial basis function with specified degree and multi_index rule

        
        """
        ### get self.basis_degree and self.num_basis
        ###    - basis_degree, list of tuples containing degree component for each basis function. i.e. (3,0,2) -> x1**3 + x2**0 + x3**2
        super()._update_basis()
        if self.basis_degree is None:
            self.basis        = None
            self.basis_norms  = None
        else:
            if self.hem_type == 'probabilists':
                self.basis_norms_const = np.sqrt(2* np.pi)
                norms_1d    = np.array([math.factorial(i) for i in range(self.deg+1)])
                basis       = []
                basis_norms = [] 
                for ibasis_degree in self.basis_degree:
                    ibasis = 1.0
                    inorms = 1.0
                    for ideg in ibasis_degree:
                        ibasis = ibasis * np.polynomial.hermite_e.HermiteE.basis(ideg) 
                        inorms = inorms * norms_1d[ideg]
                    basis.append(ibasis)
                    basis_norms.append(inorms)
                self.basis = basis
                self.basis_norms = np.array(basis_norms)

            elif self.hem_type == 'physicists':
                self.basis_norms_const = np.sqrt(np.pi)
                norms_1d    = np.array([math.factorial(i) * 2**i for i in range(self.deg+1)])
                basis       = []
                basis_norms = [] 
                for ibasis_degree in self.basis_degree:
                    ibasis = 1.0
                    inorms = 1.0
                    for ideg in ibasis_degree:
                        ibasis = ibasis * np.polynomial.hermite.Hermite.basis(ideg) 
                        inorms = inorms * norms_1d[ideg]
                    basis.append(ibasis)
                    basis_norms.append(inorms)
                self.basis = basis
                self.basis_norms = np.array(basis_norms)
            else:
                raise ValueError('hem_type is either probabilists or physicists')

        return self.basis, self.basis_norms
    def __call__(self, x):
        """
        Evaluate polynomials at given values x
        Arguments:
            x, ndarray of shape (ndim, nsamples)
        """
        self._update_basis()
        x = np.array(x, copy=False, ndmin=2)
        vander = self.vandermonde(x, normed=False)
        d, n = x.shape ## (ndim, samples)
        if d != self.ndim:
            raise TypeError('Expected x has dimension {}, but {} is given'.format(self.ndim, d))
        if self.coef is None:
            self.coef = np.ones((self.num_basis,))
        y = np.sum(vander * self.coef, -1)
        return y

    def __str__(self):
        self._update_basis()
        if self.coef is None:
            self.coef = np.ones((self.num_basis,))
        return str(sum([ibasis * icoef for ibasis, icoef in zip(self.basis, self.coef)]))
        # raise NotImplementedError




