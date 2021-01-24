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
from . import polyutils as pu 
import scipy.stats as stats

class Hermite(PolyBase):
    """
    Probabilists Hermite polynomial

    Orthoganality:
    probabilists: \int Hm(x) Hn(x) exp(-x^2/2) dx = sqrt(2pi) n! 1{mn}
    physicists  : \int Hm(x) Hn(x) exp(-x^2  ) dx = sqrt(pi)  2**n  n! 1{mn}

    """

    def __init__(self, d=None, deg=None, coef=None, domain=None, window=None, multi_index='total', hem_type='probabilists'):
        self.multi_index = multi_index
        self.ndim = pu.check_int(d)
        self.deg  = pu.check_int(deg)
        self.hem_type  = hem_type.lower()
        self.name      = 'Hermite_e' if hem_type.startswith('prob') else 'Hermite'
        self.nickname  = 'Heme' if hem_type.startswith('prob') else 'Hem'
        self.dist_name = 'norm'
        self.weight    = self._wiener_askey_distribution()
        self.set_coef(coef)
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
        self.n_gauss = pu.check_int(n)

        ## for unspecified distribution parameters, default (loc, scale) = (0,1)
        for _ in range(len(loc), self.ndim):
            loc.append(0)
            scale.append(1)

        coords = []
        weight = []

        if self.hem_type.startswith('prob'):
            for iloc, iscale in zip(loc, scale):
                x, w = np.polynomial.hermite_e.hermegauss(self.n_gauss) 
                x = iloc + iscale* x
                w = iscale * w
                coords.append(x)
                weight.append(w)
        elif self.hem_type.startswith('phy'):
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

    def vandermonde(self, x, normalize=True):
        """
        Pseudo-Vandermonde matrix of given degree.
        Arguments:
            x: ndarray of shape(ndim, nsamples)
            normalize: boolean
        return:
            vandermonde matrix of shape(nsampels, deg)
            
        """
        x    = np.array(x, copy=0, ndmin=2) + 0.0
        d, n = x.shape
        assert (d == self.ndim), 'Expected input dimension {:d}, but {:d} given '.format(self.ndim, d)
        if self.hem_type == 'probabilists':
            vander_1d  = np.array([np.polynomial.hermite_e.hermevander(ix, self.deg) for ix in x])
        elif self.hem_type == 'physicists':
            vander_1d  = np.array([np.polynomial.hermite.hermvander(ix, self.deg) for ix in x])
        else:
            raise ValueError('hem_type is either probabilists or physicists')

        vander = np.ones((n, self.num_basis))
        ## basis_degree, list of tuples containing degree component for each basis function. i.e. (3,0,2) -> x1**3 + x2**0 + x3**2
        if self.basis_degree is None:
            self._update_basis()
        for i, ibasis_degree in enumerate(self.basis_degree):
            ### ibasis_degree = (l,m,n,k), assume ndim=4
            for idim, ideg in enumerate(ibasis_degree):
                ### (0,l), (1,m), (2,n), (3,k)
                vander[:,i] = vander[:,i] * vander_1d[idim,:,ideg]

        if normalize:
            vander = vander / np.sqrt(self.basis_norms)

        return vander

    def set_ndim(self, ndim):
        """
        set the dimension of polynomial
        """
        self.ndim = pu.check_int(ndim)
        self._update_basis()

    def set_degree(self, deg):
        """
        set polynomial degree order
        """
        self.deg = pu.check_int(deg)
        self._update_basis()

    def set_coef(self, coef):
        """
        set polynomial coef
        Arguments: None, or scalar, array-like 
        if coef is scalar, all polynomial coefficient will be assigned as that same value
        """
        self._update_basis()
        if coef is None:
            coef = None
        elif np.ndim(coef) == 0:
            coef = np.ones(self.num_basis) * coef + 0.0
        else:
            if len(coef) != self.num_basis:
                raise TypeError('Expected coefficients has length {}, but {} is given'.format(self.num_basis, len(coef)))
        self.coef = coef

    def _wiener_askey_distribution(self):
        """
        Return Askey-Wiener distributions
        """

        if self.ndim is None:
            weight = None
        elif self.hem_type.lower().startswith('prob'):
            weight = stats.norm(0,1)
        elif self.hem_type.lower().startswith('phy'):
            weight = stats.norm(0,np.sqrt(0.5))
        else:
            raise ValueError('UQRA.Hermite: {} not defined for hem_type '.format(hem_type))
        return weight

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
                norms_1d    = np.array([math.factorial(i) for i in range(self.deg+1)])
                basis       = []
                basis_norms = [] 
                ## calculate the ith multidimensional polynomial element of order(l,m,n)
                for ibasis_degree in self.basis_degree: 
                    ibasis = 1.0
                    inorms = 1.0
                    ## polynomial element (l,m,n)
                    for ideg in ibasis_degree:
                        ibasis = ibasis * np.polynomial.hermite_e.HermiteE.basis(ideg) 
                        inorms = inorms * norms_1d[ideg]
                    basis.append(ibasis)
                    basis_norms.append(inorms)
                self.basis = basis
                self.basis_norms = np.array(basis_norms)

            elif self.hem_type == 'physicists':
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
        d, n = x.shape ## (ndim, samples)
        if d != self.ndim:
            raise TypeError('Expected x has dimension {}, but {} is given'.format(self.ndim, d))
        if self.coef is None:
            self.coef = np.ones((self.num_basis,))


        size_of_array_4gb = 1e8/2.0
        ## size of largest array is of shape (n-k, k, k)
        if x.shape[1] * self.num_basis < size_of_array_4gb:
            vander  = self.vandermonde(x)
            y       = np.sum(vander * self.coef, -1)
        else:
            batch_size = math.floor(size_of_array_4gb/self.num_basis)  ## large memory is allocated as 8 GB
            y = []
            for i in range(math.ceil(x.shape[1]/batch_size)):
                idx_beg = i*batch_size
                idx_end = min((i+1) * batch_size, x.shape[1])
                x_      = x[:,idx_beg:idx_end]
                # vander_ = self.vandermonde(x_)
                vander_ = self.vandermonde(x_)
                y      += list(np.sum(vander_ * self.coef, -1))
            y = np.array(y) 
        return y

    def __str__(self):
        self._update_basis()
        if self.coef is None:
            self.coef = np.ones((self.num_basis,))
        return str(sum([ibasis * icoef for ibasis, icoef in zip(self.basis, self.coef)]))
        # raise NotImplementedError




