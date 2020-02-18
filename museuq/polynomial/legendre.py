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

class Legendre(PolyBase):
    """
    Legendre polynomial

    Orthogonality:
    domain: [-1,1]
    \int Pm(x) Pn(x) dx = 1/(2n+1) 1{mn} 
    """

    def __init__(self, d, deg, coef=None, domain=None, window=None, multi_index='total'):
        super().__init__(d, deg, coef=coef, domain=domain, window=window, multi_index=multi_index)
        self.nickname = 'leg'


    def get_basis(self):
        """
        Return a list of polynomial basis function with specified degree and multi_index rule
        """
        super().get_basis()
        self.basis = []
        for ibasis_degree in self.basis_degree:
            ibasis = 1
            print(ibasis_degree)
            for p in ibasis_degree:
                ibasis = ibasis * np.polynomial.legendre.Legendre.basis(p) 
            self.basis.append(ibasis)
        return self.basis

    def gauss_quadrature(self, n, loc=[], scale=[]):
        """
        Gauss-Legendre quadrature.
        Computes the sample points and weights for Gauss-HermiteE quadrature. 
        These sample points and weights will correctly integrate polynomials of degree 2*deg - 1 or less over the interval [-\inf, \inf] with the weight function f(x) = \exp(-x^2/2).

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

        ## for unspecified distribution parameters, default (loc, scale) = (0,1)
        for _ in range(len(loc), self.ndim):
            loc.append(0)
            scale.append(1)

        coords = []
        weight = []
        print(loc, scale)
        for iloc, iscale in zip(loc, scale):
            x, w = np.polynomial.legendre.leggauss(self.n_gauss) 
            x = iloc + iscale* x
            w = iscale * w
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
        """
        x    = np.array(x, copy=0, ndmin=2) + 0.0
        d, n = x.shape
        assert (d == self.ndim), 'Input dimension is {:d}, given {:d}'.format(self.ndim, d)
        vander = np.ones((n, self.num_basis), x.dtype)
        self.basis_norms = np.array([math.sqrt(1/(2*i+1)) for i in range(self.deg+1)])
        vander_ind = np.array([np.polynomial.legendre.legvander(ix, self.deg)/self.basis_norms for ix in x])
        if self.basis_degree is None:
            super().get_basis()
        for i, ibasis_degree in enumerate(self.basis_degree):
            for idim, ibasis in enumerate(ibasis_degree):
                vander[:,i] = vander[:,i] * vander_ind[idim,:,ibasis]
        return vander

