#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
from museuq.doe.base import ExperimentalDesign
from museuq.utilities.helpers import num2print
import itertools
import collections

ASKEY_WIENER = {
        'normal': 'hem', 'uniform': 'leg'
        }
class QuadratureDesign(ExperimentalDesign):
    """ Experimental Design with Quadrature basis_names """

    def __init__(self, p, *args, **kwargs):
        """
        Parameters:
            basis_names: str or list of str, polynomial basis form. hem: Hermtie, leg: Legendre
            p    : int, number of quadrature points in each dimension
        """
        super().__init__(*args, **kwargs)
        self.p          = p
        self.basis_names= ['None' if idist_names is None else ASKEY_WIENER[idist_names] for idist_names in self.dist_names]
        self.filename   = '_'.join(['DoE', 'Quad'+self.basis_names[0].capitalize() + num2print(self.p)])

        ## results
        self.w           = []  # Gaussian quadrature weights corresponding to each quadrature node 

    def __str__(self):
        return('Gauss Quadrature: {}, p={:d}, ndim={:d} '.format(self.basis_names, self.p, self.ndim))

    def samples(self):
        """
        Return:
            Experiment samples in space (samples for each orders)
        """
        coords, weights = [], [] 

        for ibasis_names, idist_theta in zip(self.basis_names, self.dist_theta):
            iu, iw = self._gen_quad_1d(ibasis_names, self.p, idist_theta) 
            coords.append(iu)
            weights.append(iw)
        u = np.array(list(itertools.product(*coords))).T
        u = u.reshape(self.ndim, -1)
        w = np.prod(np.array(list(itertools.product(*weights))).T, axis=0)
        w = np.squeeze(w)
        return u, w


    def _gen_quad_1d(self, basis_name, p, dist_theta=None):
        if basis_name.lower() in ['hem', 'hermite']:
            ## probabilists , chaospy orth_ttr generate probabilists orthogonal polynomial
            x, w = np.polynomial.hermite_e.hermegauss(p) 
            mu, sigma = dist_theta if dist_theta else (0,1)
            x = mu + sigma * x
            w = sigma * w
            # coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
            # weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
            # weights = np.prod(weights, axis=0)
        elif basis_name.lower() in ['leg', 'legendre']:
            x, w = np.polynomial.legendre.leggauss(p)
            a, b = (dist_theta[0], dist_theta[0] + dist_theta[1]) if dist_theta else (-1,1)
            x = (b-a)/2 * x + (a+b)/2
            w = (b-a)/2 * w

        # elif basis_name in ['jac', 'jacobi']:
            # coord1d, weight1d = sp.special.roots_jacobi(order,4,2) #  np.polynomial.laguerre.laggauss(order)
            # coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
            # weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
            # weights = np.prod(weights, axis=0)

        # elif basis_name in ['lag', 'laguerre']:
            # coord1d, weight1d = np.polynomial.laguerre.laggauss(order)
            # coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
            # weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
            # weights = np.prod(weights, axis=0)

        # elif basis_name in ['cheb', 'chebyshev']:
            # coord1d, weight1d = np.polynomial.chebyshev.chebgauss(order)
            # coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
            # weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
            # weights = np.prod(weights, axis=0)
        else:
            raise NotImplementedError("Quadrature basis_names '{:s}' is not defined".format(basis_names))
        return x, w
