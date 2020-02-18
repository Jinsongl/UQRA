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
from museuq.experiment._experimentbase import ExperimentBase
from museuq.utilities.helpers import num2print
import itertools
import collections
ASKEY_WIENER = {
        'normal': 'Hem',
        'gaussian': 'Hem',
        'norm'  : 'Hem',
        'gamma' : 'Lag',
        'beta'  : 'Jac',
        'uniform':'Lag',
        }

class QuadratureDesign(ExperimentBase):
    """ Experimental Design with Quadrature basis_names """

    def __init__(self, distributions):
        """
        Parameters:
            basis_names: str or list of str, polynomial basis form. hem: Hermtie, leg: Legendre
            p    : int, number of quadrature points in each dimension
        """
        super().__init__(distributions=distributions)
        dist_names    = list(set([ASKEY_WIENER[idist.name] for idist in self.dist]))
        self.filename = '_'.join(['DoE', 'Quad'+ ''.join(dist_names)])
        self.w        = []  # Gaussian quadrature weights corresponding to each quadrature node 

        # self.basis_names= ['None' if idist_names is None else ASKEY_WIENER[idist_names] for idist_names in self.dist_names]
    def __str__(self):
        dist_names    = [idist.name for idist in self.dist]
        return('Gauss Quadrature: {}'.format(dist_names))

    def samples(self, p, theta=[0,1]):
        """
        Sampling p Gauss-Quadrature pints from distributions 
        Arguments:
            n_samples: int, number of samples 
            theta: list of [loc, scale] parameters for distributions
            For those distributions not specified with (loc, scale), the default value (0,1) will be applied
        Return:
            Experiment samples of shape(ndim, n_samples)
        Return:
            Experiment samples in space (samples for each orders)
        """
        if self.dist is None:
            raise ValueError('No distributions are specified')
        elif not isinstance(self.dist, list):
            self.dist = [self.dist,]
        ## possible theta input formats:
        ## 1. [0,1] ndim == 1
        ## 2. [0,1] ndim == n
        ## 3. [[0,1],] ndim == 1
        ## 4. [[0,1],] ndim == n
        ## 5. [[0,1],[0,1],...] ndim == n

        loc   = [0,] * self.ndim
        scale = [1,] * self.ndim
        ## case 1,2 -> case 3,5
        if isinstance(theta, list) and np.ndim(theta[0]) == 0:
            theta = [theta,] * self.ndim
        # case :4
        for i, itheta in enumerate(theta):
            loc[i] = itheta[0]
            scale[i] = itheta[1]


        self.p      = p
        self.loc    = loc
        self.scale  = scale
        coords, weights = [], [] 
        ## updating filename 
        self.filename = self.filename+num2print(self.p)

        for idist, iloc, iscale in zip(self.dist, self.loc, self.scale):
            iu, iw = self._gen_quad_1d(idist, self.p, loc=iloc, scale=iscale) 
            coords.append(iu)
            weights.append(iw)
        u = np.array(list(itertools.product(*coords))).T
        u = u.reshape(self.ndim, -1)
        w = np.prod(np.array(list(itertools.product(*weights))).T, axis=0)
        w = np.squeeze(w)
        return u, w


    def _gen_quad_1d(self, dist, p, loc, scale):
        if dist.name == 'norm':
            ## probabilists , chaospy orth_ttr generate probabilists orthogonal polynomial
            x, w = np.polynomial.hermite_e.hermegauss(p) 
            x = loc + scale* x
            w = scale * w


            # x, w = np.polynomial.hermite_e.hermegauss(p) 
            # mu, sigma = dist_theta if dist_theta else (0,1)
            # x = mu + sigma * x
            # w = sigma * w
            # coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
            # weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
            # weights = np.prod(weights, axis=0)
        elif dist.name == 'uniform':
            x, w = np.polynomial.legendre.leggauss(p)
            x = scale/2 * x + (scale+2*loc)/2
            w = scale/2 * w

            # a, b = (dist_theta[0], dist_theta[0] + dist_theta[1]) if dist_theta else (-1,1)
            # x = (b-a)/2 * x + (a+b)/2
            # w = (b-a)/2 * w

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
