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
import itertools
DOE_METHOD_FULL_NAMES = {
    "GQ"    : "Quadrature"  , "QUAD"  : "Quadrature",
    "MC"    : "Monte Carlo" , "FIX"   : "Fixed point"
    } 

DOE_RULE_FULL_NAMES = {
    "CC": "clenshaw_curtis"  , "LEG"   : "gauss_legendre"  , "PAT"   : "gauss_patterson",
    "GK": "genz_keister"     , "GWEL"   : "golub_welsch"    , "LEJA"   : "leja",
    "HEM": "gauss_hermite"    ,"LAG"  : "gauss_laguerre"  , "CHEB": "gauss_chebyshev",
    "HERMITE"   :"gauss_hermite",
    "LEGENDRE"  :"gauss_legendre",
    "JACOBI"    :"gauss_jacobi",
    "R": "Pseudo-Random", "RG": "Regular Grid", "NG": "Nested Grid", "L": "Latin Hypercube",
    "S": "Sobol", "H":"Halton", "M": "Hammersley",
    "FIX": "Fixed point"
    }
class QuadratureDesign(ExperimentalDesign):
    """ Experimental Design with Quadrature poly_types """

    def __init__(self, types, p, params=None, *args, **kwargs):
        """
        Space: 
            1. cp.distributions
        """
        super().__init__(*args, **kwargs)
        self.poly_types  = [types, ] if isinstance(types, str) else types ### str or list of str
        self.poly_orders = [p,] if isinstance(p, int) else p ### int or list of int 
        self.poly_orders = self.poly_orders * len(self.poly_types) if len(self.poly_orders) == 1 else self.poly_orders
        self.poly_params = [None,] if params is None else params
        self.poly_params = self.poly_params * len(self.poly_types) if len(self.poly_params) == 1 else self.poly_params 
        self.poly_names  = [DOE_RULE_FULL_NAMES[ipoly_type.upper()] for ipoly_type in self.poly_types]
        ## results
        self.w           = []  # Gaussian quadrature weights corresponding to each quadrature node 
        # self.filename    = 'DoE_Quad{}'.format(self.poly_types.capitalize())
        # self.filename_tags = []

        # for item, count in collections.Counter(self.orders).items():
            # if count == 1:
                # itag = [ num2print(item)]
            # else:
                # itag = [ num2print(item) + 'R{}'.format(i) for i in range(count)]
            # self.filename_tags += itag

    def __str__(self):
        return('Gauss Quadrature: {}, p-order: {} '.format(self.poly_names, self.poly_orders))

    def samples(self):
        """
        Return:
            Experiment samples in space (samples for each orders)
        """
        coords, weights = [], [] 

        for ipoly_type, ipoly_order, ipoly_params in zip(self.poly_types, self.poly_orders, self.poly_params):
            ix, iw = self._gen_quad_1d(ipoly_type, ipoly_order, ipoly_params) 
            coords.append(ix)
            weights.append(iw)
        self.x = np.array(list(itertools.product(*coords))).T
        self.w = np.prod(np.array(list(itertools.product(*weights))).T, axis=0)

    def _gen_quad_1d(self, poly_type, p, params=None):
        if poly_type in ['hem', 'hermite']:
            ## probabilists , chaospy orth_ttr generate probabilists orthogonal polynomial
            x, w = np.polynomial.hermite_e.hermegauss(p) 
            mu, sigma = params if params else (0,1)
            x = mu + sigma * x
            w = sigma * w
            # coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
            # weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
            # weights = np.prod(weights, axis=0)
        elif poly_type in ['leg', 'legendre']:
            x, w = np.polynomial.legendre.leggauss(p)
            a, b = params if params else (-1,1)
            x = (b-a)/2 * x + (a+b)/2
            w = (b-a)/2 * w

        # elif poly_type in ['jac', 'jacobi']:
            # coord1d, weight1d = sp.special.roots_jacobi(order,4,2) #  np.polynomial.laguerre.laggauss(order)
            # coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
            # weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
            # weights = np.prod(weights, axis=0)

        # elif poly_type in ['lag', 'laguerre']:
            # coord1d, weight1d = np.polynomial.laguerre.laggauss(order)
            # coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
            # weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
            # weights = np.prod(weights, axis=0)

        # elif poly_type in ['cheb', 'chebyshev']:
            # coord1d, weight1d = np.polynomial.chebyshev.chebgauss(order)
            # coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
            # weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
            # weights = np.prod(weights, axis=0)
        else:
            raise NotImplementedError("Quadrature poly_types '{:s}' is not defined".format(poly_types))
        return x, w
