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
    """ Experimental Design with Quadrature forms """

    def __init__(self, forms, p, ndim, dist_params=None, *args, **kwargs):
        """
        Parameters:
            forms: str or list of str, polynomial basis form. hem: Hermtie, leg: Legendre
            p:      int, number of quadrature points in each dimension
            ndim:   int, dimension of variable
            dist_params: list: corresponding distribution parameters
        """
        super().__init__(*args, **kwargs)
        self.ndim   = ndim
        self.p      = p
        if self.ndim ==1 :
            assert isinstance(forms, str)
            self.forms       = forms
            self.dist_params = dist_params
            self.filename    = '_'.join(['DoE_Quad', self.forms.capitalize() + num2print(self.p)])
        else:
            if isinstance(forms, list):
                if len(forms) == 1:
                    self.forms = forms * self.ndim
                else:
                    assert len(forms) == self.ndim
            elif isinstance(forms, str):
                self.forms = [forms,] * self.ndim
            else:
                raise ValueError('QuadratureDesign.forms takes either str or list of str, but {} is given'.format(type(forms)))

            self.filename    = '_'.join(['DoE_Quad', self.forms[0].capitalize() + num2print(self.p)])

            if dist_params is None:
                self.dist_params = [None,] * self.ndim
            elif isinstance(dist_params, list) and isinstance(dist_params[0], list):
                assert len(dist_params) == self.ndim
                self.dist_params = dist_params
            elif isinstance(dist_params, list) and not isinstance(dist_params[0], list):
                self.dist_params = [dist_params,] * self.ndim
            else:
                raise ValueError('Wrong format given for dist_params')

        ## results
        self.w           = []  # Gaussian quadrature weights corresponding to each quadrature node 

    def __str__(self):
        return('Gauss Quadrature: {}, p={:d}, ndim={:d} '.format(self.forms, self.p, self.ndim))

    def samples(self):
        """
        Return:
            Experiment samples in space (samples for each orders)
        """
        coords, weights = [], [] 

        if self.ndim == 1:
            u, w = self._gen_quad_1d(self.forms, self.p, self.dist_params) 
            self.u = u.reshape(1, -1) 
            self.w = np.squeeze(w)
        else:
            for ipoly_form, idist_params in zip(self.forms, self.dist_params):
                iu, iw = self._gen_quad_1d(ipoly_form, self.p, idist_params) 
                coords.append(iu)
                weights.append(iw)
            self.u = np.array(list(itertools.product(*coords))).T
            self.w = np.prod(np.array(list(itertools.product(*weights))).T, axis=0).reshape(1,-1)


    def _gen_quad_1d(self, poly_type, p, dist_params=None):
        if poly_type.lower() in ['hem', 'hermite']:
            ## probabilists , chaospy orth_ttr generate probabilists orthogonal polynomial
            x, w = np.polynomial.hermite_e.hermegauss(p) 
            mu, sigma = dist_params if dist_params else (0,1)
            x = mu + sigma * x
            w = sigma * w
            # coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
            # weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
            # weights = np.prod(weights, axis=0)
        elif poly_type.lower() in ['leg', 'legendre']:
            x, w = np.polynomial.legendre.leggauss(p)
            a, b = dist_params if dist_params else (-1,1)
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
            raise NotImplementedError("Quadrature forms '{:s}' is not defined".format(forms))
        return x, w
