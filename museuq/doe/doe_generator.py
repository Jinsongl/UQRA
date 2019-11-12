#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""


"""
import itertools
import chaospy as cp
import numpy as np
import numpy.polynomial as nppoly
import scipy as sp
import itertools
# import pyDOE2 as pydoe

# QUAD_SHORT_NAMES = {
# "c": "clenshaw_curtis",
# "e": "gauss_legendre",
# "p": "gauss_patterson",
# "z": "genz_keister",
# "g": "golub_welsch",
# "j": "leja",
# "h": "gauss_hermite",
# "lag": "gauss_laguerre",
# "cheb": "gauss_chebyshev",
# "hermite":"gauss_hermite",
# "legendre":"gauss_legendre",
# "jacobi":"gauss_jacobi",
# }

dic_chaospy_quad_name = {
    "cc": "c", "leg"  : "e" , "pat" : "p",
    "gk": "z", "gwel" : "g" , "leja": "j",
    "hem": "h"   
}

def samplegen(doe_method, order, domain, rule=None, antithetic=None,
        verbose=False):
    """
    Design of experiment samples generator
    
    Arguments:
        doe_method: 
            "GQ"    : "Quadrature"  , "QUAD"  : "Quadrature",
            "MC"    : "Monte Carlo" , "FIX"   : "Fixed point"
        doe_order :  
            Quadrature: number of quadrature points
            MC: number of sample points
            FIX: not needed
        domain: 
            Quadrature: distributions
            MC: distributions to draw samples from
        rule:
            Quadrature: rules used to get quadrature points
            MC: sampling method
            FIX:
        antithetic:

        verbose:
    Returns:
        doe samples, array of shape:
            Quadrature: res.shape = (2,) 
                res[0]: coord of shape (ndim, nsamples) 
                res[1]: weights of shape (nsamples,) 
            MC: res.shape = (ndim, nsamples)
            
    """
    # DOE_METHOD_NAMES = {
        # "GQ"    : "Quadrature",
        # "QUAD"  : "Quadrature",
        # "MC"    : "Monte Carlo",
        # "FIX"   : "Fixed point"
        # } 

    # QUAD_SHORT_NAMES = {
        # "c": "clenshaw_curtis",
        # "e": "gauss_legendre",
        # "p": "gauss_patterson",
        # "z": "genz_keister",
        # "g": "golub_welsch",
        # "j": "leja",
        # "h": "gauss_hermite",
        # "lag": "gauss_laguerre",
        # "cheb": "gauss_chebyshev",
        # "hermite":"gauss_hermite",
        # "legendre":"gauss_legendre",
        # "jacobi":"gauss_jacobi",
        # }
    chaospy_quad= ['cc','leg','pat','gk','gwel','leja']
    numpy_quad  = ['hem', 'hermite','lgd', 'laguerre','lag','chebyshev','cheb', 'jac', 'jacobi']
    doe_method  = doe_method.upper()

    if doe_method in ['QUADRATURE','QUAD', 'GQ']:
        ## Return samples in 
        rule = 'hem' if rule is None else rule ## Default gauss_legendre
        if rule in chaospy_quad:
            doe_samples = _gen_quad_chaospy(order, domain, dic_chaospy_quad_name.get(rule))
        elif rule in numpy_quad:
            doe_samples = _gen_quad_numpy(order, domain, rule)
        else:
            raise NotImplementedError("Quadrature rule '{:s}' not defined".format(rule))
        quad_order = [order,] * len(domain) if np.isscalar(order) else order
        # print(u'   * Quadrature points complete  : {}'.format(quad_order))
        # print(udoe_samples[0].shape)
        # print(u'------------------------------------------------------------')
    elif doe_method in ['MONTE CARLO', 'MC']:
        """
        Monte Carlo Sampling 
        """
        rule = 'R' if rule is None else rule
        # print(u'************************************************************')
        # print(u'Design of experiment with Monte Carlo method')
        # print(u'Rule : {:s}, Number of monte carlo points (1d): {:d}'.format(rule, order))
        # print(u"Generating Monte Carlo samples...")
        
        doe_samples = domain.sample(order,rule=rule)
        doe_samples = doe_samples.reshape(len(domain), -1)
        # doe_samples = doe_samples.reshape(len(domain),-1)
        
    # elif doe_method == 'FIXED POINT':
        # """
        # Fixed points in doe_order will be used
        # """
        # # print(u'************************************************************')
        # # print(u'Design of experiment with Fixed points given in doe_order')
        # doe_samples = np.array(rule).reshape(len(domain), -1)
        # print(u'DOE samples shape:{}'.format(doe_samples.shape))
        
        # print(u'Design of experiment done with Fixed points')
        # print(u'------------------------------------------------------------')
    else:
        raise NotImplementedError("DOE method '{:s}' not defined".format(doe_method))
    return doe_samples 

def _gen_quad_chaospy(order, domain, rule):
    rule = 'e' if rule == 'legendre' else rule
    coord, weights= cp.generate_quadrature(order-1, domain, rule=rule) 
    doe_samples = np.concatenate((coord, weights[np.newaxis,:]), axis=0)
    return doe_samples

def _gen_quad_numpy(order, domain, rule):
    if rule in ['hem', 'hermite']:
        coord1d, weight1d = np.polynomial.hermite_e.hermegauss(order) ## probabilists , chaospy orth_ttr generate probabilists orthogonal polynomial
        # coord1d, weight1d = np.polynomial.hermite.hermgauss(order) ## physicist
        coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
        weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
        weights = np.prod(weights, axis=0)
    elif rule in ['jac', 'jacobi']:
        coord1d, weight1d = sp.special.roots_jacobi(order,4,2) #  np.polynomial.laguerre.laggauss(order)
        coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
        weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
        weights = np.prod(weights, axis=0)

        # raise NotImplementedError("Quadrature method '{:s}' based on numpy hasn't been implemented yet".format(rule))
    elif rule in ['lag', 'laguerre']:
        coord1d, weight1d = np.polynomial.laguerre.laggauss(order)
        coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
        weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
        weights = np.prod(weights, axis=0)

    elif rule in ['cheb', 'chebyshev']:
        coord1d, weight1d = np.polynomial.chebyshev.chebgauss(order)
        coord   = np.array(list(itertools.product(*[coord1d]*len(domain)))).T
        weights = np.array(list(itertools.product(*[weight1d]*len(domain)))).T
        weights = np.prod(weights, axis=0)
    else:
        raise NotImplementedError("Quadrature rule '{:s}' is not defined".format(rule))
    doe_samples = np.concatenate((coord, weights[np.newaxis,:]), axis=0)
    return doe_samples
