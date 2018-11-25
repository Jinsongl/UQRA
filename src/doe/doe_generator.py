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

def samplegen(doe_method, order, domain, rule=None, antithetic=None,
        verbose=False):
    """
    Design of experiment samples generator
    
    Arguments:

    Interpretation of the doe_method argument:
    

    +------------+------------------------------------------------------------+
    | Value | Interpretation                                             |
    +============+============================================================+
    | "A"   | Mapped to distribution domain using inverse Rosenblatt.    |
    +------------+------------------------------------------------------------+
    | "C"   | No mapping, but sets the number of dimension.              |
    +------------+------------------------------------------------------------+
    | "D"   | Stretch samples such that they are in domain[0], domain[1] |
    +------------+------------------------------------------------------------+
    | "E"   | Stretch samples such that they are in domain[0], domain[1] |
    +------------+------------------------------------------------------------+
    | "G"   | Stretch samples such that they are in domain[0], domain[1] |
    +------------+------------------------------------------------------------+
    | "S"   | Stretch samples such that they are in domain[0], domain[1] |
    +------------+------------------------------------------------------------+
    | "BA"  | Stretch samples such that they are in domain[0], domain[1] |
    +------------+------------------------------------------------------------+
    | "BD"  | Stretch samples such that they are in domain[0], domain[1] |
    +------------+------------------------------------------------------------+
    | "BD"  | Stretch samples such that they are in domain[0], domain[1] |
    +------------+------------------------------------------------------------+
    | "GQ"  | Gaussian-Quadrature rule                                      |
    +------------+------------------------------------------------------------+
    | "Q"   | Quantile rule                                                 |
    +------------+------------------------------------------------------------+
    | "MC"  | Monte carlo method to get random samples                      |
    +------------+------------------------------------------------------------+




    Interpretation of the domain argument:

    +------------+------------------------------------------------------------+
    | Value      | Interpretation                                             |
    +============+============================================================+
    | Dist       | Mapped to distribution domain using inverse Rosenblatt.    |
    +------------+------------------------------------------------------------+
    | int        | No mapping, but sets the number of dimension.              |
    +------------+------------------------------------------------------------+
    | array_like | Stretch samples such that they are in domain[0], domain[1] |
    +------------+------------------------------------------------------------+

    Intepretation of the rule argument:

    +------+---------------------+--------+
    | Key  | Name                | Nested |
    +======+=====================+========+
    | "C"  | Chebyshev nodes     | no     |
    +------+---------------------+--------+
    | "NC" | Nested Chebyshev    | yes    |
    +------+---------------------+--------+
    | "K"  | Korobov             | no     |
    +------+---------------------+--------+
    | "R"  | (Pseudo-)Random     | no     |
    +------+---------------------+--------+
    | "RG" | Regular grid        | no     |
    +------+---------------------+--------+
    | "NG" | Nested grid         | yes    |
    +------+---------------------+--------+


    """
    doe_method = doe_method.upper()
    
    QUAD_SHORT_NAMES = {
        "c": "clenshaw_curtis",
        "e": "gauss_legendre",
        "p": "gauss_patterson",
        "z": "genz_keister",
        "g": "golub_welsch",
        "j": "leja",
        "h": "gauss_hermite",
        "lag": "gauss_laguerre",
        "cheb": "gauss_chebyshev",
        "hermite":"gauss_hermite",
        "legendre":"gauss_legendre",
        "jacobi":"gauss_jacobi",
        }
    chaospy_quad = ['c','e','p','z','g','j']
    numpy_quad = ['h', 'hermite', 'legendre','lgd', 'laguerre','lag','chebyshev','cheb', 'jac', 'jacobi']
    if doe_method == 'GQ' or doe_method == 'QUAD':
        ## Return samples in 
        rule = 'h' if rule is None else rule ## Default gauss_legendre
        print('************************************************************')
        print('Design of experiment with Quadrature method')
        quad_order = [order,] * domain.length if np.isscalar(order) else order
        print('Quadrature rule: {:s}, Number of quadrature points: {}'.format(QUAD_SHORT_NAMES[rule], quad_order))
        if rule in chaospy_quad:
            doe_samples = _gen_quad_chaospy(order, domain, rule)
        elif rule in numpy_quad:
            doe_samples = _gen_quad_numpy(order, domain, rule)
        else:
            raise NotImplementedError("Quadrature rule '{:s}' not defined".format(rule))
        # print(doe_samples[0].shape)
        print('>>> Done...')
        print('------------------------------------------------------------')
    elif doe_method == 'MC':
        """
        Monte Carlo Points are generated one by one by design, avoiding possible large memory requirement ???
        """
        rule = 'R' if rule is None else rule
        print('************************************************************')
        print('Design of experiment with Monte Carlo method')
        print('Rule : {:s}, Number of monte carlo points (1d): {:d}'.format(rule, order))
        # print("Generating Monte Carlo samples...")
        
        doe_samples = domain.sample(order,rule=rule)
        doe_samples = doe_samples.reshape(domain.length,-1)
        print('Design of experiment done with {:d} Monte Carlo points'.format(order))
        print('------------------------------------------------------------')
        
    elif doe_method == 'FIX':
        """
        Fixed points in doe_order will be used
        """
        print('************************************************************')
        print('Design of experiment with Fixed points given in doe_order')
        doe_samples = np.array(rule).reshape(domain.length, -1)
        print('DOE samples shape:{}'.format(doe_samples.shape))
        
        print('Design of experiment done with Fixed points')
        print('------------------------------------------------------------')
    else:
        raise NotImplementedError("DOE method '{:s}' not defined".format(doe_method))
    return doe_samples 

def _gen_quad_chaospy(order, domain, rule):
    coord, weights= cp.generate_quadrature(order-1, domain, rule=rule) 
    doe_samples = np.array([coord,weights])
    return doe_samples

def _gen_quad_numpy(order, domain, rule):
    if rule in ['h', 'hermite']:
        coord1d, weight1d = np.polynomial.hermite_e.hermegauss(order) ## probabilists
        # coord1d, weight1d = np.polynomial.hermite.hermgauss(order) ## physicist
        coord   = np.array(list(itertools.product(*[coord1d]*domain.length))).T
        weights = np.array(list(itertools.product(*[weight1d]*domain.length))).T
        weights = np.prod(weights, axis=0)
    elif rule in ['jac', 'jacobi']:
        coord1d, weight1d = sp.special.roots_jacobi(order,4,2) #  np.polynomial.laguerre.laggauss(order)
        coord   = np.array(list(itertools.product(*[coord1d]*domain.length))).T
        weights = np.array(list(itertools.product(*[weight1d]*domain.length))).T
        weights = np.prod(weights, axis=0)

        # raise NotImplementedError("Quadrature method '{:s}' based on numpy hasn't been implemented yet".format(rule))
    elif rule in ['lag', 'laguerre']:
        coord1d, weight1d = np.polynomial.laguerre.laggauss(order)
        coord   = np.array(list(itertools.product(*[coord1d]*domain.length))).T
        weights = np.array(list(itertools.product(*[weight1d]*domain.length))).T
        weights = np.prod(weights, axis=0)

    elif rule in ['cheb', 'chebyshev']:
        coord1d, weight1d = np.polynomial.chebyshev.chebgauss(order)
        coord   = np.array(list(itertools.product(*[coord1d]*domain.length))).T
        weights = np.array(list(itertools.product(*[weight1d]*domain.length))).T
        weights = np.prod(weights, axis=0)
    else:
        raise NotImplementedError("Quadrature method '{:s}' is not defined".format(rule))
    doe_samples = np.array([coord, weights])
    return doe_samples
