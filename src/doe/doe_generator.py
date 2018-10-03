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
import os


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
    | "GQ"  | Gaussian-Quadrature rule |
    +------------+------------------------------------------------------------+
    | "Q"   | Quantile rule|
    +------------+------------------------------------------------------------+
    | "MC"  | Monte carlo method to get random samples|
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
    
    if doe_method == 'GQ':
        ## Return samples in 
        rule = 'e' if rule is None else rule ## Default 
        print('************************************************************')
        print('Design of experiment with Quadrature method')
        print('Rule : {:s}, Number of quadrature points (1d): {:d}}'.format(rule, order))
        coord, weights= cp.generate_quadrature(order, domain, rule=rule) 
        doe_samples = np.array([coord,weights])
        print('Design of experiment done with {:d} quadrature points'.format(len(weights)))
        print('------------------------------------------------------------')
    elif doe_method == 'MC':
        """
        Monte Carlo Points are generated one by one by design, avoiding possible large memory requirement ???
        """
        rule = 'R' if rule is None else rule
        print('************************************************************')
        print('Design of experiment with Monte Carlo method')
        print('Rule : {:s}, Number of monte carlo points (1d): {:d}}'.format(rule, order))
        # print("Generating Monte Carlo samples...")
        
        doe_samples = domain.sample(order,rule=rule)
        print('Design of experiment done with {:d} quadrature points'.format(order))
        print('------------------------------------------------------------')
        
    else:
        raise NotImplementedError("scheme not defined")
    return doe_samples 


