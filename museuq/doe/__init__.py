#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""
Experimental Design package

Reference:

    Shin, Yeonjong, and Dongbin Xiu. 
    "Nonadaptive quasi-optimal points selection for least squares linear regression." 
    SIAM Journal on Scientific Computing 38.1 (2016): A385-A411.
===============================
"""

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
# import doe_generator


