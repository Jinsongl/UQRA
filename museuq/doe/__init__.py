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
from museuq.doe.base import ExperimentalDesign
from museuq.doe.quadrature import QuadratureDesign
from museuq.doe.random_design import RandomDesign
from museuq.doe.optimal_design import OptimalDesign

__all__= (
        'ExperimentalDesign', 
        'QuadratureDesign',
        'RandomDesign',
        'OptimalDesign',
        )

