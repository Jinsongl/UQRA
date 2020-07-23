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
from uqra.experiment._experimentbase import ExperimentBase
from uqra.experiment.random_design import RandomDesign
from uqra.experiment.optimal_design import OptimalDesign
from uqra.experiment.quadrature import QuadratureDesign

__all__= (
        'ExperimentalDesign', 
        'QuadratureDesign',
        'RandomDesign',
        'OptimalDesign',
        )

