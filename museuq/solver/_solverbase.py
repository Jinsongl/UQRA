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
import scipy.stats as stats

class SolverBase(object):
    """
    Abstract class for solvers
    """

    def __init__(self):
        self.name = ''
        self.ndim = None
    def run(self, x):
        """
        run solver with input variables
        Parameters:
            x: np.ndarray input data of shape(ndim, nsamples) 
              or str for input filename
        Returns:
            No returns
        """
        raise NotImplementedError


    def map_domain(self, u, dist_u):
        """
        mapping random variables u from distribution dist_u (default U(0,1)) to self.distributions 
        Argument:
            two options:
            1. cdf(u)
            2. u and dist_u
        """

        raise NotImplementedError



