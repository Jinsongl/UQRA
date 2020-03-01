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
from ..utilities.helpers import isfromstats

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
            u: ndarray of shape(ndim, nsamples)
            dist_u: list of distributions from scipy.stats
        """
        u = np.array(u, copy=False, ndmin=2)
        if isinstance(dist_u, (list, tuple)):
            ## if a list is given but not enough distributions, appending with Uniform(0,1)
            for idist in dist_u:
                assert isfromstats(idist)
            for _ in range(len(dist_u), self.ndim):
                dist_u.append(stats.uniform(0,1))
        else:
            assert isfromstats(dist_u)
            dist_u = [dist_u,] * self.ndim

        return u, dist_u



