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
            dist_u: list of distributions have .ppf method 
        """
        u = np.array(u, copy=False, ndmin=2)
        if isinstance(dist_u, (list, tuple)):
            ## checking if the u distributions are from scipy.stats
            for idist in dist_u:
                assert isfromstats(idist)
            ## if a list is given but not enough distributions, appending with Uniform(0,1)
            for _ in range(len(dist_u), self.ndim):
                dist_u.append(stats.uniform(0,1))
        else:
            assert isfromstats(dist_u)
            dist_u = [dist_u,] * self.ndim
        # u_cdf = np.array([idist.cdf(iu) for iu, idist in zip(u, dist_u)])
        # if (abs(u_cdf) > 1).any():
            # print('Warning: map_domain found cdf values greater than 1\n {}'.format(u_cdf[abs(u_cdf)>1]))
            # u_cdf[u_cdf>1] = 1
            # u_cdf[u_cdf<-1] = -1

        return u, dist_u


