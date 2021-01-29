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

    def map_domain(self, u, dist_u=stats.uniform(0,1)):
        """
        mapping random variables u from its dist_u (default U(0,1)) to self.distributions 
        Argument:
            u: ndarray of shape(ndim, nsamples)
            dist_u: list of distributions have .ppf method 
        """
        u = np.array(u, copy=False, ndmin=2)
        assert (u.shape[0] == self.ndim), 'error: solver.map_domain expecting {:d} \
                random variables, {:s} given'.format(self.ndim, u.shape[0])

        assert hasattr(self, 'distributions'), 'No distributions attribute'
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

        x = []
        for iu, idist_x, idist_u in zip(u, self.distributions, dist_u):
            if idist_x.dist.name == 'uniform' and idist_u.dist.name == 'uniform':
                ua, ub = idist_u.support()
                loc_u, scl_u = ua, ub-ua
                xa, xb = idist_x.support()
                loc_x, scl_x = xa, xb-xa 
                x.append((iu-loc_u)/scl_u * scl_x + loc_x)

            elif idist_x.dist.name == 'norm' and idist_u.dist.name == 'norm':
                mean_u = idist_u.mean()
                mean_x = idist_x.mean()
                std_u  = idist_u.std()
                std_x  = idist_x.std()
                x.append((iu-mean_u)/std_u * std_x + mean_x)
            else:
                x.append(idist_x.ppf(idist_u.cdf(iu)))
        x = np.vstack(x)
        return x

