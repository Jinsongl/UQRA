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
import uqra
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
        ### check arguments
        u = np.array(u, copy=False, ndmin=2)
        assert (u.shape[0] == self.ndim), 'solver.map_domain expecting {:d} \
                random variables, but {:s} given'.format(self.ndim, u.shape[0])

        try:
            dist_x = self.distributions
        except:
            raise ValueError("{:s}: missing distributions for random variables".format(self.name))

        ### check distribution for u
        ### dist_u is a list/tuple of scipy.stats
        if isfromstats(dist_u):
            dist_u = [dist_u, ] * self.ndim
        elif isinstance(dist_u, (list, tuple)):
            assert len(dist_u) == self.ndim , '{:s} expecting {:d} scipy.dist but {:d} given'.format(
                    self.name, self.ndim, len(dist_u))
            ## checking if the u distributions are from scipy.stats
            for idist in dist_u:
                assert isfromstats(idist)
        else:
            raise ValueError(' {} not recognized for dist_u'.format(dist_u))

        x = []
        if isinstance(dist_x, uqra.EnvBase):
            x = dist_x.ppf(np.array([idist_u.cdf(iu) for iu, idist_u in zip(u, dist_u)])) 
        elif isinstance(dist_x, (tuple, list)):
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
        else:
            raise ValueError(' {} not recognized for dist_x'.format(dist_x))
        x = np.vstack(x)
        if np.isnan(x).any():
            print(u[np.isnan(x)])
            raise ValueError('nan in UQRA.solver.map_domain()')

        if np.isinf(x).any():
            print(u[np.isinf(x)])
            raise ValueError('nan in UQRA.solver.map_domain()')
        return x

