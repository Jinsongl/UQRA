#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import inspect
import numpy as np
import scipy
import pyDOE2
from uqra.experiment._experimentbase import ExperimentBase
from uqra.utilities.helpers import num2print

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class LatinHyperCube(ExperimentBase):
    """ Experimental Design with Lazin Hyper Cube"""

    def __init__(self, distributions, **kwargs):
        """
        ndim: dimension of random variables

        Optional: 
            n: number of samples to generate at each dimension, default: ndim
            criterion: a string that tells lhs how to sample the points
               - default: “maximin” or “m”: maximize the minimum distance between points, but place the point in a randomized location within its interval
               -  None, simply randomizes the points within the intervals
               - 'center' or “c”: center the points within the sampling intervals
               - 
               - “centermaximin” or “cm”: same as “maximin”, but centered within the intervals
               - “correlation” or “corr”: minimize the maximum correlation coefficient
        """
        self.ndim, self.distributions = super()._set_distributions(distributions)
        self.criterion = kwargs.get('criterion', 'maximin')
        self.iterations= kwargs.get('iterations', 5)
        self.filename = '_'.join(['DoE', 'Lhs'])

    def __str__(self):
        if self.distributions is None:
            message = 'Random samples, no distribution has been set yet }'

        else:
            dist_names = []
            for idist in self.distributions:
                try:
                    dist_names.append(idist.name)
                except:
                    dist_names.append(idist.dist.name)
            message = 'Random samples from: {}'.format(dist_names)

        return message

    def samples(self, size=1,loc=0, scale=1,  random_state=None):
        """
        LHS sampling from distributions 
        Arguments:
            n_samples: int, number of samples 
            theta: list of [loc, scale] parameters for distributions
            For those distributions not specified with (loc, scale), the default value (0,1) will be applied
        Return:
            Experiment samples of shape(ndim, n_samples)
        """

        size = super()._check_int(size)
        locs, scales = super()._set_parameters(loc, scale)

        lhs_u = []
        for isize in size:
            u = pyDOE2.lhs(self.ndim, samples=isize, 
                    criterion=self.criterion, iterations=self.iterations,random_state=random_state).T
            lhs_u.append(u)
        lhs_u = np.squeeze(lhs_u)
        lhs_u = np.array([idist.ppf(iu) for iu, idist in zip(lhs_u, self.distributions)])
        lhs_u = np.array([iu * iscale + iloc for iu, iloc, iscale in zip(lhs_u, locs, scales)])
        return lhs_u




