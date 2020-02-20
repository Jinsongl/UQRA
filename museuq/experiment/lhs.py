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
import scipy
import pyDOE2
from museuq.utilities.decorators import random_state
from museuq.experiment._experimentbase import ExperimentBase
from museuq.utilities.helpers import num2print

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class LatinHyperCube(ExperimentBase):
    """ Experimental Design with Lazin Hyper Cube"""

    def __init__(self, distributions, random_seed=None, **kwargs):
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
        super().__init__(distributions=distributions, random_seed=random_seed)
        self.criterion = kwargs.get('criterion', 'maximin')
        self.iterations= kwargs.get('iterations', 5)
        self.filename = '_'.join(['DoE', 'Lhs'])

    def __str__(self):
        dist_names = [idist.dist.name for idist in self.distributions]
        message = 'LHS Design with criterion: {:s}, distributions: {}'.format(self.criterion, dist_names)
        return message

    @random_state
    def samples(self, n_samples, theta=[0,1]):
        """
        LHS sampling from distributions 
        Arguments:
            n_samples: int, number of samples 
            theta: list of [loc, scale] parameters for distributions
            For those distributions not specified with (loc, scale), the default value (0,1) will be applied
        Return:
            Experiment samples of shape(ndim, n_samples)
        """

        super().samples(n_samples, theta)
        lhs_u   = pyDOE2.lhs(self.ndim, samples=self.n_samples, criterion=self.criterion, iterations=self.iterations)
        lhs_u   = lhs_u.reshape(self.ndim, n_samples)
        lhs_x   = np.array([idist.ppf(ilhs_u, loc=iloc, scale=iscale) for idist, ilhs_u, iloc, iscale in zip(self.distributions, lhs_u, self.loc, self.scale)])

        return lhs_u, lhs_x




