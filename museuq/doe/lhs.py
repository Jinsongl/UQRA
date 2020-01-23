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
from museuq.doe.base import ExperimentalDesign
from museuq.utilities.helpers import num2print

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class LatinHyperCube(ExperimentalDesign):
    """ Experimental Design with Latin Hyper Cube"""

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
        dist_names = [idist.name for idist in self.dist]
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

        if self.dist is None:
            raise ValueError('No distributions are specified')
        elif not isinstance(self.dist, list):
            self.dist = [self.dist,]

        self.n_samples = n_samples
        self.loc   = [0,] * self.ndim
        self.scale = [1,] * self.ndim
        ## possible theta input formats:
        ## 1. [0,1] ndim == 1
        ## 2. [0,1] ndim == n
        ## 3. [[0,1],] ndim == 1
        ## 4. [[0,1],] ndim == n
        ## 5. [[0,1],[0,1],...] ndim == n

        ## case 1,2 -> case 3,5
        if isinstance(theta, list) and np.ndim(theta[0]) == 0:
            theta = [theta,] * self.ndim
        # case :4
        for i, itheta in enumerate(theta):
            self.loc[i] = itheta[0]
            self.scale[i] = itheta[1]

        ## updating filename 
        self.filename = self.filename+num2print(n_samples)

        lhs_u   = pyDOE2.lhs(self.ndim, samples=n_samples, criterion=self.criterion, iterations=self.iterations)
        lhs_u   = lhs_u.reshape(self.ndim, n_samples)
        lhs_x   = np.array([idist.ppf(ilhs_u, loc=iloc, scale=iscale) for idist, ilhs_u, iloc, iscale in zip(self.dist, lhs_u, self.loc, self.scale)])

        return lhs_u, lhs_x




