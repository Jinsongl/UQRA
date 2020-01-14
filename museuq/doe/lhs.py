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
import itertools
from museuq.utilities.decorators import random_state
from museuq.doe.base import ExperimentalDesign
from museuq.utilities.helpers import num2print

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class LatinHyperCube(ExperimentalDesign):
    """ Experimental Design with Latin Hyper Cube"""

    def __init__(self, *args, **kwargs):
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
        super().__init__(*args, **kwargs)
        self.criterion = kwargs.get('criterion', 'maximin')
        self.iterations= kwargs.get('iterations', 5)
        self.filename = '_'.join(['DoE', 'Lhs' + num2print(self.n_samples)])

    def __str__(self):
        return('Sampling method: {:<15s}, num. samples: {:ndim} '.format('LHS', self.n_samples))

    @random_state
    def samples(self):
        lhs_u   = pyDOE2.lhs(self.ndim, samples=self.n_samples, criterion=self.criterion, iterations=self.iterations)
        lhs_u   = lhs_u.reshape(self.ndim, self.n_samples)
        lhs_x   = []
        for ilhd_u, idist_name, idist_theta in zip(lhs_u, self.dist_names, self.dist_theta):
            idist_name = idist_name.lower()
            idist_name = 'norm' if idist_name == 'normal' else idist_name
            idist = getattr(scipy.stats.distributions, idist_name)
            if idist_theta is None:
                lhs_x.append(idist.ppf(ilhd_u))
            else:
                # print(idist_theta)
                lhs_x.append(idist.ppf(ilhd_u, *idist_theta))
        lhs_x = np.array(lhs_x)
        return lhs_u, lhs_x




