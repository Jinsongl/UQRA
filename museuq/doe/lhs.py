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

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class LatinHyperCube(ExperimentalDesign):
    """ Experimental Design with Latin Hyper Cube"""

    def __init__(self, d, n, *args, **kwargs):
        """
        d: dimension of random variables

        Optional: 
            n: number of samples to generate at each dimension, default: d
            criterion: a string that tells lhs how to sample the points
               - default: “maximin” or “m”: maximize the minimum distance between points, but place the point in a randomized location within its interval
               -  None, simply randomizes the points within the intervals
               - 'center' or “c”: center the points within the sampling intervals
               - 
               - “centermaximin” or “cm”: same as “maximin”, but centered within the intervals
               - “correlation” or “corr”: minimize the maximum correlation coefficient
        """
        super().__init__(*args, **kwargs)

        self.d         = d
        self.n_samples = int(n)
        self.criterion = kwargs.get('criterion', 'maximin')
        self.iterations= kwargs.get('iterations', 5)

    def __str__(self):
        return('Sampling method: {:<15s}, num. samples: {:d} '.format('LHS', self.n_samples))

    @random_state
    def samples(self, dist_names='uniform', dist_params=[[0,1]]):
        # with warnings.catch_warnings():
            # warnings.simplefilter('ignore', RuntimeWarning)
            # lhs_u = pyDOE2.lhs(self.d, samples=self.n_samples, criterion=self.criterion, iterations=self.iterations)
        lhs_u = pyDOE2.lhs(self.d, samples=self.n_samples, criterion=self.criterion, iterations=self.iterations)
        lhs_u = lhs_u.reshape(self.d, self.n_samples)
        # print(lhs_u)
        dist_names = [dist_names, ] * self.d if isinstance(dist_names, str) else dist_names
        assert len(dist_names) == self.d
        lhs_x = []
        for ilhd_u, idist_name, idist_params in itertools.zip_longest(lhs_u, dist_names, dist_params):
            idist = getattr(scipy.stats.distributions, idist_name)
            print(idist_name)
            print(idist_params)
            if idist_params is None:
                lhs_x.append(idist.ppf(ilhd_u))
            else:
                lhs_x.append(idist(*idist_params).ppf(ilhd_u))
        self.x = np.array(lhs_x)



