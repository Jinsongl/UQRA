#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
from museuq.doe.base import ExperimentalDesign
from museuq.utilities.decorators import random_state
from museuq.utilities.helpers import num2print
import numpy as np
import scipy
import itertools

class RandomDesign(ExperimentalDesign):
    """ Experimental Design with random sampling methods"""

    def __init__(self, distributions, method, random_seed=None):
        """
        "Random"/Quasi random sampling design, dist_names are independent 
        Arguments:
        method:
            "R": pseudo random sampling, brute force Monte Carlo 
            "Halton": Halton quasi-Monte Carlo
            'Sobol': Sobol sequence quasi-Monte Carlo
        n: int, number of samples 
        ndim: 
        dist_names: str or list of str
        params: list of params set for each dist, set is given in tuple
        """
        super().__init__(distributions=distributions, random_seed=random_seed)
        self.method   = method 
        self.filename = '_'.join(['DoE', self.method.capitalize() ])

    def __str__(self):
        dist_names = [idist.name for idist in self.dist]
        message = 'Random Design with method: {:s}, Distributions: {}'.format(self.method, dist_names)
        return message

    @random_state
    def samples(self, n_samples, theta=[0,1]):
        """
        Random sampling from distributions with specified method
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

        ## possible theta input formats:
        ## 1. [0,1] ndim == 1
        ## 2. [0,1] ndim == n
        ## 3. [[0,1],] ndim == 1
        ## 4. [[0,1],] ndim == n
        ## 5. [[0,1],[0,1],...] ndim == n
        self.n_samples = n_samples
        self.loc   = [0,] * self.ndim
        self.scale = [1,] * self.ndim
        ## case 1,2 -> case 3,5
        if isinstance(theta, list) and np.ndim(theta[0]) == 0:
            theta = [theta,] * self.ndim
        # case :4
        for i, itheta in enumerate(theta):
            self.loc[i] = itheta[0]
            self.scale[i] = itheta[1]


        ## updating filename 
        self.filename = self.filename+num2print(n_samples)

        if self.method.upper() in ['R', 'MC', 'MCS']:
            u_samples = [idist.rvs(size=n_samples, loc=iloc, scale=iscale) for idist, iloc, iscale in zip(self.dist, self.loc, self.scale)]
            return  np.array(u_samples) 
            
        elif self.method.upper() in ['HALTON', 'HAL', 'H']:
            raise NotImplementedError 

        elif self.method.upper() in ['SOBOL', 'SOB', 'S']:
            raise NotImplementedError 

        else:
            raise NotImplementedError 
