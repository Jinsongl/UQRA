#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
from museuq.experiment._experimentbase import ExperimentBase
from museuq.utilities.decorators import random_state
import numpy as np
import scipy
import itertools

class RandomDesign(ExperimentBase):
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
        params: list of params set for each distributions, set is given in tuple
        """
        super().__init__(distributions=distributions, random_seed=random_seed)
        self.method   = method 
        self.filename = '_'.join(['DoE', self.method.capitalize() ])

    def __str__(self):
        dist_names = [idist.dist.name for idist in self.distributions]
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

        super().samples(n_samples, theta)

        if self.method.upper() in ['R', 'MC', 'MCS']:
            u_samples = [idist.rvs(size=self.n_samples, loc=iloc, scale=iscale) for idist, iloc, iscale in zip(self.distributions, self.loc, self.scale)]
            return  np.array(u_samples) 
            
        elif self.method.upper() in ['HALTON', 'HAL', 'H']:
            raise NotImplementedError 

        elif self.method.upper() in ['SOBOL', 'SOB', 'S']:
            raise NotImplementedError 

        else:
            raise NotImplementedError 
