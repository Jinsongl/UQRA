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
import numpy as np
import itertools

class RandomDesign(ExperimentalDesign):
    """ Experimental Design with random sampling methods"""

    def __init__(self, method, n, dist_names, params=None, *args, **kwargs):
        """
        "Random"/Quasi random sampling design:
        Arguments:
        method:
            "R": pseudo random sampling, brute force Monte Carlo 
            "Halton": Halton quasi-Monte Carlo
            'Sobol': Sobol sequence quasi-Monte Carlo
        n: int, number of samples 
        dist_names: str or list of str
        params: list of params for each dist
        """
        super().__init__(*args, **kwargs)
        self.method     = method 
        self.n_samples  = int(n)
        self.dist_names = dist_names if isinstance(dist_names, list) else [dist_names,]
        self.dist_params= [params, ] if params is None else params

    def __str__(self):
        return('Sampling method: {:<15s}, num. samples: {:d} '.format(self.method, self.n_samples))

    @random_state
    def samples(self):
        """
        Return:
            Experiment samples of shape(ndim, n_samples)
        """
        if self.method.upper() in ['R', 'MC', 'MCS']:
            x_samples = []
            for idist_name, idist_params in itertools.zip_longest(self.dist_names, self.dist_params):
                dist_random = getattr(np.random, idist_name)
                if idist_params is None:
                    x_samples.append(dist_random(size=self.n_samples))
                else:
                    x_samples.append(dist_random(*idist_params, size=self.n_samples))
            self.x = np.array(x_samples) 
        elif self.method.upper() in ['HALTON', 'HAL', 'H']:
            raise NotImplementedError 

        elif self.method.upper() in ['SOBOL', 'SOB', 'S']:
            raise NotImplementedError 

        else:
            raise NotImplementedError 
