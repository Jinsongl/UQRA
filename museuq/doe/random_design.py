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
import itertools

class RandomDesign(ExperimentalDesign):
    """ Experimental Design with random sampling methods"""

    def __init__(self, method, n, dist_names, params=None, ndim=1, *args, **kwargs):
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
        super().__init__(*args, **kwargs)
        self.method     = method 
        self.n_samples  = int(n)
        self.ndim       = ndim
        if self.ndim == 1:
            if isinstance(dist_names, str):
                self.dist_names = [dist_names, ]
            elif isinstance(dist_names, list):
                assert len(dist_names) ==1
                self.dist_names = dist_names
            else:
                raise ValueError('Wrong dist_names format is provided')
        else:
            if isinstance(dist_names, str):
                self.dist_names = [dist_names,] * self.ndim
            elif isinstance(dist_names, list):
                if len(dist_names) == self.ndim:
                    self.dist_names = dist_names
                elif len(dist_names) == 1:
                    self.dist_names = dist_names * self.ndim
                else:
                    raise ValueError('Expecting {:d} random variables, but {:d} are given'.format(self.ndim, len(dist_names)))
            else:
                raise ValueError('Wrong dist_names format is provided')

        if params is None:
            self.dist_params = [None, ] * self.ndim
        elif isinstance(params, list):
            ### when only one set is given, assume to be for the first distribution assigned. Others are assumed to be None
            for _ in range(len(params), self.ndim):
                params.append(None) 
            self.dist_params = params
        elif isinstance(params, tuple):
            self.dist_params = [None, ] * self.ndim
            self.dist_params[0] = params

        self.filename    = '_'.join(['DoE', self.method.capitalize() + num2print(self.n_samples)])


    def __str__(self):
        message1 = '  > Random Design with method: {:s}\n'.format(self.method)
        message2 = '    {:<15s}: {:.0E}\n'.format('# samples', self.n_samples)
        message3 = '    {:<15s}: {}\n'.format('Distributions', self.dist_names)
        message4 = '    {:<15s}: {}'.format('Parameters', self.dist_params)
        message  = message1 + message2 + message3 + message4
        return message

    @random_state
    def samples(self):
        """
        Return:
            Experiment samples of shape(ndim, n_samples)
        """
        if self.method.upper() in ['R', 'MC', 'MCS']:
            u_samples = []
            for idist_name, idist_params in itertools.zip_longest(self.dist_names, self.dist_params):
                dist_random = getattr(np.random, idist_name)
                if idist_params is None:
                    u_samples.append(dist_random(size=self.n_samples))
                else:
                    u_samples.append(dist_random(*idist_params, size=self.n_samples))
            self.u = np.array(u_samples) 
        elif self.method.upper() in ['HALTON', 'HAL', 'H']:
            raise NotImplementedError 

        elif self.method.upper() in ['SOBOL', 'SOB', 'S']:
            raise NotImplementedError 

        else:
            raise NotImplementedError 
