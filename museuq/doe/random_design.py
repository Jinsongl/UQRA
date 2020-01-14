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

    def __init__(self, method, *args, **kwargs):
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
        self.method   = method 
        self.filename = '_'.join(['DoE', self.method.capitalize() + num2print(self.n_samples)])

    def __str__(self):
        message1 = '  > Random Design with method: {:s}\n'.format(self.method)
        message2 = '    {:<15s}: {:.0E}\n'.format('# samples', self.n_samples)
        message3 = '    {:<15s}: {}\n'.format('Distributions', self.dist_names)
        message4 = '    {:<15s}: {}'.format('Parameters', self.dist_theta)
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
            for idist_name, idist_theta in itertools.zip_longest(self.dist_names, self.dist_theta):
                idist_name = idist_name.lower()
                idist_name = 'norm' if idist_name == 'normal' else idist_name
                idist = getattr(scipy.stats.distributions, idist_name)
                if idist_theta is None:
                    u_samples.append(idist.rvs(size=self.n_samples))
                else:
                    u_samples.append(idist.rvs(*idist_theta, size=self.n_samples))
            return  np.array(u_samples) 
        elif self.method.upper() in ['HALTON', 'HAL', 'H']:
            raise NotImplementedError 

        elif self.method.upper() in ['SOBOL', 'SOB', 'S']:
            raise NotImplementedError 

        else:
            raise NotImplementedError 
