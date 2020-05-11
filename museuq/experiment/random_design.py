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
import numpy as np
import scipy
import itertools
import scipy.stats as stats

class RandomDesign(ExperimentBase):
    """ Experimental Design with random sampling methods"""

    def __init__(self, distributions, method):
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
        super().__init__(samplingfrom=distributions)
        self.method   = method 
        self.filename = '_'.join(['DoE', self.method.capitalize() ])

    def __str__(self):
        dist_names = [idist.dist.name for idist in self.distributions]
        message = 'Random Design with method: {:s}, Distributions: {}'.format(self.method, dist_names)
        return message

    def get_samples(self, n_samples, theta=[0,1]):
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
            u = np.array([idist.rvs(size=self.n_samples, loc=iloc, scale=iscale) for idist, iloc, iscale in zip(self.distributions, self.loc, self.scale)])
            return  u 
            
        elif self.method.upper() in ['CHRISTOFFEL', 'CLS']:
            """
            Sampling from the pluripoential equilibrium corresponding to distributions specified in self.distributions.
            
            Ref: Table 2 of "A Christoffel function weighted least squares algorithm for collocation approximations, Akil Harayan, John D. Jakeman, and Tao Zhou"
            Domain D    | Sampling density domain   |     Sampling density v(y) |
            [-1,1]^d    |   [1,1]^d                 |   1/(pi ^d \prod _{i=1} ^ d sqrt(1-x_i^2)) 


            """

            ## Assuming all distributions in self.distributions are same
            idist = self.distributions[0]
            try:
                dist_name = idist.name
            except AttributeError:
                dist_name = idist.dist.name

            if dist_name == 'uniform':
                u = stats.uniform.rvs(0,np.pi,size=(self.ndim, self.n_samples))
                x = np.cos(u)

            elif dist_name in ['norm', 'normal']:
                #### Sampling from Ball with radius sqrt(2). 
                ## 1. z ~ normal(0,1), z / norm(z)
                ## 2. u ~ uniform(0,1 )

                n = int(self.n_samples)
                z = stats.norm.rvs(0,1,size=(self.ndim, n))
                z = z/np.linalg.norm(z, axis=0)
                u = stats.beta.rvs(self.ndim/2.0,self.ndim/2.0 + 1,size=n)
                u = np.sqrt(2 * u) 
                x = z * u 
            else:
                raise NotImplementedError

            return  x

        elif self.method.upper() in ['HALTON', 'HAL', 'H']:
            raise NotImplementedError 

        elif self.method.upper() in ['SOBOL', 'SOB', 'S']:
            raise NotImplementedError 

        else:
            raise NotImplementedError 

