#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
from ._experimentbase import ExperimentBase
import numpy as np
import scipy.stats as stats
from uqra.utilities.helpers import num2print

class MCS(ExperimentBase):
    def __init__(self, distributions=None):
        """
        Random sampling based on scipy.stats distributions 
        """
        self.ndim, self.distributions = super()._set_distributions(distributions)
        self.filename = 'DoE_Mcs'

    def __str__(self):
        if self.distributions is None:
            message = 'Random samples, no distribution has been set yet }'

        else:
            dist_names = []
            for idist in self.distributions:
                try:
                    dist_names.append(idist.name)
                except:
                    dist_names.append(idist.dist.name)
            message = 'Random samples from: {}'.format(dist_names)

        return message

    def samples(self, loc=0, scale=1, size=1):
        """
        Generating random variables of shape 'size'
        """
        size = super()._check_int(size)
        if self.distributions is None:
            raise ValueError('Distributions must be specified first before sampling')

        locs, scales = super()._set_parameters(loc, scale)
        u_samples = []

        for idist, iloc, iscale in zip(self.distributions, locs, scales):
            u_samples.append(idist.rvs(loc=iloc, scale=iscale, size=size))
        ## udpate filename 
        self.filename = self.filename + num2print(np.array(size, ndmin=2).shape[-1])
        return np.array(u_samples)

class CLS(ExperimentBase):
    def __init__(self, cls_type, d):
        """
        Sampling based on Pluripotential distributions used in Christoffel least square 
        """
        self.ndim     = int(d)
        self.cls_type = cls_type.upper()
        self.filename = 'DoE_' + self.cls_type.capitalize()

    def __str__(self):

        if self.distributions is None:
            message = '{:d}d-{:s}sampling}'.format(self.ndim, self.cls_type)
        return message

    def samples(self, loc=0, scale=1, size=1):
        """
        Generating CLS variables of shape 'size'
        """
        size = super()._check_int(size)
        locs, scales = super()._set_parameters(loc, scale)

        if self.cls_type.upper() in ['CHRISTOFFEL1', 'CLS1']:
            """
            Ref: Table 2 of "A Christoffel function weighted least squares algorithm for collocation approximations, 
            Akil Harayan, John D. Jakeman, and Tao Zhou"
            Domain D    | Sampling density domain   |     Sampling density v(y) |
            [-1,1]^d    |   [1,1]^d                 |   1/(pi ^d \prod _{i=1} ^ d sqrt(1-x_i^2)) 
            """

            u = stats.uniform.rvs(0,np.pi,size=(self.ndim, size))
            x = np.cos(u)

        elif self.cls_type.upper() in ['CHRISTOFFEL2', 'CLS2']:
            ###  Acceptance and rejection sampling 
            ## 1. Sampling 
            ## 2. Accept and reject samples 
            ## Method to sample uniformly inside a d-dimensional ball
            ## 1. z ~ normal(0,1), z <- z/|z|, then z ~ uniformally on d-ball sphere  
            ## 2. u ~ uniform(0,1) 
            ## 3. z * u^{1/d}
            z = stats.norm.rvs(0,1,size=(self.ndim, size))
            z = z/np.linalg.norm(z, axis=0)
            u = np.cos(stats.uniform.rvs(0,np.pi/2,size=(1, size)))
            x = z * u 

        elif self.cls_type.upper() in ['CHRISTOFFEL3', 'CLS3']:
            raise NotImplementedError

        elif self.cls_type.upper() in ['CHRISTOFFEL4', 'CLS4']:
            #### Sampling from Ball with radius sqrt(2). 
            ## 1. z ~ normal(0,1), z / norm(z): uniformly sampling on Ball^d sphere
            ## 2. u ~ uniform(0,1 )

            z = stats.norm.rvs(0,1,size=(self.ndim, size))
            z = z/np.linalg.norm(z, axis=0)
            u = stats.beta.rvs(self.ndim/2.0,self.ndim/2.0 + 1,size=size)
            u = np.sqrt(2 * u) 
            x = z * u 

        elif self.cls_type.upper() in ['CHRISTOFFEL5', 'CLS5']:
            raise NotImplementedError

        u = np.array(x)
        assert u.shape[0] == self.ndim

        for i in range(self.ndim):
            u[i] = u[i] * scales[i] + locs[i]

        ## udpate filename 
        self.filename = self.filename + num2print(np.array(size, ndmin=2).shape[-1])
        return u

                
class QuasiMCS(ExperimentBase):

    """
    Quasi random sampling based on low-discrepency sequences 
    """
    def __init__(self, quasi_type=None):
        raise NotImplementedError 

    def __str__(self):

        raise NotImplementedError 
        # if self.distributions is None:
            # message = 'Random samples, no distribution has been set yet }'

        # else:
            # dist_names = []
            # for idist in self.distributions:
                # try:
                    # dist_names.append(idist.name)
                # except:
                    # dist_names.append(idist.dist.name)
            # message = 'Random samples from: {}'.format(dist_names)

        # return message

    def samples(self, loc=0, scale=1, size=1):
        """
        Generating random variables of shape 'size'
        """
        raise NotImplementedError 

        # if self.method.upper() in ['HALTON', 'HAL', 'H']:
            # raise NotImplementedError 

        # elif self.method.upper() in ['SOBOL', 'SOB', 'S']:
            # raise NotImplementedError 


