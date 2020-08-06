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
import scipy as sp
from uqra.utilities.helpers import num2print
from uqra.polynomial._polybase import PolyBase
from uqra.utilities.helpers import isfromstats

class ExperimentBase(object):
    """
    Abstract class for experimental design
    """

    def __init__(self, samplingfrom=None):
        """
        Initialization of ExperimentBase:
        Arguments:
            samplingfrom: 
                distributions to generate random samples
                or polynomial to generate quadrature samples 
        """
        self.samplingfrom= samplingfrom
        if self.samplingfrom is None:
            self.ndim = None
        ### sampling from distributions froms scipy.stats 
        #> 1. samplingfrom are list or tuple
        elif isinstance(self.samplingfrom, (list, tuple)):
            self.distributions = []
            for idist in self.samplingfrom:
                assert isfromstats(idist)
                self.distributions.append(idist)
            self.ndim = len(self.distributions)
        #> 2. Just one distribution is given 
        elif isfromstats(self.samplingfrom):
            self.distributions = [self.samplingfrom,]
            self.ndim = int(1)
            
        ### sampling are based on gauss quadrature
        elif isinstance(self.samplingfrom, PolyBase):
            self.polynomial = self.samplingfrom
            self.ndim = self.samplingfrom.ndim
        else:
            raise ValueError('Sampling from type {} are not defined'.format(type(samplingfrom)))

    def _update_parameters(self, n_samples=None, theta=[0,1]):
        """
        Return DoE samples based on specified DoE methods

        Arguments:
            n: int or list of int, number of samples to be sampled
        Returns:
            np.ndarray
        """

        if self.samplingfrom is None:
            pass 
            ### Optimal design doesn't have to specify samplingfrom, use generated candidates

        else:
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
        self.n_samples= 0 if n_samples is None else int(n_samples)
        self.filename = self.filename + num2print(n_samples)

    def adaptive(self):
        """
        Adaptive DoE with new samples 

        """
        raise NotImplementedError





