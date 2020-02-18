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
from museuq.utilities.helpers import num2print

class ExperimentBase(object):
    """
    Abstract class for experimental design
    """

    def __init__(self, distributions=None, random_seed=None):
        """
        Initialization of ExperimentBase:
        Arguments:
            distributions: None or distribution object from scipy.stats or list of scipy.stats objects
            random_seed: 
        """
        self.random_seed = random_seed
        if distributions is None:
            self.distributions = distributions
            self.ndim = None
        elif isinstance(distributions, (list, tuple)):
            self.distributions = []
            for idist in distributions:
                assert hasattr(sp.stats, idist.name)
                self.distributions.append(idist)
            self.ndim = len(self.distributions)

        elif hasattr(sp.stats, distributions.name):
            self.distributions = [distributions,]
            self.ndim = int(1)
        else:
            raise ValueError('distributions should from scipy.stats or None for empty instance')

        self.x          = []  # DoE values in physical space 
        self.u          = []  # DoE values in u-space (Askey)
        self.filename   = 'DoE'

    def samples(self, n_samples=None, theta=[0,1]):
        """
        Return DoE samples based on specified DoE methods

        Arguments:
            n: int or list of int, number of samples to be sampled
        Returns:
            np.ndarray
        """

        if self.distributions is None:
            pass 
            ### Optimal design doesn't have to specify distributions, use generated candidates

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





