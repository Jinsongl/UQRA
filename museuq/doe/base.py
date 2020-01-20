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

class ExperimentalDesign(object):
    """
    Abstract class for experimental design
    """

    def __init__(self, n_samples = None, random_seed=None, dist_names=None, dist_theta=None, ndim=1):
        self.random_seed = random_seed
        self.x          = []  # DoE values in physical space 
        self.u          = []  # DoE values in u-space (Askey)
        self.y          = []  # DoE output corresponding to DoE x values 
        self.filename   = 'DoE'
        self.ndim       = ndim
        self.n_samples  = n_samples if n_samples is None else int(n_samples)
        ### reformating dist_names and dist_theta to have the same length
        
        if self.ndim == 1:
            if isinstance(dist_names, str) or dist_names is None:
                self.dist_names = [dist_names,]
            elif isinstance(dist_names, list):
                assert len(dist_names) == 1
                self.dist_names = dist_names
            else:
                raise ValueError('dist_names accept either str or list of str, but {} is given '.format(type(self.dist_names)))

            if isinstance(dist_theta, tuple) or dist_theta is None:
                self.dist_theta = [dist_theta, ]
            elif isinstance(dist_theta, list):
                assert len(dist_theta) == 1
                self.dist_theta = dist_theta
            else:
                raise ValueError('dist_theta accept either tuple or list of tuple, but {} is given '.format(type(self.dist_names)))
        else:
            if isinstance(dist_names, str) or dist_names is None:
                self.dist_names = [dist_names,] * self.ndim
            elif isinstance(dist_names, list):
                self.dist_names = dist_names
                ## if a list of dist_names is given but not enough, setting deafult distributions being uniform
                for _ in range(len(dist_names), self.ndim):
                    self.dist_names.append('uniform')
            else:
                raise ValueError('dist_names accept either str or list of str, but {} is given '.format(type(self.dist_names)))

            if isinstance(dist_theta, tuple) or dist_theta is None:
                self.dist_theta = [dist_theta, ] * self.ndim
            elif isinstance(dist_theta, list):
                ### appending None for theta's which are not specified 
                ### when sampling, default theta values defined in scipy.stats.distributions will be used
                self.dist_theta = dist_theta
                for _ in range(len(dist_theta), self.ndim):
                    self.dist_theta.append(None)
            else:
                raise ValueError('dist_theta accept either tuple or list of tuple, but {} is given '.format(type(self.dist_names)))

        assert (len(self.dist_names) == len(self.dist_theta) == self.ndim), ' {:ndim} {:ndim} {:ndim}'.format(len(self.dist_names),len(self.dist_theta),self.ndim)

    def samples(self):
        """
        Return DoE samples based on specified DoE methods

        Arguments:
            n: int or list of int, number of samples to be sampled
        Returns:
            np.ndarray
        """
        raise NotImplementedError
    def adaptive(self):
        """
        Adaptive DoE with new samples 

        """
        raise NotImplementedError





