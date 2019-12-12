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
from museuq.utilities.decorators import (NotFittedError, check_valid_values, missing_method_scipy_wrapper)
from museuq.utilities import dataIO 

class ExperimentalDesign(object):
    """
    Abstract class for experimental design
    """

    def __init__(self, random_seed=None):
        self.random_seed = random_seed
        self.x      =[]  # DoE values in physical space 
        self.u      =[]  # DoE values in u-space
        self.y      =[]  # DoE output corresponding to DoE x values 
        self.filename      = ''

    def samples(self):
        """
        Return DoE samples based on specified DoE methods

        Arguments:
            n: int or list of int, number of samples to be sampled
        Returns:
            np.ndarray
        """
        raise NotImplementedError

    def save_data(self, data_dir):
        """
        save input variables to file
        """
        raise NotImplementedError

