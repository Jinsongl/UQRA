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

class Solver(object):
    """
    Abstract class for solvers
    """

    def __init__(self,*args, **kwargs):
        """

        """
        self.name    = ''
        self.theta_m = [] 
        self.theta_s = [] 
        self.y       = []
        self.y_stats = []

    def run(self, x, *args, **kwargs):
        """
        run solver with input variables
        Parameters:
            x: np.ndarray input data of shape(ndim, nsamples) 
              or str for input filename
        Returns:
            No returns
        """
        raise NotImplementedError()

