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

class Solver(object):
    """
    Abstract class for solvers
    """

    def __init__(self):
        self.name = ''
    def run(self, x):
        """
        run solver with input variables
        Parameters:
            x: np.ndarray input data of shape(ndim, nsamples) 
              or str for input filename
        Returns:
            No returns
        """
        raise NotImplementedError()

