#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""
Single degree of freedom with time series external loads
"""

import uqra
from uqra.solver._solverbase import SolverBase
import os, numpy as np, scipy as sp
# from scipy.integrate import odeint
# from scipy.optimize import brentq
from .PowerSpectrum import PowerSpectrum
from uqra.environment import Kvitebjorn 
from tqdm import tqdm
import scipy.stats as stats


class Solver(SolverBase):
    """

    """
    def __init__(self, name, ndim, **kwargs):
        """
        Empty Solver object
        """
        super().__init__()
        self.name       = name 
        self.nickname   = name
        self.ndim       = int(ndim)
        self.distributions  = kwargs.get('distributions', None)

    def __str__(self):
        message = '{:s} (ndim={:d})'.format(self.name, self.ndim)
        return message

    # def run(self):
        # raise NotImplementedError

            
    # def map_domain(self):
        # raise NotImplementedError




