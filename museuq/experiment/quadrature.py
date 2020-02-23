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
from museuq.experiment._experimentbase import ExperimentBase
from museuq.utilities.helpers import num2print
import itertools
import collections

class QuadratureDesign(ExperimentBase):
    """ Experimental Design with Gauss-Quadrature

    This class should be expanded to accept different polynomials

    """

    def __init__(self, poly):
        """
        Parameters:
            basis_names: str or list of str, polynomial basis form. hem: Hermtie, leg: Legendre
            n    : int, number of quadrature points in each dimension
        """
        super().__init__(samplingfrom=poly)
        self.quad_name= poly.name 
        self.filename = '_'.join(['DoE', 'Quad'+ ''.join(poly.nickname)])
        self.x = []  # Gaussian quadrature node 
        self.w = []  # Gaussian quadrature weights 

    def __str__(self):
        return('Gauss Quadrature: {}'.format(quad_name))

    def samples(self, n, theta=[0,1]):
        """
        Sampling n Gauss-Quadrature pints from distributions 
        Arguments:
            n_samples: int, number of samples 
            theta: list of [loc, scale] parameters for distributions
            For those distributions not specified with (loc, scale), the default value (0,1) will be applied
        Return:
            Experiment samples of shape(ndim, n_samples)
        Return:
            Experiment samples in space (samples for each orders)
        """
        super().samples(n, theta=theta)
        x, w = self.polynomial.gauss_quadrature(self.n_samples, loc=self.loc, scale=self.scale)
        self.x = x
        self.w = w
        return x, w

