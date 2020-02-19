#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""

import numpy as np
import math
import itertools
import warnings

__all__ = ['PolyBase']


class PolyBase(object):
    """
    An abstract base class for series classes.
    """
    def __init__(self, d=None, deg=None, coef=None, domain=None, window=None, multi_index='total'):
        self.ndim = self.check_int(d)
        self.deg  = self.check_int(deg)
        self.coef = coef
        self.multi_index = multi_index
        self.basis_degree= None
        self.name     = ''
        self.nickname = ''
        self.num_basis = 0


    def get_basis(self):
        """
        Series basis polynomials of total degree p

        """
        self.basis_degree = []
        if self.multi_index.lower() == 'total':
            for icombination in itertools.product(range(self.deg + 1), repeat=self.ndim):
                if sum(icombination) <= self.deg:
                    self.basis_degree.append(icombination)
            self.num_basis = len(self.basis_degree)
        else:
            raise NotImplementedError


    def set_ndim(self, ndim):
        """
        set the dimension of polynomial
        """
        raise NotImplementedError

    def set_degree(self, deg):
        """
        set the degree of polynomial
        """
        raise NotImplementedError

    
    def Vandermonde(self, x, normed=True):
        raise NotImplementedError

    def gauss_quadrature(self, n):
        self.n_gauss = self.check_int(n)
        if self.n_gauss < self.deg +1:
            warnings.warn('n < p + 1')

    def fit_quadrature(self):
        raise NotImplementedError

    def fit_regression(self):
        raise NotImplementedError

    def check_int(self, x):
        if x is None:
            return int(0)
        else:
            int_x = int(x)
            if int_x != x:
                raise ValueError("deg must be integer")
            if int_x < 0:
                raise ValueError("deg must be non-negative")
            return int_x


    def _update_num_basis(self):

        if self.multi_index.lower() == 'total':
            self.num_basis = round(math.factorial(self.ndim + self.deg)/math.factorial(self.ndim)/math.factorial(self.deg))

        elif multi_index.lower() == 'tensor':
            self.num_basis = self.ndim ** self.deg

