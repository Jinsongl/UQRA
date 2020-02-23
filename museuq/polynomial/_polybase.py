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
        self.ndim = self._check_int(d)
        self.deg  = self._check_int(deg)
        self.coef = coef
        self.multi_index = multi_index
        self.basis_degree= None
        self.name     = ''
        self.nickname = ''
        self.num_basis = 0

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
        self.n_gauss = self._check_int(n)
        # if self.n_gauss < self.deg +1:
            # warnings.warn('n < p + 1')

    def fit_quadrature(self):
        raise NotImplementedError

    def fit_regression(self):
        raise NotImplementedError

    def _check_int(self, x):
        if x is None:
            return None
        else:
            int_x = int(x)
            if int_x != x:
                raise ValueError("deg must be integer")
            if int_x < 0:
                raise ValueError("deg must be non-negative")
            return int_x

    def __call__(self, x):
        """
        Evaluate polynomials at given values x
        """
        raise NotImplementedError
    def __get_basis_degree(self):
        """
        self.basis_degree, list of tuples containing degree component for each basis function. i.e. (3,0,2) -> x1**3 + x2**0 + x3**2
        """

        basis_degree = []
        if self.multi_index.lower() == 'total':
            for icombination in itertools.product(range(self.deg + 1), repeat=self.ndim):
                if sum(icombination) <= self.deg:
                    basis_degree.append(icombination)
        else:
            raise NotImplementedError

        return basis_degree

    def _update_basis(self):
        """
        Series basis polynomials of degree self.deg

        Three terms related to basis:
        1. self.basis = [], a list of basis functions
        2. self.num_basis, int: number of basis functions, i.e. len(self.basis)
        3. self.basis_degree, list of tuples containing degree component for each basis function. i.e. (3,0,2) -> x1**3 + x2**0 + x3**2

        """
        if self.deg is None or self.ndim is None:
            self.basis_degree = None
            self.num_basis = None
        else:
            self.basis_degree = self.__get_basis_degree()
            self.num_basis = len(self.basis_degree)

    def set_coef(self, coef):
        raise NotImplementedError
