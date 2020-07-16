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
from .polynomial_chaos_expansion import PolynomialChaosExpansion

class mPCE(PolynomialChaosExpansion):
    """
    Class to build multiple polynomial chaos expansion (mPCE) model
    """

    def __init__(self, p=None, dist=None, random_seed = None):
        self.name = 'Multiple Polynomial Chaos Expansion'
        super().__init__(p=p, dist=dist ,random_seed=random_seed)

    def predict(self,x, **kwargs):
        """
        Predict using surrogate models 

        Arguments:	
        X : array-like, shape = (n_features/ndim, nsamples)
            Query points where the surrogate model are evaluated

        Returns:	
        y : list of array, array shape = (nsamples, )
            predicted value from surrogate models at query points
        """
        if self.fit_method.upper() in ['GALERKIN', 'GLK','PROJECTION','OLS', 'WLS','OLSLARS','LASSOLARS']:
            y_pred = self.metamodels(*x).T ## [nsamples, nQoI ]
            n_samples, n_models = y_pred.shape
            idx = np.random.randint(0, n_models, size=n_samples)
            y_pred = np.choose(idx, y_pred.T)
        else:
            raise NotImplementedError

        print(r'   * {:<25s} : {}'.format('Prediction output', y_pred.shape))
        return y_pred



