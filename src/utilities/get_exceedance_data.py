#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import warnings
import numpy as np
import collections
import time
from statsmodels.distributions.empirical_distribution import ECDF

Ecdf2plot = collections.namedtuple('Ecdf2plot', ['x','y'])


def get_exceedance_data(x,prob_failure=None):
    """
    Return
    If x is 2d array, get exceedance column wise


    """
    exceedance = []
    if x.ndim == 1:
        exceedance = _get_exceedance_data(x, prob_failure=prob_failure)
    elif x.ndim == 2:
        for ix_col in x:
            exceedance.append(_get_exceedance_data(ix_col, prob_failure=prob_failure))

    return exceedance


def _get_exceedance_data(x,prob_failure=None):
    """
    return sub data set retrieved from data set x

    data size: 1/(prob_failure * 10) to end
    """
    x_ecdf = ECDF(x)
    n_samples = len(x_ecdf.x)
    prob_failure = 1e-3 if prob_failure is None else prob_failure

    if n_samples <= 1.0/prob_failure:
        exceedance = (x_ecdf.x, x_ecdf.y, x_ecdf.y)
        warnings.warn('\n Not enough samples to calculate failure probability. -> No. samples: {:d}, failure probability: {:f}'.format(n_samples, prob_failure))
    else:
        exceedance_index = -int(prob_failure * n_samples)
        exceedance_value = x_ecdf.x[exceedance_index]
        _, idx1 = np.unique(np.round(x_ecdf.x[:exceedance_index], decimals=2), return_index=True)
        x1 = x_ecdf.x[idx1]
        y1 = x_ecdf.y[idx1]
        x2 = x_ecdf.x[exceedance_index:]
        y2 = x_ecdf.y[exceedance_index:]
        x  = np.hstack((x1,x2))
        y  = np.hstack((y1,y2))
        v  = exceedance_value * np.ones(x.shape)
        exceedance = (x,y,v) 
    return exceedance



