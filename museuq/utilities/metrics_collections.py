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
import scipy as sp
from numpy.linalg import norm
## import regression metrics from scikit-learn
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score

# ord	norm for matrices	        norm for vectors
# None	Frobenius norm	                2-norm
# ‘fro’	Frobenius norm	                –
# ‘nuc’	nuclear norm	                –
# inf	max(sum(abs(x), axis=1))	max(abs(x))
# -inf	min(sum(abs(x), axis=1))	min(abs(x))
# 0	–	                        sum(x != 0)
# 1	max(sum(abs(x), axis=0))	as below
# -1	min(sum(abs(x), axis=0))	as below
# 2	2-norm (largest sing. value)	as below
# -2	smallest singular value	i       as below
# other	–	                        sum(abs(x)**ord)**(1./ord)


from scipy.stats import moment
from scipy.stats.mstats import mquantiles

def moments(y_true, y_pred, m=1):
    res = [sp.stats.moment(y_true,moment=m), sp.stats.moment(y_pred,moment=m)]
    return res

def upper_tails(y_true, y_pred, prob=[0.75, 0.9, 0.99], alphap=0.4, betap=0.4, axis=None, limit=()):
    res = [sp.stats.mstats.mquantiles(y_true,prob=prob,alphap=alphap,betap=betap,axis=axis,limit=limit),
           sp.stats.mstats.mquantiles(y_pred,prob=prob,alphap=alphap,betap=betap,axis=axis,limit=limit)]
    return res

