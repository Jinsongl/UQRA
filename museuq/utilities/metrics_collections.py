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


def r2_score_adj(y_true, y_pred, num_predictor, sample_weight=None, multioutput='uniform_average'):
    """
    calculation of adjusted r2
    Parameters
        y_truearray-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

        y_predarray-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

        sample_weightarray-like of shape (n_samples,), optional
        Sample weights.

        num_predictor: number of independent variables

        multioutputstring in [‘raw_values’, ‘uniform_average’, ‘variance_weighted’] or array-like of shape (n_outputs)
        Defines aggregating of multiple output scores. Array-like value defines weights used to average scores.

            ‘raw_values’ :
            Returns a full set of scores in case of multioutput input.

            ‘uniform_average’ :
            Scores of all outputs are averaged with uniform weight.

            ‘variance_weighted’ :
            Scores of all outputs are averaged, weighted by the variances of each individual output.

    Returns
        scorefloat or ndarray of floats
        The explained variance or ndarray if ‘multioutput’ is ‘raw_values’.


    """
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
    n  = len(y_true) if y_true.ndim ==1 else y_true.shape[0]
    r2_adj = 1 - (1-r2**2) * (n-1)/(n-num_predictor-1)
    return r2_adj


def mquantiles(a, prob=[0.25, 0.5, 0.75],axis=0,limit=(),multioutput='uniform_average'):
    """
    Parameters
        a: array_like
        Input data, as a sequence or array of dimension at most 2.

        prob: array_like, optional
        List of quantiles to compute.

        axis: int, optional
        Axis along which to perform the trimming. If None, the input array is first flattened. axis=0 default

        limit: tuple, optional
        Tuple of (lower, upper) values. Values of a outside this open interval are ignored.

    """
    ### output format (nprob, noutputs)
    res =  sp.stats.mstats.mquantiles(a, prob=prob,axis=axis,limit=limit)

    if multioutput.lower() == 'uniform_average':
        if res.ndim == 1:
            res = res 
        elif res.ndim == 2:
            res = np.mean(res,axis=1, dtype=np.float64)
        else:
            raise ValueError('Input data, as a sequence or array of dimension at most 2.')
        # print('{:s} :{}'.format('mquantiles', res.shape))
    elif multioutput.lower() == 'raw_values':
        res = res
    else:
        raise NotImplementedError
    return res

def moment(a, moment=1, axis=0, nan_policy='propagate',multioutput='uniform_average'):
    ### output format (nmoments, noutputs)
    res = sp.stats.moment(a, moment=moment, axis=axis, nan_policy=nan_policy)
    if multioutput.lower() == 'uniform_average':
        if res.ndim == 1:
            res = res 
        elif res.ndim == 2:
            res = np.mean(res,axis=1, dtype=np.float64)
        else:
            raise ValueError('Input data, as a sequence or array of dimension at most 2.')
        # print('{:s} :{}'.format('moments', res.shape))
    elif multioutput.lower() == 'raw_values':
        res = res
    else:
        raise NotImplementedError
    return res




