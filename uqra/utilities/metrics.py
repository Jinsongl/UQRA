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
    res =  np.array(sp.stats.mstats.mquantiles(a, prob=prob,axis=axis,limit=limit))

    if multioutput.lower() == 'uniform_average':
        if res.ndim == 1:
            res = res 
        elif res.ndim == 2:
            res = np.mean(res,axis=1, dtype=np.float64)
        else:
            raise ValueError('Input data, as a sequence or array of dimension at most 2.')
        # print('{:s} :{}'.format('mquantiles', res.shape))
    elif multioutput.lower() == 'raw_values':
        res = np.squeeze(res)
    else:
        raise NotImplementedError
    res = np.squeeze(res)
    try:
        res = res.item()
    except ValueError:
        res = res
    return res

def moment(a, moment=1, axis=0, nan_policy='propagate',multioutput='uniform_average'):
    """
    Parameters
        aarray_like
        Input array.

        momentint or array_like of ints, optional
        Order of central moment that is returned. Default is 1.

        axisint or None, optional
        Axis along which the central moment is computed. Default is 0. If None, compute over the whole array a.

        nan_policy{‘propagate’, ‘raise’, ‘omit’}, optional
        Defines how to handle when input contains nan. The following options are available (default is ‘propagate’):

        ‘propagate’: returns nan

        ‘raise’: throws an error

        ‘omit’: performs the calculations ignoring nan values

    Returns
        n-th central momentndarray or float
        The appropriate moment along the given axis or over all values if axis is None. The denominator for the moment calculation is the number of observations, no degrees of freedom correction is done.
    """
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

def leave_one_out_error(X,y, is_adjusted=True):
    """
    Calculate leave one out error for linear regression

    Derivation: 
    https://stats.stackexchange.com/questions/164223/proof-of-loocv-formula
    https://robjhyndman.com/hyndsight/loocv-linear-models/

    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n,P = X.shape
    assert len(y) == n, 'Design matrix of shape {}, Observation vector y of shape: {}'.format(X.shape, y.shape)

    # H1 = np.linalg.inv(np.dot(X.T, X))
    # H  = np.dot(X, np.dot(H1, X.T)) ## projection matrix
    # y_hat = np.dot(H, y)
    # print(y_hat[:5])

    Q, R = np.linalg.qr(X)
    H = np.dot(Q, Q.T)
    y_hat = np.dot(H, y)
    h  = np.diagonal(H)
    error_loo = np.mean(((y - y_hat) / (1-h))**2, dtype = np.float64)

    ## correcting factor derived in [49] for regression using a small experimental design 
    ## [O. Chapelle, V. Vapnik, Y. Bengio, Model selection for small sample regression, Mach. Learn. 48 (1) (2002) 9–23.]

    if is_adjusted:
        C = np.dot(X.T, X)/n
        correcting_factor =  n/(n-P) * (1+ np.trace(np.linalg.inv(C))/n)
        error_loo =  error_loo * correcting_factor
    return error_loo

def mean_squared_relative_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True):

    output_errors = np.average(((y_true - y_pred)/y_true) ** 2, axis=0, weights=sample_weight)
    if not squared:
        output_errors = np.sqrt(output_errors)

    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None
    return np.average(output_errors, weights=multioutput)
