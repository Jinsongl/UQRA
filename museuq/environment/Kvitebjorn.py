#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""
Long-term extreme response analysis of offshore structures by combining importance sampling with subset simulation Ying Min Low, Xiaoxu Huang

Haver S. On the prediction of extreme wave crest heights. In: Proceedings of the 7th international workshop on wave hindcasting and forecasting, Meteorological Service of Canada. 2002.

Johannessen, K. and Nygaard, E. (2000): “Metocean Design Criteria for Kvitebjørn”, Statoil Report , C193-KVB-N-FD-0001, Rev. date: 2000-12-14, Stavanger, 2000.

"""
import numpy as np
import scipy.stats as stats
import chaospy as cp

# Sequence of conditional distributions based on Rosenblatt transformation 
def dist_Hs(x, key='value'):
    """
    Return Hs distribution based on x value
    key: 
       - value: physical value of Hs
       - ppf: point percentile value of Hs
    """
    mu_Hs    = 0.77
    sigma_Hs = 0.6565
    dist1    = cp.LogNormal(mu_Hs, sigma_Hs)
    Hs_shape = 1.503
    Hs_scale = 2.691
    dist2    = cp.Weibull(Hs_shape,Hs_scale)
    h0       = 2.9
    cdf_h0   = dist1.cdf(h0)

    if key.lower() == 'value':
        dist = dist1 if x <= h0 else dist2
    elif key.lower() == 'ppf':
        dist = dsit1 if x <= cdf_h0 else dist2

    else:
        raise ValueError('Key value: {} is not defined'.format(key))

def dist_Tp(Hs):
    a1 = 1.134
    a2 = 0.892
    a3 = 0.225
    b1 = 0.005
    b2 = 0.120
    b3 = 0.455
    mu_tp = a1 + a2* Hs**a3 
    sigma_tp = np.sqrt(b1 + b2*np.exp(-b3*Hs))
    return cp.LogNormal(mu_tp, sigma_tp)

# Sequence of conditional distributions based on Rosenblatt transformation 
def samples_hs(x):
    """
    Random Hs values could be generated either by
        - x: int, number of samples needed
        - x: nd.array, target cdf values
    """

    mu_Hs    = 0.77
    sigma_Hs = 0.6565
    Hs_shape = 1.503
    Hs_scale = 2.691
    h0       = 2.9
    cdf_h0   = 0.6732524353557928


    if isinstance(x, int):
        samples1 = np.random.lognormal(mu_Hs, sigma_Hs,x)
        samples2 = np.random.weibull(Hs_shape,x) * Hs_scale
        samples_hs = np.where(samples1<=h0,samples1, samples2)

    elif isinstance(x, np.ndarray):
        x = np.array(x)
        assert min(x) >=0 and max(x) <=1
        samples1 = stats.lognorm.ppf(x, s=sigma_Hs, loc=mu_Hs)
        samples2 = Hs_scale * (-np.log(1-x)) **(1/Hs_shape)
        samples_hs = np.where(samples1<=cdf_h0,samples1, samples2)
    else:
        raise ValueError('samples_hs(x) takes either int or ndarray, but {} is given'.format(type(x)))

    return samples_hs

def samples_tp(hs,tp_cdf=None):
    """
    Generate tp sample values based on given Hs values:
    Optional:
    if tp_cdf is given, corresponding cdf values for Tp|Hs are returns, otherwise a random number from Tp|Hs distribution is returned
    or given Hs cdf values
    """

    a1 = 1.134
    a2 = 0.892
    a3 = 0.225
    b1 = 0.005
    b2 = 0.120
    b3 = 0.455

    mu_tp   = a1 + a2* hs**a3 
    sigma_tp= np.sqrt(b1 + b2*np.exp(-b3*hs))
    if tp_cdf is None:
        samples = np.array([ np.random.lognormal(ihs, itp, 1) for ihs, itp in zip(mu_tp, sigma_tp)])
    else:
        samples = np.array([ stats.lognorm.ppf(ip, s=isigma, loc=imu) for ip, imu, isigma in zip(tp_cdf, mu_tp, sigma_tp)])

    return samples 

def samples(x):
    """
    Return samples from joint (Hs, Tp) distributions
    parameters:
    x: 
      1. int, number of samples to be generated
      2. ndarray of cdf values 
        - shape (2, n) : n samples to be generated based on values cdf values of x
    """
    if isinstance(x, int):
        samples0 = samples_hs(x)
        samples1 = samples_tp(samples0)
    elif isinstance(x, np.ndarray):
        if x.shape[0] != 2:
            raise ValueError(' CDF values for two random variables (Hs,Tp) are expected for Kvitebjorn, but {} was given'.format(x.shape[0])) 
        samples0 = samples_hs(x[0,:])
        samples1 = samples_tp(samples0, x[1,:])

    else:
        raise ValueError('samples(x) takes either int or ndarray, but {} is given'.format(type(x)))

    return np.array([samples0, samples1])


