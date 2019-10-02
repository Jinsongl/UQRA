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
import chaospy as cp
import scipy.stats as stats
def make_circle(r,n=100):
    t = np.linspace(0, np.pi * 2.0, n)
    t = t.reshape((len(t), 1))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.hstack((x, y)).T
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


## Environment Contour

def EC(p=1e-4, n=100):
    beta        = stats.norm.ppf(1-p)
    print('{:<20s}:{:.4f}'.format('Reliability Index', beta))
    EC_normal   = make_circle(beta,n=n)
    EC_norm_cdf = stats.norm.cdf(EC_normal)
    EC_samples  = samples(EC_norm_cdf)
    print(EC_normal.shape)
    print(EC_samples.shape)
    return np.vstack((EC_normal, EC_samples))

# Sequence of conditional distributions based on Rosenblatt transformation 

def hs_pdf(hs):
    """

    """

    mu_Hs    = 0.77
    sigma_Hs = 0.6565
    Hs_shape = 1.503
    Hs_scale = 2.691
    h0       = 2.9

    hs1_pdf = stats.lognorm.pdf(hs, s=sigma_Hs, loc=0, scale=np.exp(mu_Hs)) 
    hs2_pdf = stats.weibull_min.pdf(hs, c=Hs_shape, scale=Hs_scale) 
    hs_pdf  = np.where(hs<=h0, hs1_pdf, hs2_pdf)
    return hs_pdf

# Sequence of conditional distributions based on Rosenblatt transformation 
def samples_hs(u):
    """
    Random Hs values could be generated either by
        - u: int, number of samples needed
        - u: nd.array, target cdf values
    """

    mu_Hs    = 0.77
    sigma_Hs = 0.6565
    Hs_shape = 1.503
    Hs_scale = 2.691
    h0       = 2.9

    u = np.array(u)
    assert min(u) >=0 and max(u) <=1
    samples1 = stats.lognorm.ppf(u, s=sigma_Hs, loc=0, scale=np.exp(mu_Hs))
    samples2 = stats.weibull_min.ppf(u, c=Hs_shape, loc=0, scale=Hs_scale) #0 #Hs_scale * (-np.log(1-u)) **(1/Hs_shape)
    samples_hs = np.where(samples1<=h0,samples1, samples2)

    return np.squeeze(samples_hs)

def samples_tp(hs,u_tp=None):
    """
    Generate tp sample values based on given Hs values:
    Optional:
    if u_tp is given, corresponding cdf values for Tp|Hs are returns, otherwise a random number from Tp|Hs distribution is returned
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

    if u_tp is None:
        samples = np.random.lognormal(mean=mu_tp, sigma=sigma_tp, size=len(hs)) 
    else:
        samples = stats.lognorm.ppf(u_tp, s=sigma_tp, loc=0, scale=np.exp(mu_tp)) 

    return np.squeeze(samples)

def samples(x):
    """
    Return samples from joint (Hs, Tp) distributions
    parameters:
    x: 
      1. int, number of samples to be generated
      2. ndarray of cdf values 
        - shape (2, n) : n samples to be generated based on values cdf values of x
    """
    if isinstance(x, (int, float)):
        x = int(x)
        x = np.random.uniform(0,1,x)
        samples0 = samples_hs(x)
        samples1 = samples_tp(samples0)

    elif isinstance(x, np.ndarray):
        if x.ndim == 1 or x.shape[0] == 1:
            samples0 = samples_hs(x)
            samples1 = samples_tp(samples0)
        elif x.ndim ==2 and x.shape[0] == 2:
            samples0 = samples_hs(x[0,:])
            samples1 = samples_tp(samples0, u_tp=x[1,:])
        else:
            raise ValueError(' CDF values for either only Hs or both (Hs,Tp) are expected for Kvitebjorn, but {} was given'.format(x.shape[0])) 
    else:
        raise ValueError('samples(x) takes either int or ndarray, but {} is given'.format(type(x)))

    return np.array([samples0, samples1])


