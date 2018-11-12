#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
import scipy as sp
import scipy.stats as scistats
import csv
import os

def _up_crossing(data, axis=0):
    pass

def _moving_avg(data, window=3,axis=None):
    ret = np.cumsum(data, dtype=float, axis=axis)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

def _moving_square(data, window=3, axis=None):
    return _moving_avg(data**2, window=window, axis=axis)

def _moving_std(data, window=3,axis=None):
    exp_x2 = _moving_square(data, window=window, axis=axis)
    expx_2 = _moving_avg(data, window=window, axis=axis) **2
    return exp_x2 - expx_2

def get_stats(data, stats=[1,1,1,1,1,1,0]):
    """ Calculate the statistics of data
        data: file type or array-like (ndim/ntime_series, nsamples/nqois)
            > file: file full name must given, including extension
            > array-like of shape (nsampes, m,n) or (nsample,)
                m,n: format for one solver simulation
                nsamples: number of simulations run

        stats: list, indicator of statistics to be calculated, 
            [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
        Return: ndarray (nstats, nsamples/nqois) 
    """
    if isinstance(data,str):
        if data[-3:] =="csv":
            deli = ','
        else:
            deli = None 
        data = np.genfromtxt(data, delimiter=deli)
    else:
        data = np.array(data)
    # if data is just a column or row vector (samples, ), get the stats for that vector
    # return would be of shape (nstats,)
    if data.ndim == 1:
        res = np.array(np.mean(data), np.std(data), scistats.skew(data), scistats.kurtosis(data), np.max(abs(data)), np.min(abs(data)), up_crossing(data))
        # use filter to select stats
        res = [istat for i, istat in enumerate(res) if stats[i]]
    else:
        assert data.ndim == int(3)

        res = np.empty(data.shape[0], int(sum(stats)), data.shape[2])
        if stats[0] == 1:
            res = np.append(res, np.mean(data, axis=1), axis=1)
        if stats[1] == 1:
            res = np.append(res, np.std(data, axis=1), axis=1)
        if stats[2] == 1:
            res = np.append(res, scistats.skew(data, axis=1), axis=1)
        if stats[3] == 1:
            res = np.append(res, scistats.kurtosis(data, axis=1), axis=1)
        if stats[4] == 1:
            res = np.append(res, np.max(abs(data), axis=1), axis=1)
        if stats[5] == 1:
            res = np.append(res, np.min(abs(data), axis=1), axis=1)
        if stats[6] == 1:
            res = append(res, _up_crossing(data, axis=1), axis=1)
        # if stats[7] == 1:
            # res = append(res, _moving_avg(data), axis=1)
        # if stats[8] == 1:
            # res = append(res, _moving_std(data), axis=1)
    
    return res







