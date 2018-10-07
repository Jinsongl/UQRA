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

def up_crossing(data, axis=0):
    pass

def get_stats(data, stats=[1,1,1,1,1,1,0]):
    """ Calculate the statistics of data
        data: file type or array-like (ndim/ntime_series, nsamples/nqois)
            > file: file full name must given, including extension
            > array-like:
                nsamples/nqois: number of samples realized or number of QoIs to
                    be analyzed.
                ndim/ntime_series: each column is either a full time series or
                    a full relization of each sample

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
        data = np.asfarray(data)
    res = []
    if stats[0] == 1:
        res.append(np.mean(data, axis=0))
    if stats[1] == 1:
        res.append(np.std(data, axis=0))
    if stats[2] == 1:
        res.append(scistats.skew(data, axis=0))
    if stats[3] == 1:
        res.append(scistats.kurtosis(data, axis=0))
    if stats[4] == 1:
        res.append(np.max(abs(data), axis=0))
    if stats[5] == 1:
        res.append(np.min(abs(data), axis=0))
    if stats[6] == 1:
        res.append(up_crossing(data, axis=0))
    res = np.asfarray(res)
    return res


def main(stats=[1,1,1,1,1,1,0]):

    cwd = os.getcwd()
    filelist = [f for f in os.listdir(cwd) if f.startswith("SDOF")] 
    print("Number of files: {:d}".format(len(filelist)))
    # get_stats(filelist,qoi2analysis)
    d = get_stats('SDOF1.csv', stats)
    # print d



if __name__ == "__main__":
    main()







