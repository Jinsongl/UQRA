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

def getStats(data, stats=[1,1,1,1,1,1,0]):
    """ Calculate the statistics of data
        data: file type or array-like (m x n), each row is a time series
            if is a file, file full name must given, including extension
        stats: list, indicator of statistics to be calculated, [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
        Return: ndarray (m,sum(stats)) 
    """
    if isinstance(data,str):
        if data[-3:] =="csv":
            deli = ','
        else:
            deli = None 
        data = np.genfromtxt(data, delimiter=deli)
    else:
        data = np.array(data)
    # data = data[rows2use,:].T
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
        res.append(up_crossing(data,axis=0))
    res = np.asarray(res).T
    return res

# def getStats_files(filelist,stats=[1,1,1,1,1,1,0], outputNames=None):
    # """ 
        # filelist: list of file names 
        # stats: list, indicator of statistics to be calculated, [mean, std, skewness, kurtosis, absmin, absmax, up_crossing]
        # outputNames: list of names of each row, used for output file names. If not specified, output0, output1,... will be used 
    # """
    # if outputNames is None:
        # outputNames = []
        # for i in range(len(rows2use)):
            # outputNames.append('output'+str(i)+'.csv')

    # rows2usetats = [[]]  # each element store the statistics for one row 
    # for i in range(len(rows2use)-1):
        # rows2usetats.append([])
    
    # indicator_steps = int(len(filelist)/10)
    # for i, filename in enumerate(filelist):
        # if i % indicator_steps == 0:
            # print "processing file number:  ", i 
        # filestats = getStats(filename,rows2use,stats)
        # for j in range(len(rows2usetats)):
            # rows2usetats[j].append(filestats[j,:])
            # # print rows2usetats[1]


    # # print len(rows2usetats[0])
    # for i in range(len(rows2use)):
        # with open(outputNames[i],'wb') as fileid:
            # writer = csv.writer(fileid)
            # writer.writerows(rows2usetats[i])

# def main(rows2use, stats=[1,1,1,1,1,1,0]):

    # cwd = os.getcwd()
    # filelist = [f for f in os.listdir(cwd) if f.startswith("SDOF")] 
    # print "Number of files: " , len(filelist)
    # getStats_files(filelist,rows2use)
    # # d = getStats('SDOF1.csv',rows2use, stats)
    # # print d



# if __name__ == "__main__":
    # main([1,2])







