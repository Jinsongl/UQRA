#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""

import csv
import os
import numpy as np

def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


def isFilesExist(Dir,filenames):
    res = []
    for filename in filenames:
        res.append(os.path.isfile(Dir+'/'+filename))
    return sum(res)

def setfilename(params):
    i = 0
    filenames = [params.outfileName]
    for j in xrange(params.numRows):
        filenames.append(params.outfileName + str(j))
    if params.scheme == 'QUAD':
        dirname = params.site +'/'+ params.scheme + '_' + '{:0>2d}'.format(list(params.schemePts)[-1])[:2]+'_'+ params.rule.upper() 
    else:
        dirname = params.site +'/'+ params.scheme + '_' + '{:2.0E}'.format(params.nsamplesDone)[:2]+'{:2.0E}'.format(params.nsamplesDone)[-1]+'_'+ params.rule.upper() 
    NEWDIR = './Data/'+dirname
    if not os.path.exists(NEWDIR):
        os.makedirs(NEWDIR)

    extension = '_' + '{:0>3d}'.format(i)+'.csv' 
    filenames_new = [filename + extension for filename in filenames]
    # while os.path.isfile(NEWDIR+'/'+filenames[0]+extension) or os.path.isfile(NEWDIR+'/'+filenames[1]+extension):
    while isFilesExist(NEWDIR, filenames_new):
        i+=1
        extension = '_' + '{:0>3d}'.format(i)+'.csv' 
        filenames_new = [filename + extension for filename in filenames]
    # filenames = [filename + extension for filename in filenames]
    params.updateDir(NEWDIR) 
    return filenames_new

def save_data(data, filename, dir_name=None, tags=None):
    """
    Parameters:
    data:
      1. ndarray: directly save data
      2. list of ndarray: save each element in list individually
    """
    ## if data is ndarray type, direct save data with given filename, no tag needed
    print('===>>> Saving data to: {}'.format(dir_name))

    if isinstance(data, (np.ndarray, np.generic)):
        np.save(os.path.join(dir_name, filename + '{}'.format(tags[0]) ), data)
    ## if data is list of ndarray type, save each ndarray in list with given filename differentaed with tag 
    elif isinstance(data, list) :
        assert len(data) == len(tags), "Length of data set to save and length of tags available must be same, but len(data)={}, len(tags)={}".format(len(data), len(tags))
        for idata, itag in zip(data, tags):
            np.save(os.path.join(dir_name, filename + '{}'.format(itag) ), idata)
    # elif isinstance(data, list) and isinstance(data[0], (np.ndarray, np.generic)):
        # assert len(data) == len(tags), "Length of data set to save and length of tags available must be same, but len(data)={}, len(tags)={}".format(len(data), len(tags))
        # for idata, i in zip(data, tags):
            # np.save(os.path.join(dir_name, filename + '{}'.format(i) ), idata)
    # elif isinstance(data, list) and isinstance(data[0], list):
        # assert len(data[0]) == len(data[1]) == len(tags)
        # # i = 0
        # for i, data_ in enumerate(zip(*data)):
            # idata = np.concatenate(data_, axis=0)
            # np.save(os.path.join(dir_name, filename + '{}'.format(tags[i])), idata )
            # # i +=1
    else:
        raise ValueError('Input data type not defined')




def _save_datax(data, filename, dirname=None):
    for idata in data:
        print('saving data of shape: {}'.format(idata.shape))
        data_all = np.hstack((data_all, idata))



