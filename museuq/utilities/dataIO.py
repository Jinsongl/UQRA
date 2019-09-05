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

def save_data(data, filename, dirname=None):
    """
    Save given data(array) with specified filename in dirname folder
    """
    if len(data) == 2:
        datax, datay = data
    elif len(data) == 1:
        datax, datay = data, None
    else:
        raise ValueError('Input data shape not specified, len(data)={:d}'.format(len(data)))

    ## save datax
    print('data shape: {}'.format(datax.shape))
    # print 'saving file:' + filename + '...'
    # with open(filename, mode) as fileid:
        # writer = csv.writer(fileid, delimiter=',')
        # if data.ndim == 1:
            # writer.writerow(data)
        # else:
            # for v in data: 
                # writer.writerow(['{:8.4e}'.format(float(x)) for x in v])
    np.savetxt(filename,data,fmt='%.4e',delimiter=',')
    if isMove:
        os.rename(filename,dirname+'/'+filename)


def _save_datax(data, filename, dirname=None):
    for idata in data:
        print('saving data of shape: {}'.format(idata.shape))
        data_all = np.hstack((data_all, idata))



