#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""

import numpy as np, scipy as sp, scipy.stats as scistats
import os, sys, warnings, collections, csv
from statsmodels.distributions.empirical_distribution import ECDF

Ecdf2plot = collections.namedtuple('Ecdf2plot', ['x','y'])

# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print
def enablePrint():
    sys.stdout.close()
    sys.stdout = sys.__stdout__

def nextpow2(x):
    return 2**(int(x)-1).bit_length()

def num2print(n):
    if n<100:
        return '{:d}'.format(n)
    else:
        __str ='{:.0E}'.format(n) 
        return __str[0]+'E'+__str[-1] 

def get_gdrive_folder_id(folder_name):
    """
    Check if the given folder_name exists in Google Drive. 
    If not, create one and return the google drive ID
    Else: return folder ID directly
    """
    # GDRIVE_DIR_ID = {
            # 'BENCH1': '1d1CRxZ00f4CiwHON5qT_0ijgSGkSbfqv',
            # 'BENCH4': '15KqRCXBwTTdHppRtDjfFZtmZq1HNHAGY',
            # 'BENCH3': '1TcVfZ6riXh9pLoJE9H8ZCxXiHLH_jigc',
            # }
    command = os.path.join('/Users/jinsongliu/Google Drive File Stream/My Drive/MUSE_UQ_DATA', folder_name)
    try:
        os.makedirs(command)
    except FileExistsError:
        pass
    command = 'gdrive list --order folder |grep ' +  folder_name
    folder_id = os.popen(command).read()
    return folder_id[:33]

def make_output_dir(MODEL_NAME):
    """
    WORKING_DIR/
    +-- MODEL_DIR
    |   +-- FIGURE_DIR

    /directory saving data depends on OS/
    +-- MODEL_DIR
    |   +-- DATA_DIR

    """
    WORKING_DIR     = os.getcwd()
    MODEL_DIR       = os.path.join(WORKING_DIR, MODEL_NAME)
    FIGURE_DIR= os.path.join(MODEL_DIR,r'Figures')
    # DATA_DIR  = os.path.join(MODEL_DIR,r'Data')
    current_os  = sys.platform
    if current_os.upper()[:3] == 'WIN':
        DATA_DIR= "G:\My Drive\MUSE_UQ_DATA"
    elif current_os.upper() == 'DARWIN':
        DATA_DIR= '/Users/jinsongliu/External/MUSE_UQ_DATA'
    elif current_os.upper() == 'LINUX':
        DATA_DIR= '/home/jinsong/Box/MUSE_UQ_DATA'
    else:
        raise ValueError('Operating system {} not found'.format(current_os))    
    
    DATA_DIR  = os.path.join(DATA_DIR,MODEL_NAME,r'Data')
    # MODEL_DIR_DATA_ID = GDRIVE_DIR_ID[MODEL_NAME.upper()] 
    MODEL_DIR_DATA_ID = get_gdrive_folder_id(MODEL_NAME)


    # Create directory for model  
    try:
        os.makedirs(MODEL_DIR)
        os.makedirs(DATA_DIR)
        os.makedirs(FIGURE_DIR)
        # print('Data, Figure directories for model {} is created'.format(MODEL_NAME))
    except FileExistsError:
        # one of the above directories already exists
        # print('Data, Figure directories for model {} already exist'.format(MODEL_NAME))
        pass
    return MODEL_DIR_DATA_ID, DATA_DIR, FIGURE_DIR

def upload2gdrive(filename, data, parent_id):
    """
    upload file specified with filename to google drive under folder with id parent_id. 
    If upload successfully, delete filename from local. Otherwise, try 5 more times and keep filename locally 
    """
    current_os  = sys.platform
    if current_os.upper()[:3] == 'WIN':
        gdrive= "C:\Software\gdrive.exe "
    elif current_os.upper() == 'DARWIN':
        gdrive = "gdrive "
    else:
        raise ValueError('Operating system {} not found'.format(current_os)) 

    filename_name, filename_ext = os.path.splitext(filename)
    filename_ext = filename_ext if filename_ext else '.npy'
    filename = filename_name + filename_ext
    np.save(filename, data)

    upload_success = False
    n_times2upload = 1 

    # print('   ♦ {:<15s} : {}'.format('Uploading', filename[26:]))
    print('   ♦ {:<15s} : {}'.format('Uploading', filename))
    while (not upload_success) and n_times2upload <=5:
        command = ' '.join([gdrive, 'upload ', filename,' --parent ', parent_id])
        upload_message = os.popen(command).read().upper()

        if 'UPLOADED' in upload_message: 
            upload_success=True
            rm_file_command = ' '.join(['rm ', filename])
            os.popen(rm_file_command)
        else:
            # print("Progress {:2.1%}".format(x / 10), end="\r")
            print('   ♦ {:<7s} : {:d}/ 5'.format('trial', n_times2upload), end='\r')
        n_times2upload +=1

def get_exceedance_data(x,prob=None):
    """
    Retrieve the exceedance data value for specified prob from data set x
    Arguments:
        x: array-like data set 
        prob: exceedance probability
    Return:
        
    """
    exceedance = []
    x = np.array(x)

    if x.ndim == 1:
        exceedance.append(_get_exceedance_data(x, prob=prob))
    else:
        exceedance.append([_get_exceedance_data(iset, prob=prob) for iset in x])

    return exceedance

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

def _get_exceedance_data(x,prob=None):
    """
    return sub data set retrieved from data set x

    data size: 1/(prob * 10) to end
    """
    x_ecdf = ECDF(x)
    n_samples = len(x_ecdf.x)
    prob = 1e-3 if prob is None else prob

    if n_samples <= 1.0/prob:
        exceedance = (x_ecdf.x, x_ecdf.y, x_ecdf.y)
        warnings.warn('\n Not enough samples to calculate failure probability. -> No. samples: {:d}, failure probability: {:f}'.format(n_samples, prob))
    else:
        exceedance_index = -int(prob * n_samples)
        exceedance_value = x_ecdf.x[exceedance_index]
        _, idx1 = np.unique(np.round(x_ecdf.x[:exceedance_index], decimals=2), return_index=True)
        x1 = x_ecdf.x[idx1]
        y1 = x_ecdf.y[idx1]
        x2 = x_ecdf.x[exceedance_index:]
        y2 = x_ecdf.y[exceedance_index:]
        x  = np.hstack((x1,x2))
        y  = np.hstack((y1,y2))
        v  = exceedance_value * np.ones(x.shape)
        exceedance = (x,y,v) 
    return exceedance

def _central_moms(dist, n=np.arange(1,5), Fisher=True):
    """
    Calculate the first central moments of distribution dist 
    """
    mu = [dist.mom(i) for i in n]
    res = []
    mu1 = mu[0]
    mu2 = mu[1] - mu1**2
    mu3 = mu[2] - 3 * mu[0] * mu[1] + 2 * mu[0]**3 
    mu4 = mu[3] - 4* mu[0] * mu[2] + 6 * mu[0]**2*mu[1] - 3 * mu[0]**4 
    sigma = np.sqrt(mu2)
    if Fisher:
        res = [mu1/1, sigma, mu3/sigma**3, mu4/sigma**4-3]
    else:
        res = [mu1/1, sigma, mu3/sigma**3, mu4/sigma**4]

    return res
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

