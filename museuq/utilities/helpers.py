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
import os, sys, warnings, collections, csv, itertools, math
from statsmodels.distributions.empirical_distribution import ECDF

Ecdf2plot = collections.namedtuple('Ecdf2plot', ['x','y'])

def ordinal(n):
    return "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])

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
        if int(__str[0]) == 1:
            return 'E'+__str[-1] 
        else:
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
        DATA_DIR= os.path.join('G:','My Drive','MUSE_UQ_DATA')
        MODEL_DIR_DATA_ID = get_gdrive_folder_id(MODEL_NAME)
    elif current_os.upper() == 'DARWIN':
        DATA_DIR= '/Users/jinsongliu/External/MUSE_UQ_DATA'
        MODEL_DIR_DATA_ID = get_gdrive_folder_id(MODEL_NAME)
    elif current_os.upper() == 'LINUX':
        MODEL_DIR_DATA_ID = get_gdrive_folder_id(MODEL_NAME)
        DATA_DIR= '/home/jinsong/Box/MUSE_UQ_DATA'
    else:
        raise ValueError('Operating system {} not found'.format(current_os))    
    
    DATA_DIR  = os.path.join(DATA_DIR,MODEL_NAME,r'Data')
    # MODEL_DIR_DATA_ID = GDRIVE_DIR_ID[MODEL_NAME.upper()] 

    # Create directory for model  
    try:
        os.makedirs(MODEL_DIR)
        os.makedirs(DATA_DIR)
        os.makedirs(FIGURE_DIR)
        # print(r'Data, Figure directories for model {} is created'.format(MODEL_NAME))
    except FileExistsError:
        # one of the above directories already exists
        # print(r'Data, Figure directories for model {} already exist'.format(MODEL_NAME))
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

    # print(r'   * {:<15s} : {}'.format('Uploading', filename[26:]))
    print(r'   * {:<15s} : {}'.format('Uploading', filename))
    while (not upload_success) and n_times2upload <=5:
        command = ' '.join([gdrive, 'upload ', filename,' --parent ', parent_id])
        upload_message = os.popen(command).read().upper()

        if 'UPLOADED' in upload_message: 
            upload_success=True
            rm_file_command = ' '.join(['rm ', filename])
            os.popen(rm_file_command)
        else:
            # print(r"Progress {:2.1%}".format(x / 10), end="\r")
            print('   * {:<7s} : {:d}/ 5'.format('trial', n_times2upload),end="\r")
        n_times2upload +=1

def get_exceedance_data(x,prob=1e-3,isExpand=False, return_index=False):
    """
    Retrieve the exceedance data for specified prob from data set x
    if x is 2D array, calculate exceedance row-wise
    Arguments:
        x: array-like data set of shape(m, n)
        prob: exceedance probability
        isExpand: boolean type, default False
          if True: retrieve exceedance data for 1st row, and sort the rest rows based on first row  
          if False: retrieve exceeance data for each row 
    Return:
        if x.ndim == 1, return [ecdf.x, ecdf.y [,ecdf_index]],
            np.ndarray of shape (2,k) or (3,k), k: number of exceedance samples to plot easily
        else
            if isExpand:
                return (m, k)
            else:
                return a list of (3,n) arrays
    """
    x = np.array(x)
    if x.ndim == 1 or np.squeeze(x).ndim == 1:
        result = _get_exceedance1d(np.squeeze(x), prob=prob, return_index=return_index)
    else:
        if isExpand:
            res1row_x, res1row_y, res1row_idx = _get_exceedance1d(x[0,:], prob=prob, return_index=True)
            res1row_idx = np.array(res1row_idx, dtype=np.int32)
            result = [res1row_x,]
            for irow in x[1:,:]:
                irow_sorted = irow[res1row_idx[1:]]
                irow_sorted = np.insert(irow_sorted, 0, irow_sorted.size)
                result.append(irow_sorted) 
            result.append(res1row_y)
            result = np.vstack(result)
        else:
            if np.isscalar(prob):
                prob = [prob,] * x.shape[0]
            elif len(prob) == 1:
                prob = [prob[0],] * x.shape[0]
            else:
                assert (len(prob) == x.shape[0]), "Length of target probability should either be 1 or equal to number of rows in x, but len(prob)={:d}, x.shape[0]={:d}".format(len(prob), x.shape[0])
            result = [_get_exceedance1d(irow, prob=iprob, return_index=return_index) for irow,iprob in zip(x, prob)]
    ## each result element corresponds to one result for each row in x. Number of element in result could be different. Can only return list
    return result

def get_stats(data, qoi2analysis='ALL', stats2cal=[1,1,1,1,1,1,0], axis=0):
    """
    Return column-wise statistic properties for given qoi2analysis and stats2cal
    Parameters:
        - data: file type or array-like (ndim/ntime_series, nsamples/nqois)
            > file: file full name must given, including extension
            > array-like of shape (nsampes, m,n) or (nsample,)
                m,n: format for one solver simulation
                nsamples: number of simulations run
        - qoi2analysis: array of integers, Column indices to analysis
        - stats2cal: array of boolen, indicators of statistics to calculate
          [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
    Return:
        list of calculated statistics [ np.array(nstats, nqoi2analysis)] 
    """
    print(r' > Calculating statistics...')
    print(r'   * {:<15s} '.format('post analysis parameters'))
    qoi2analysis = qoi2analysis if qoi2analysis is not None else 'ALL'
    print(r'     - {:<15s} : {} '.format('qoi2analysis', qoi2analysis))
    stats_list = ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing']
    print(r'     - {:<15s} : {} '.format('statistics'  , list(itertools.compress(stats_list, stats2cal)) ))

    if isinstance(data, (np.ndarray, np.generic)):
        data_sets = [data,]
    elif isinstance(data, str):
        data_sets = [_load_data_from_file(data)]
    elif isinstance(data, list) and isinstance(data[0], (np.ndarray, np.generic)):
        data_sets = data
    elif isinstance(data, list) and isinstance(data[0], str):
        data_sets = [_load_data_from_file(iname) for iname in data]
    else:
        raise ValueError('Input format for get_stats are not defined, {}'.format(type(data)))
        
    res = []
    for i, idata_set  in enumerate(data_sets):
        idata_set = idata_set if qoi2analysis == 'ALL' else idata_set[qoi2analysis]
        stat = _get_stats(np.squeeze(idata_set), stats=stats2cal, axis=axis)
        res.append(stat)
        print(r'     - Data set : {:d} / {:d}    -> Statistics output : {}'.format(i, len(data_sets), stat.shape))
    res = res if len(res) > 1 else res[0]
    return np.array(res)

def _get_stats(data, stats=[1,1,1,1,1,1,0], axis=0):
    """ Calculate column-wise statistics of data
        Parameters:
          - data: np.ndarray of shape (m,) or (m,n)
          - stats: list, indicator of statistics to be calculated, 
            [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
        Return: 
            ndarray (nstats, nsamples/nqois) 
    """
    res = np.array([                               \
            np.mean(data, axis=axis),              \
            np.std(data, axis=axis),               \
            scistats.skew(data, axis=axis),        \
            scistats.kurtosis(data, axis=axis),    \
            np.max(abs(data), axis=axis),          \
            np.min(abs(data), axis=axis),          \
            _up_crossing(data, axis=axis),         \
            ])
    idx = np.reshape(stats, res.shape[0])
    res = res[idx==1,:]
    return res

def _get_exceedance1d(x,prob=1e-3, return_index=False):
    """
    return sub data set retrieved from data set x
    Parameters:
        data: 1d array of shape(n,)
        prob: exceedance probability

    Return:
        ndarray of shape (3,k)
        (0,:): ecdf.x, sorted values for x
        (1,:): ecdf.y, corresponding probability for each x
        (2,:): exceedance value corresponding to specified prob (just one number, to be able to return array, duplicate that number to have same size as of (1,:))

    """
    assert np.array(x).ndim == 1
    x_ecdf  = ECDF(x)
    n       = len(x_ecdf.x)
    sort_idx= np.argsort(x)

    if n <= 1.0/prob:
        if return_index:
            index_ =np.insert(sort_idx, 0, sort_idx.size)
            result = np.vstack((x_ecdf.x, x_ecdf.y, index_ ))
        else:
            result = np.vstack((x_ecdf.x, x_ecdf.y))
        warnings.warn('\n Not enough samples to calculate failure probability. -> No. samples: {:d}, failure probability: {:f}'.format(n, prob))
    else:
        ### When there are a large number of points, exceedance plot with all data points will lead to large figures. Usually, it is not necessary to use all data points to have a decent exceedance plots since large portion of the data points will be located in the 'middle' region. Here we collapse data points to a reasonal number
        prob_index = -int(prob * n)   # index of the result value at targeted exceedance prob
        prob_value = x_ecdf.x[prob_index] # result x value
        _, index2return = np.unique(np.round(x_ecdf.x[:prob_index], decimals=2), return_index=True)
        # remove 'duplicate' values up to index prob_index, wish to have much smaller size of data when making plot
        # append the rest to the array
        x1 = x_ecdf.x[index2return]
        y1 = x_ecdf.y[index2return]
        sort_idx1 = sort_idx[index2return]

        x2 = x_ecdf.x[prob_index:]
        y2 = x_ecdf.y[prob_index:]
        sort_idx2 = sort_idx[prob_index:]
        x  = np.hstack((x1,x2))
        y  = np.hstack((y1,y2))
        v  = np.hstack((sort_idx1, sort_idx2)) 
        if return_index:
            result = np.vstack((x,y,v))
        else:
            result = np.vstack((x,y))
    return result

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

    return np.mean(data, axis=axis) *0

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


def _load_data_from_file(fname, data_dir=os.getcwd()):
    """
    load data from give file at current directory (default)
    """
    try:
        data = np.load(fname)
    except FileNotFoundError:
        ## get a list of all files in data_dir
        allfiles = [f for f in os.listdir(data_dir) if os.isfile(join(data_dir, f))]
        similar_files = [f for f in allfiles if f.startswith(fname)]
        if len(similar_files) == 1:
            data = np.load(similar_files[0])
        else:
            raise ValueError('FileNotFoundError, {:d} similar files exists'.format(len(similar_files)))
    return data


