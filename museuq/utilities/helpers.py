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

def get_exceedance_data(x,prob=1e-3,**kwargs):
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
    isExpand    = kwargs.get('isExpand'     , False)
    return_index= kwargs.get('return_index' , False)
    return_all  = kwargs.get('return_all'   , False) 

    if x.ndim == 1 or np.squeeze(x).ndim == 1:
        result = _get_exceedance1d(np.squeeze(x), prob=prob, return_index=return_index, return_all=return_all)
    else:
        if isExpand:
            ## get sorting index with the first row
            res1row_x, res1row_y, res1row_idx = _get_exceedance1d(x[0,:], prob=prob, return_index=True, return_all=return_all)
            res1row_idx = np.array(res1row_idx, dtype=np.int32)
            result = [res1row_x,]
            ## taking care of the rest rows
            for irow in x[1:,:]:
                irow_sorted = irow[res1row_idx[1:]] ## the first element is the total number of samples
                irow_sorted = np.insert(irow_sorted, 0, irow_sorted.size)
                result.append(irow_sorted) 
            result.append(res1row_y)
            result = np.vstack(result)
        else:
            ## Geting exceedance for each row of x
            ##   If only one prob number is given, same prob will be applied to all rows
            ##   If a list of prob is given, each prob is applied to corresponding row
            if np.isscalar(prob):
                prob = [prob,] * x.shape[0]
            elif len(prob) == 1:
                prob = [prob[0],] * x.shape[0]
            else:
                assert (len(prob) == x.shape[0]), "Length of target probability should either be 1 or equal to number of rows in x, but len(prob)={:d}, x.shape[0]={:d}".format(len(prob), x.shape[0])
            result = [_get_exceedance1d(irow, prob=iprob, return_index=return_index, return_all=return_all) for irow,iprob in zip(x, prob)]
## each result element corresponds to one result for each row in x. Number of element in result could be different. Can only return list
    return result

def get_weighted_exceedance(x, **kwargs):
    """
    return row wise exceedance  
    """
    x = np.array(x)
    numbins = kwargs.get('numbins', 10)
    defaultreallimits = kwargs.get('defaultreallimits', None)
    weights = kwargs.get('weights', None)

    res = []
    for ix in x:
        res.append(_get_weighted_exceedance1d(ix, numbins=numbins, defaultreallimits=defaultreallimits, weights=weights))

    res = res[0] if len(res) == 1 else res
    return res


def get_stats(data, qoi2analysis='ALL', stats2cal=['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing'], axis=-1):
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

def _get_stats(data, stats=['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin', 'up_crossing'], axis=-1):
    """ Calculate statistics of data along specified axis
        Parameters:
          - data: np.ndarray 
          - stats: list, indicator of statistics to be calculated, 
            [mean, std, skewness, kurtosis, absmax, absmin, up_crossing]
        Return: 
            ndarray (nstats, nsamples/nqois) 
    """
    res = []

    for istats in stats:
        if istats.lower()  in ['mean', 'mu']:
            res.append(np.mean(data, axis=axis))
        elif istats.lower() in ['std', 'sigma']:
            res.append(np.std(data, axis=axis))
        elif istats.lower() in ['variance']:
            res.append(np.std(data, axis=axis)**2)
        elif istats.lower() in ['skewness', 'skew']:
            res.append(scistats.skew(data, axis=axis))
        elif istats.lower() in ['kurtosis', 'kurt']:
            res.append(scistats.kurtosis(data, axis=axis))
        elif istats.lower() in ['absmax']:
            res.append(np.max(abs(data), axis=axis))
        elif istats.lower() in ['max', 'maximum']:
            res.append(np.max(data, axis=axis))
        elif istats.lower() in ['absmin']:
            res.append(np.min(abs(data), axis=axis))
        elif istats.lower() in ['min', 'minimum']:
            res.append(np.max(data, axis=axis))
        elif istats.lower() in ['up_crossing_rate', 'up_crossing']:
            res.append(_up_crossing(data, axis=axis))

        else:
            raise FileNotFoundError


    return np.array(res)

def _get_exceedance1d(x,prob=1e-3, return_index=False, return_all=False ):
    """
    return emperical cdf from dataset x
    Parameters:
        x: 1d array of shape(n,)
        prob: exceedance probability
        return_index: boolean [default False], If true, will return the indices of sorted data to get ecdf.x
        return_all: boolean [default False], If true, return all ecdf.x ecdf.y, otherwise, compress dataset size and return

    Return:
        ndarray of shape (3,k)
        (0,:): ecdf.x, sorted values for x
        (1,:): ecdf.y, corresponding probability for each x
        if return_index:
        (2,:): indice based on ecdf.x

    """

    assert np.array(x).ndim == 1
    x_ecdf  = ECDF(x)
    n       = len(x_ecdf.x)
    sort_idx= np.argsort(x)

    if n <= 1.0/prob:
        if return_index:
            ## ECDF return size will always adding one point (ECDF=0 or ECDF=1). To make it possible to stack, inserting the total number of samples in index_
            index_ = np.insert(sort_idx, 0, sort_idx.size)
            result = np.vstack((x_ecdf.x, x_ecdf.y, index_ ))
        else:
            result = np.vstack((x_ecdf.x, x_ecdf.y))
        warnings.warn('\n Not enough samples to calculate failure probability. -> No. samples: {:d}, failure probability: {:f}'.format(n, prob))
    else:
        if return_all:
            if return_index:
                index_ =np.insert(sort_idx, 0, sort_idx.size)
                result = np.vstack((x_ecdf.x, x_ecdf.y, index_ ))
            else:
                result = np.vstack((x_ecdf.x, x_ecdf.y))
        else:

            ### When there are a large number of points, exceedance plot with all data points will lead to large figures. 
            ### Usually, it is not necessary to use all data points to have a decent exceedance plots since large portion 
            ### of the data points will be located in the 'middle' region. Here we collapse data points to a reasonal number

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

def _get_weighted_exceedance1d(x,numbins=10, defaultreallimits=None, weights=None):
    """
    return emperical cdf from dataset x
    Parameters:
        x: 1d array of shape(n,)
        prob: exceedance probability
        return_index: boolean [default False], If true, will return the indices of sorted data to get ecdf.x
        return_all: boolean [default False], If true, return all ecdf.x ecdf.y, otherwise, compress dataset size and return

    Return:
        ndarray of shape (3,k)
        (0,:): ecdf.x, sorted values for x
        (1,:): ecdf.y, corresponding probability for each x
        if return_index:
        (2,:): indice based on ecdf.x

    """
    res = scistats.cumfreq(x,numbins=numbins, defaultreallimits=defaultreallimits, weights=weights)
    cdf_x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size, res.cumcount.size)
    cdf_y = res.cumcount/x.size
    ecdf_y = 1- cdf_y
    ecdf_x = cdf_x

    return np.array([ecdf_x, ecdf_y])

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


