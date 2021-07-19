#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""

import numpy as np, scipy as sp, scipy.stats as stats
import os, sys, warnings, collections, csv, itertools, math
from statsmodels.distributions.empirical_distribution import ECDF as mECDF
import copy

Ecdf2plot = collections.namedtuple('Ecdf2plot', ['x','y'])

def ordinal(n):
    return "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout.close()
    sys.stdout = sys.__stdout__


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

def check_int(x):
    if x is None:
        return None
    else:
        int_x = int(x)
        if int_x != x:
            raise ValueError("deg must be integer, {} given".format(x))
        if int_x < 0:
            raise ValueError("deg must be non-negative, {} given".format(x))
        return int_x

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

def time_in_range(start, end, x):
    """Return true if x is in the range [start, end]"""
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end

