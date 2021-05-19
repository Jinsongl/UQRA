#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy.linalg as LA
import numpy as np
from sklearn import linear_model
from sklearn import metrics
import uqra
import math
import itertools
import os
import scipy.stats as stats
import scipy.io
import scipy.special as sp_special
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import model_selection
import pickle
np.set_printoptions(precision=4)
import pandas as pd
import numpy as np
import random, copy
import multiprocessing as mp
r = 0
theta = 1
uqra_env   = uqra.environment.NDBC46022()
solver     = uqra.Solver('RM3', 2, distributions=uqra_env)
model_name = solver.nickname
poly_name  = 'Hem'
# pf = 1e-4
pf = 1/(50*365.25*24)
model_dir  = os.path.join('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples', model_name)
data_dir   = os.path.join(model_dir, 'Data')
data_dir_test = os.path.join(model_dir, 'TestData')
figure_dir = os.path.join(model_dir, 'Figures')
# merge global data results
data = scipy.io.loadmat(os.path.join(data_dir, 'RM3.mat'))
headers = [iheader[0][0] for iheader in data['headers'].T]
print(headers)
model_params = uqra.Modeling('PCE')
model_params.degs    = np.arange(2,11) #[2,6,10]#
model_params.ndim    = solver.ndim
model_params.basis   = 'Heme'
model_params.dist_u  = stats.uniform(0,1)  #### random CDF values for samples
model_params.fitting = 'OLSLAR' 
model_params.n_splits= 50
model_params.alpha   = 2
model_params.num_test= int(1e7)
model_params.num_pred= int(1e7)
model_params.pf      = np.array([1.0/(365.25*24*50)])
model_params.abs_err = 1e-4
model_params.rel_err = 2.5e-2
model_params.n_jobs  = mp.cpu_count()
model_params.update_basis()
model_params.info()
doe_params = uqra.ExperimentParameters('MCS', 'S')
doe_params.update_poly_name(model_params.basis)
doe_params.num_cand  = int(1e5)
    
data_dir_cand = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random'
for ss in range(1, 10):
    random.seed(0)
    filename = 'RM3_Adap2Hem_McsSE5R0S{:d}_global.npy'.format(ss)
    print(filename)
    data_itheta = np.load(os.path.join(data_dir, filename), allow_pickle=True).tolist()
    for i, data_ideg in enumerate(data_itheta):
        print('     - ', data_ideg.__dict__.keys())
        ndim, deg = data_ideg.ndim, data_ideg.deg
        print('     - ndim=', data_ideg.ndim, 'deg=', data_ideg.deg, )
        idoe_params = copy.deepcopy(doe_params)
        idoe_params.ndim = ndim
        idoe_params.deg  = int(deg)
        idoe_params.update_filenames(filename_template=None)
        filename_cand = idoe_params.fname_cand(r)
        # filename_design = idoe_params.fname_design(r)
        print('     - {:<23s} : {}'.format(' Candidate filename'  , filename_cand  ))

        if filename_cand:
            data_cand = np.load(os.path.join(data_dir_cand, filename_cand))
            data_cand = data_cand[:ndim,random.sample(range(data_cand.shape[1]), k=idoe_params.num_cand)]
            data_cand = data_cand * deg ** 0.5 if doe_params.doe_sampling.upper() in ['CLS4', 'CLS5'] else data_cand
            print('       {:<23s} : {}'.format(' shape', data_cand.shape))
        else:
            data_cand = None
            print('       {:<23s} : {}'.format(' shape', data_cand))
        orth_poly = uqra.poly.orthogonal(2, deg, 'Heme')
        pce_model = uqra.PCE(orth_poly)
        n_samples = math.ceil(2.0*pce_model.num_basis)
        xi_train, idx_optimal = idoe_params.get_samples(data_cand, orth_poly, n_samples, x0=[], 
                active_index=None, initialization='RRQR', return_index=True) 
 
        idx = [np.argwhere(data_ideg.xi_train.T==ixi)[0][0] for ixi in xi_train.T]
        assert np.array_equal(data_ideg.xi_train[:,idx], xi_train)
        print(data_ideg.y_train.shape)
        data_ideg.xi_train = data_ideg.xi_train[:, idx]
        data_ideg.x_train  = data_ideg.x_train[:, idx]
        data_ideg.y_train  = data_ideg.y_train[idx,:]
        data_itheta[i] = data_ideg
    print(os.getcwd())
    np.save(filename, data_itheta, allow_pickle=True)

