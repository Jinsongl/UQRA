#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import context
import chaospy as cp
import numpy as np
# import envi, doe, solver, utilities

import os, sys
from envi import environment
from metaModel import metaModel
from simParams import simParameter
from run_sim import run_sim
from utilities.get_exceedance_data import get_exceedance_data
from utilities.upload2gdrive import upload2gdrive
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel,WhiteKernel)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def get_gdrive_folder_id(folder_name):
    """
    Check if the given folder_name exists in Google Drive. 
    If not, create one and return the google drive ID
    Else: return folder ID directly
    """
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
    MODEL_DIR_DATA_ID = get_gdrive_folder_id(MODEL_NAME.upper())



    # Create directory for model  
    print('------------------------------------------------------------')
    print('►►► Making directories for model {}'.format(MODEL_NAME))
    print('------------------------------------------------------------')
    try:
        os.makedirs(MODEL_DIR)
        os.makedirs(DATA_DIR)
        os.makedirs(FIGURE_DIR)
    except FileExistsError:
        # one of the above directories already exists
        pass
    print('WORKING_DIR: {}'.format(WORKING_DIR))
    print('+-- MODEL: {}'.format(MODEL_DIR))
    print('|   +-- {:<6s}: {}'.format('FIGURE',FIGURE_DIR))
    print('|   +-- {:<6s}: {}'.format('DATA',DATA_DIR))
    return MODEL_DIR_DATA_ID, DATA_DIR, FIGURE_DIR


def main():
    
    ## ------------------------------------------------------------------- ###
    ##  Parameters set-up 
    ## ------------------------------------------------------------------- ###
    prob_failures        = [1e-3, 1e-4, 1e-5, 1e-6]              # Exceedence probability
    # data_train_params   = [[1e6], 'R']      # nsamples_test, sample_rule
    data_test_params    = [1e7, 10, 'R']    # nsamples_test, nrepeat, sample_rule
    MODEL_NAME          = 'Bench1'
    # model_def       = ['bench4']        # [solver function name, error_dist.name, [params], size]
    # model_def       = [MODEL_NAME,'normal']        # [solver function name, error_dist.name, [params], size]
    DATA_DIR_ID,DATA_DIR,FIGURE_DIR = make_output_dir(MODEL_NAME)
    ## ------------------------------------------------------------------- ###
    ##  Define Solver parameters ###
    ## ------------------------------------------------------------------- ###
    ## >>> 1. Choose Wiener-Askey scheme random variable
    ##            # |   zeta    | Wiener-Askey chaos | support
    ## # ==============================================================
    ## # Continuous | Gaussian  | Hermite-chaos      |  (-inf, inf)
    ##              | Gamma     | Laguerre-chaos     |  [0, inf ) 
    ##              | Beta      | Jacobi-chaos       |  [a,b] 
    ##              | Uniform   | Legendre-chaos     |  [a,b] 
    ## # --------------------------------------------------------------
    ## # Discrete   | Poisson   | 
    ##              | Binomial  | 
    ##              | - Binomial| 
    ##              | hypergeometric
    ## 
    dist_zeta = cp.Normal(0,1)  # shape=1, scale=1, shift=0

    ## >>> 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## >>> 3. Define independent random variable in physical problems
    dist_x = cp.Normal(5,2) # normal mean = 0, normal std=0.25

    np.random.seed(3)
    ## ------------------------------------------------------------------- ###
    ##  Build surrogate model 
    ## ------------------------------------------------------------------- ###

    noise_dir   = 'DATA_NOISE_FREE' ## DATA_NOISE_FREE or DATA_NOISE_'name'
    noise_type  = '' ## '' (blank) if noise free else name of noise distribution 
    ndoe2train  = [10] 
    # basis_kernel=['Kernels'] # For Kriging
    data_file   = r'bench1_realization_noise_free.npy'
    data_set    = np.load(os.path.join(DATA_DIR, data_file))
    x_samples   = data_set[0]
    y_samples   = data_set[1]
    zeta_samples= data_set[2]

    for idoe in ndoe2train:

        ## ******************************************************************
        ## Build PCE surrogate model
        ## ******************************************************************

        doe_type    = 'DoE_Quadrature'
        fname_train_in = '_'.join(['TrainData',noise_type, doe_type+r'{:d}.npy'.format(idoe)])
        train_data = np.load(os.path.join(DATA_DIR,noise_dir, fname_train_in))
        x_train    = np.squeeze(train_data[0][0])
        x_weight   = np.squeeze(train_data[0][1])
        zeta_weight= x_weight 
        y_train    = np.squeeze(train_data[1])
        zeta_train = np.squeeze(train_data[2][0])

        ## Build PCE model with different basis orders
        metamodel_class, metamodel_basis = 'PCE', [2,3,4] 
        metamodel_params= {'cal_coeffs': 'GQ'}
        fname_train_out = '_'.join(['TrainRes',metamodel_class,doe_type])
        pce_model   = metaModel(metamodel_class, metamodel_basis, dist_zeta, **metamodel_params)
        pce_model.fit_model(zeta_train[np.newaxis,:], y_train, weight=zeta_weight)
        y_validate  = pce_model.predict(zeta_train)
        y_samples_pred= pce_model.predict(zeta_samples) 
        train_res   = [ x_train, x_weight , y_train, zeta_train,
                        np.array(y_validate),
                        np.array(x_samples),
                        np.array(y_samples),
                        np.array(zeta_samples),
                        np.array(pce_model.predict(zeta_samples))]
        np.save(os.path.join(DATA_DIR, fname_train_out + r'{:d}'.format(idoe)), train_res)

        print('------------------------------------------------------------')
        print('►►► MCS with Surrogate Models')
        print('------------------------------------------------------------')
        np.random.seed(100)
        fname_mcs = '_'.join(['Pred', metamodel_class, doe_type+str(idoe), 'MCS'+'{:.0E}'.format(data_test_params[0])[-1]])
        fname_mcs_path = os.path.join(DATA_DIR, fname_mcs)

        y_pred_mcs_ecdf = [ [] for _ in range(len(prob_failures))]
        for r in range(data_test_params[1]): # number of repeated MCS
            zeta_mcs        = dist_zeta.sample(data_test_params[0], data_test_params[2])[np.newaxis,:]
            y_pred_mcs      = pce_model.predict(zeta_mcs)

            # upload2gdrive(fname_mcs_path+r'R{:d}'.format(r),  y_pred_mcs, DATA_DIR_ID)
            for ipf, pf in enumerate(prob_failures): 
                print(' ► Calculating ECDF of MCS data and retrieve data to plot, pf={:.0E}...'.format(pf))
                y_pred_mcs_ecdf[ipf].append(get_exceedance_data(y_pred_mcs, prob_failure=pf))
        for ipf, pf in enumerate(prob_failures): 
            rfname_mcs  = '_'.join(['Ecdf_pf'+'{:.0E}'.format(pf)[-1], metamodel_class,doe_type+str(idoe), 'MCS'+'{:.0E}'.format(data_test_params[0])[-1]])
            np.save(os.path.join(DATA_DIR,rfname_mcs), y_pred_mcs_ecdf[ipf])

        # ## ******************************************************************
        # ## >>> Build GPR surrogate model
        # ## ******************************************************************

        doe_type    = 'DoE_Quadrature'
        fname_train_in = '_'.join(['TrainData',noise_type, doe_type+r'{:d}.npy'.format(idoe)])
        train_data = np.load(os.path.join(DATA_DIR,noise_dir,fname_train_in))
        x_train    = np.squeeze(train_data[0][0])
        y_train    = np.squeeze(train_data[1])
        zeta_train = np.squeeze(train_data[2][0])

        kernels  = [20.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10))]
        metamodel_class, metamodel_basis = 'GPR', kernels 
        metamodel_params = {'n_restarts_optimizer':10}
        fname_train_out  = '_'.join(['TrainRes', metamodel_class, doe_type])
        gpr_model   = metaModel(metamodel_class, metamodel_basis, dist_zeta, **metamodel_params)
        gpr_model.fit_model(x_train[np.newaxis,:], y_train)
        y_validate  = gpr_model.predict(x_train[np.newaxis,:], return_std=True)
        train_data  = [ x_train, y_train, zeta_train,
                        np.array(y_validate),
                        np.array(x_samples),
                        np.array(y_samples),
                        np.array(zeta_samples),
                        np.array(gpr_model.predict(x_samples[np.newaxis,:], return_std=True))]
        np.save(os.path.join(DATA_DIR, fname_train_out + r'{:d}'.format(idoe)), train_data)

        ## samples_y: trained samples in x + samples plot in selected domain
        samples_x = np.sort(np.hstack((x_train, np.linspace(np.min(x_samples), np.max(x_samples), len(x_samples))))) ###???????
        samples_y = gpr_model.sample_y(samples_x[np.newaxis,:], n_samples=10)
        np.save(os.path.join(DATA_DIR, fname_train_out + r'{:d}_sample_y'.format(idoe)), samples_y)

        ## Calculate log marginal likelihood at specified theta values
        res = 100
        length_scale   = np.logspace(-1,1,res)
        noise_level    = np.logspace(0,3,res)
        theta          = [noise_level,length_scale]
        lml_theta_grid = gpr_model.log_marginal_likelihood(theta)
        np.save(os.path.join(DATA_DIR, fname_train_out + r'{:d}_lml'.format(idoe)), lml_theta_grid)
        np.save(os.path.join(DATA_DIR, fname_train_out + r'{:d}_params'.format(idoe)), gpr_model.metamodel_coeffs)


        print('------------------------------------------------------------')
        print('►►► MCS with Surrogate Models')
        print('------------------------------------------------------------')
        np.random.seed(100)
        fname_mcs = '_'.join(['Pred', metamodel_class, doe_type+str(idoe), 'MCS'+'{:.0E}'.format(data_test_params[0])[-1]])
        fname_mcs_path = os.path.join(DATA_DIR, fname_mcs)
        y_pred_mcs_ecdf = [ [] for _ in range(len(prob_failures))]
        for r in range(data_test_params[1]):
            _x_samples = dist_x.sample(data_test_params[0], data_test_params[2])[np.newaxis,:]
            # print('x_samples.shape: {}'.format(_x_samples.shape))
            y_pred_mcs    = gpr_model.predict(_x_samples, return_std=True)
            ## y_pred_mcs shape (2, n_samples) (mean, std)
            # print(y_pred_mcs.shape)
            # print('y_pred_mcs shape: {}'.format(y_pred_mcs.shape))
            # upload2gdrive(fname_mcs_path+r'R{:d}'.format(r),  y_pred_mcs, DATA_DIR_ID)

            for ipf, pf in enumerate(prob_failures): 
                print(' ► Calculating ECDF of MCS data and retrieve data to plot, pf={:.0E}...'.format(pf))
                y_pred_mcs_ecdf[ipf].append(get_exceedance_data(y_pred_mcs[0,:], prob_failure=pf))
        for ipf, pf in enumerate(prob_failures): 
            rfname_mcs  = '_'.join(['Ecdf_pf'+'{:.0E}'.format(pf)[-1], metamodel_class,doe_type+str(idoe), 'MCS'+'{:.0E}'.format(data_test_params[0])[-1]])
            np.save(os.path.join(DATA_DIR,rfname_mcs), y_pred_mcs_ecdf[ipf])



if __name__ == '__main__':
    main()

