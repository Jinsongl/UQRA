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

GDRIVE_DIR_ID = {
        'BENCH1': '1d1CRxZ00f4CiwHON5qT_0ijgSGkSbfqv',
        'BENCH4': '15KqRCXBwTTdHppRtDjfFZtmZq1HNHAGY',
        'BENCH3': '1TcVfZ6riXh9pLoJE9H8ZCxXiHLH_jigc',
        }

def make_output_dir(MODEL_NAME):
    WORKING_DIR     = os.getcwd()
    print(WORKING_DIR)
    MODEL_DIR       = os.path.join(WORKING_DIR, MODEL_NAME)
    FIGURE_DIR= os.path.join(MODEL_DIR,r'Figures')
    # DATA_DIR  = os.path.join(MODEL_DIR,r'Data')
    current_os  = sys.platform
    if current_os.upper()[:3] == 'WIN':
        DATA_DIR= "G:\My Drive\MUSE_UQ_DATA"
    elif current_os.upper() == 'DARWIN':
        DATA_DIR= '/Users/jinsongliu/External/MUSE_UQ_DATA'
    else:
        raise ValueError('Operating system {} not found'.format(current_os))

    DATA_DIR  = os.path.join(DATA_DIR,MODEL_NAME,r'Data')
    MODEL_DIR_DATA_ID = GDRIVE_DIR_ID[MODEL_NAME.upper()] 
    # Create directory for model  
    try:
        os.makedirs(MODEL_DIR)
        os.makedirs(DATA_DIR)
        os.makedirs(FIGURE_DIR)
    except FileExistsError:
        # one of the above directories already exists
        pass
    return MODEL_DIR_DATA_ID, DATA_DIR, FIGURE_DIR


def main():
    
    ## ------------------------------------------------------------------- ###
    ##  Parameters set-up 
    ## ------------------------------------------------------------------- ###
    pf                  = 1e-4              # Exceedence probability
    data_train_params   = [[1e6], 'R']      # nsamples_test, sample_rule
    data_test_params    = [1e7, 10, 'R']    # nsamples_test, nrepeat, sample_rule
    MODEL_NAME          = 'Bench1'
    # model_def       = ['bench4']        # [solver function name, error_dist.name, [params], size]
    # model_def       = [MODEL_NAME,'normal']        # [solver function name, error_dist.name, [params], size]
    DATA_DIR_ID,DATA_DIR,FIGURE_DIR = make_output_dir(MODEL_NAME)
    print(DATA_DIR,FIGURE_DIR)
    ## ------------------------------------------------------------------- ###
    ##  Define Solver parameters ###
    ## ------------------------------------------------------------------- ###
    ## >>> 1. Choose Wiener-Askey scheme random variable
    dist_zeta = cp.Normal(0,1)  # shape=1, scale=1, shift=0

    ## >>> 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## >>> 3. Define independent random variable in physical problems
    dist_x = cp.Normal(5,2) # normal mean = 0, normal std=0.25

    np.random.seed(3)
    ## ------------------------------------------------------------------- ###
    ##  Design of Experiments (DoEs) 
    ## ------------------------------------------------------------------- ###
    ## example:
    ## >>> 1. Fixed point:
    # doe_method, doe_rule, doe_order = 'FIX','FIX', 1e3
    # doe_params  = [doe_method, doe_rule, doe_order]
    # fix_simparam= simParameter(dist_zeta, doe_params=doe_params)
    # x_samples   = [np.linspace(-5,15,1000)[np.newaxis, :]]
    # fix_simparam.set_doe_samples(x_samples, dist_x)
    # zeta_samples = fix_simparam.sys_input_zeta[0]

    ## >>> 2. Monte Carlo:
    # # doe_method, doe_rule, doe_order = 'MC','R', [1e7]*10
    # # doe_params  = [doe_method, doe_rule, doe_order]
    # # mc_simparam = simParameter(dist_zeta, doe_params = doe_params)
    # # mc_simparam.get_doe_samples(dist_x)

    ## >>> 3. Quadrature:
    # doe_method, doe_rule, doe_order = 'GQ','hermite',[9]
    # doe_params      = [doe_method, doe_rule, doe_order]
    # quad_simparam   = simParameter(dist_zeta, doe_params = doe_params)
    # quad_simparam.get_doe_samples(dist_x)


    ## ------------------------------------------------------------------- ###
    ##  Run simulation 
    ## ------------------------------------------------------------------- ###
    ## List of simulation results (Different doe_order, sample size would change, so return list)
    # sim_output_quad= run_sim(model_def, quad_simparam)
    # sim_output_fix = run_sim(model_def, fix_simparam)
    # sim_output_mc  = run_sim(model_def, mc_simparam)

    # train_input  = quad_simparam
    # train_output = sim_output_quad[0]

    # print(train_output[0].shape)

    ## ------------------------------------------------------------------- ###
    ##  Get train data 
    ## ------------------------------------------------------------------- ###


    ## ------------------------------------------------------------------- ###
    ##  Build surrogate model 
    ## ------------------------------------------------------------------- ###

    noise_dir   = 'DATA_NOISE_FREE'
    ndoe2train  = [10] 
    data_file   = r'Bench1_noise_free.npy'
    data_set    = np.load(os.path.join(DATA_DIR, data_file))
    x_samples   = data_set[0]
    y_samples   = data_set[1]
    zeta_samples= data_set[2]

    for idoe in ndoe2train:

        ## ******************************************************************
        ## Build PCE surrogate model
        ## ******************************************************************

        doe_type    = 'Quadrature_DoE'
        train_data = np.load(os.path.join(DATA_DIR,noise_dir, r'Train_noise_free_'+doe_type+'{}.npy'.format(idoe)))
        x_train    = np.squeeze(train_data[0][0])
        x_weight   = np.squeeze(train_data[0][1])
        zeta_weight= x_weight 
        y_train    = np.squeeze(train_data[1])
        zeta_train = np.squeeze(train_data[2][0])

        # error_mean = np.squeeze(train_data[3])
        # error_std  = np.squeeze(train_data[4])
        # zeta_train  = train_input.sys_input_zeta[idoe][0]
        # zeta_weight = train_input.sys_input_zeta[idoe][1]
        # y_train     = train_output[idoe]

        metamodel_class, metamodel_basis = 'PCE', [idoe-1] 
        metamodel_params= {'cal_coeffs': 'GQ'}
        fname_train     = '_'.join([metamodel_class, 'train_validation_test', doe_type])

        pce_model   = metaModel(metamodel_class, metamodel_basis, dist_zeta, **metamodel_params)
        pce_model.fit_model(zeta_train[np.newaxis,:], y_train, weight=zeta_weight)
        y_validate  = pce_model.predict(zeta_train)
        train_data  = [ x_train, x_weight , y_train, zeta_train,
                        np.array(y_validate),
                        np.array(x_samples),
                        np.array(zeta_samples),
                        np.array(pce_model.predict(zeta_samples))]
        np.save(os.path.join(DATA_DIR, fname_train + r'{:d}'.format(idoe)), train_data)

        print('------------------------------------------------------------')
        print('►►► MCS with Surrogate Models')
        print('------------------------------------------------------------')
        np.random.seed(100)
        fname_mcs = '_'.join([metamodel_class, doe_type+str(idoe), 'MCS'])
        fname_mcs_path = os.path.join(DATA_DIR, fname_mcs)

        for r in range(data_test_params[1]):
            zeta_mcs        = dist_zeta.sample(data_test_params[0], data_test_params[2])[np.newaxis,:]
            y_pred_mcs      = pce_model.predict(zeta_mcs)

            upload2gdrive(fname_mcs_path+r'{:d}'.format(r),  y_pred_mcs, DATA_DIR_ID)

            print(' ► Calculating ECDF of MCS data and retrieve data to plot...')
            y_pred_mcs_ecdf = get_exceedance_data(y_pred_mcs, prob_failure=pf)
            rfname_mcs  = fname_mcs_path + '{:d}_ecdf'.format(r) + '.npy' 
            np.save(rfname_mcs, y_pred_mcs_ecdf)
            # print(np.round(zeta_mcs[:5],4))

        # ******************************************************************
        # >>> Build GPR surrogate model
        # ******************************************************************

        doe_type    = 'Uniform_DoE'
        train_data = np.load(os.path.join(DATA_DIR,noise_dir,\
                r'Train_noise_free_'+doe_type+'{}.npy'.format(idoe)))
        x_train    = np.squeeze(train_data[0])
        y_train    = np.squeeze(train_data[1])
        zeta_train = np.squeeze(train_data[2])

        kernels  = [20.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10))]
        metamodel_class, metamodel_basis = 'GPR', kernels 
        metamodel_params = {'n_restarts_optimizer':0}
        fname_train  = '_'.join([metamodel_class, 'train_validation_test', doe_type])

        gpr_model   = metaModel(metamodel_class, metamodel_basis, dist_zeta, **metamodel_params)
        gpr_model.fit_model(x_train[np.newaxis,:], y_train)
        y_validate  = gpr_model.predict(x_train[np.newaxis,:], return_std=True)
        train_data  = [ x_train, y_train, zeta_train,
                        np.array(y_validate),
                        np.array(x_samples),
                        np.array(zeta_samples),
                        np.array(gpr_model.predict(x_samples[np.newaxis,:], return_std=True))]
        np.save(os.path.join(DATA_DIR, fname_train + r'{:d}'.format(idoe)), train_data)

        ## samples_y
        samples_x = np.sort(np.hstack((x_train, np.linspace(-5,20,1000))))
        samples_y = gpr_model.sample_y(samples_x[np.newaxis,:], n_samples=10)
        np.save(os.path.join(DATA_DIR, fname_train + r'{:d}_sample_y'.format(idoe)), samples_y)

        ## Calculate log marginal likelihood at specified theta values
        res = 100
        length_scale   = np.logspace(-1,1,res)
        noise_level    = np.logspace(0,3,res)
        theta          = [noise_level,length_scale]
        lml_theta_grid = gpr_model.log_marginal_likelihood(theta)
        np.save(os.path.join(DATA_DIR, fname_train + r'{:d}_lml'.format(idoe)), lml_theta_grid)

        np.save(os.path.join(DATA_DIR, fname_train + r'{:d}_params'.format(idoe)), gpr_model.metamodel_coeffs)


        print('------------------------------------------------------------')
        print('►►► MCS with Surrogate Models')
        print('------------------------------------------------------------')
        np.random.seed(100)
        fname_mcs     = '_'.join([metamodel_class, doe_type])
        fname_mcs_path = os.path.join(DATA_DIR, fname_mcs+r'{:d}'.format(idoe) + '_MCS')
        for r in range(data_test_params[1]):
            _x_samples = dist_x.sample(data_test_params[0], data_test_params[2])[np.newaxis,:]
            # print('x_samples.shape: {}'.format(_x_samples.shape))
            y_pred_mcs    = gpr_model.predict(_x_samples, return_std=True)
            ## y_pred_mcs shape (2, n_samples) (mean, std)
            # print(y_pred_mcs.shape)
            # print('y_pred_mcs shape: {}'.format(y_pred_mcs.shape))
            upload2gdrive(fname_mcs_path+r'{:d}'.format(r),  y_pred_mcs, DATA_DIR_ID)

            print(' ► Calculating ECDF of MCS data and retrieve data to plot...')
            y_pred_mcs_ecdf = get_exceedance_data(y_pred_mcs[0,:], prob_failure=pf)
            rfname_mcs  = fname_mcs_path + '{:d}_ecdf'.format(r) + '.npy' 
            np.save(rfname_mcs, y_pred_mcs_ecdf)



if __name__ == '__main__':
    main()

