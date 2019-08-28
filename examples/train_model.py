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
from sim_setup import sim_setup
from metaModel import metaModel
from simParams import simParameter
from run_sim import run_sim
from utilities import upload2gdrive, get_exceedance_data,make_output_dir, get_gdrive_folder_id 
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel,WhiteKernel)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def main(simparam):
    

    ## ------------------------------------------------------------------- ###
    ##  Build surrogate model 
    ## ------------------------------------------------------------------- ###

    noise_dir   = 'DATA_NOISE_NORMAL'
    ndoe2train  = [11,12,13,14,15] 
    data_file   = r'Train_Free_MCS1_DoE1_pf3_ecdf.npy'
    data_set    = np.load(os.path.join(simparam.data_dir, data_file))
    x_samples   = data_set[0]
    y_samples   = data_set[1]
    zeta_samples= data_set[2]

    for idoe in ndoe2train:

        ## ******************************************************************
        ## Build PCE surrogate model
        ## ******************************************************************

        doe_type    = 'Quadrature_DoE'
        fname_train_in = '_'.join(['Train',simparam.error.name, doe_type+'{}.npy'.format(idoe)])
        train_data = np.load(os.path.join(simparam.data_dir,noise_dir, fname_train_in))
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
        fname_train_out = '_'.join([metamodel_class, 'train_validation_test', doe_type])

        pce_model   = metaModel(metamodel_class, metamodel_basis, dist_zeta, **metamodel_params)
        pce_model.fit_model(zeta_train[np.newaxis,:], y_train, weight=zeta_weight)
        y_validate  = pce_model.predict(zeta_train)
        train_data  = [ x_train, x_weight , y_train, zeta_train,
                        np.array(y_validate),
                        np.array(x_samples),
                        np.array(zeta_samples),
                        np.array(pce_model.predict(zeta_samples))]
        np.save(os.path.join(simparam.data_dir, fname_train_out + r'{:d}'.format(idoe)), train_data)

        print('------------------------------------------------------------')
        print('►►► MCS with Surrogate Models')
        print('------------------------------------------------------------')
        np.random.seed(100)
        fname_mcs = '_'.join([metamodel_class, doe_type+str(idoe), 'MCS'])
        fname_mcs_path = os.path.join(simparam.data_dir, fname_mcs)

        for r in range(data_test_params[1]):
            zeta_mcs        = dist_zeta.sample(data_test_params[0], data_test_params[2])[np.newaxis,:]
            y_pred_mcs      = pce_model.predict(zeta_mcs)

            upload2gdrive(fname_mcs_path+r'{:d}'.format(r),  y_pred_mcs, DATA_DIR_ID)
            print(' ► Calculating ECDF of MCS data and retrieve data to plot...')
            y_pred_mcs_ecdf = get_exceedance_data(y_pred_mcs, prob_failure=pf)
            rfname_mcs  = fname_mcs_path + '{:d}_ecdf'.format(r) + '.npy' 
            np.save(rfname_mcs, y_pred_mcs_ecdf)

        # ******************************************************************
        # >>> Build GPR surrogate model
        # ******************************************************************

        doe_type    = 'Uniform_DoE'
        fname_train_in = '_'.join(['Train',simparam.error.name, doe_type+'{}.npy'.format(idoe)])
        train_data = np.load(os.path.join(simparam.data_dir,noise_dir,fname_train_in))
        x_train    = np.squeeze(train_data[0])
        y_train    = np.squeeze(train_data[1])
        zeta_train = np.squeeze(train_data[2])

        kernels  = [20.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10))]
        metamodel_class, metamodel_basis = 'GPR', kernels 
        metamodel_params = {'n_restarts_optimizer':10}
        fname_train_out  = '_'.join([metamodel_class, 'train_validation_test', doe_type])

        gpr_model   = metaModel(metamodel_class, metamodel_basis, dist_zeta, **metamodel_params)
        gpr_model.fit_model(x_train[np.newaxis,:], y_train)
        y_validate  = gpr_model.predict(x_train[np.newaxis,:], return_std=True)
        train_data  = [ x_train, y_train, zeta_train,
                        np.array(y_validate),
                        np.array(x_samples),
                        np.array(zeta_samples),
                        np.array(gpr_model.predict(x_samples[np.newaxis,:], return_std=True))]
        np.save(os.path.join(simparam.data_dir, fname_train_out + r'{:d}'.format(idoe)), train_data)

        ## samples_y
        samples_x = np.sort(np.hstack((x_train, np.linspace(-5,20,1000))))
        samples_y = gpr_model.sample_y(samples_x[np.newaxis,:], n_samples=10)
        np.save(os.path.join(simparam.data_dir, fname_train_out + r'{:d}_sample_y'.format(idoe)), samples_y)

        ## Calculate log marginal likelihood at specified theta values
        res = 100
        length_scale   = np.logspace(-1,1,res)
        noise_level    = np.logspace(0,3,res)
        theta          = [noise_level,length_scale]
        lml_theta_grid = gpr_model.log_marginal_likelihood(theta)
        np.save(os.path.join(simparam.data_dir, fname_train_out + r'{:d}_lml'.format(idoe)), lml_theta_grid)
        np.save(os.path.join(simparam.data_dir, fname_train_out + r'{:d}_params'.format(idoe)), gpr_model.metamodel_coeffs)


        print('------------------------------------------------------------')
        print('►►► MCS with Surrogate Models')
        print('------------------------------------------------------------')
        np.random.seed(100)
        fname_mcs     = '_'.join([metamodel_class, doe_type])
        fname_mcs_path = os.path.join(simparam.data_dir, fname_mcs+r'{:d}'.format(idoe) + '_MCS')
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
    simparam = sim_setup()
    main(simparam)

