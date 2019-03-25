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
import scipy.signal as spsignal
import envi, doe, solver, utilities
import pickle

from envi import environment
from metaModel import metaModel
from simParams import simParameter

from run_sim import run_sim
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel,WhiteKernel)
import os
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


def make_working_directory(MODEL_NAME):
    WORKING_DIR     = os.getcwd()
    MODEL_DIR       = os.path.join(WORKING_DIR, MODEL_NAME)
    MODEL_DIR_DATA  = os.path.join(MODEL_DIR,r'Data')
    MODEL_DIR_FIGURE= os.path.join(MODEL_DIR,r'Figures')
    # Create directory for model  
    try:
        os.makedirs(MODEL_DIR)
        os.makedirs(MODEL_DIR_DATA)
        os.makedirs(MODEL_DIR_FIGURE)
    except FileExistsError:
        # one of the above directories already exists
        pass
    return MODEL_DIR, MODEL_DIR_DATA, MODEL_DIR_FIGURE 


def main():
    
    ## ------------------------------------------------------------------- ###
    ##  Parameters set-up 
    ## ------------------------------------------------------------------- ###
    pf                  = 1e-4              # Exceedence probability
    data_train_params   = [[1e6], 'R']      # nsamples_test, sample_rule
    data_test_params    = [1e7, 10, 'R']    # nsamples_test, nrepeat, sample_rule
    model_def           = ['bench4'] # [solver function name, error_dist.name, [params], size]
    MODEL_DIR, MODEL_DIR_DATA, MODEL_DIR_FIGURE =  make_working_directory(model_def[0])
    ## ------------------------------------------------------------------- ###
    ##  Define Solver parameters ###
    ## ------------------------------------------------------------------- ###
    ## >>> 1. Choose Wiener-Askey scheme random variable
    dist_zeta = cp.Normal(0,1)  # shape=1, scale=1, shift=0

    ## >>> 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## >>> 3. Define independent random variable in physical problems
    dist_x = cp.Normal(5,2) # normal mean = 0, normal std=0.25

    np.random.seed(100)
    ## ------------------------------------------------------------------- ###
    ##  Design of Experiments (DoEs) 
    ## ------------------------------------------------------------------- ###
    ## example:
    ## >>> 1. Fixed point:
    doe_method, doe_rule, doe_order = 'FIX','FIX', 1e3
    doe_params  = [doe_method, doe_rule, doe_order]
    fix_simparam= simParameter(dist_zeta, doe_params=doe_params)
    x_samples   = [np.linspace(-5,15,1000)[np.newaxis, :]]
    fix_simparam.set_doe_samples(x_samples, dist_x)
    zeta_samples = fix_simparam.sys_input_zeta[0]

    ## >>> 2. Monte Carlo:
    # # doe_method, doe_rule, doe_order = 'MC','R', [1e7]*10
    # # doe_params  = [doe_method, doe_rule, doe_order]
    # # mc_simparam = simParameter(dist_zeta, doe_params = doe_params)
    # # mc_simparam.get_doe_samples(dist_x)

    ## >>> 3. Quadrature:
    doe_method, doe_rule, doe_order = 'GQ','hermite',[9]
    doe_params      = [doe_method, doe_rule, doe_order]
    quad_simparam   = simParameter(dist_zeta, doe_params = doe_params)
    quad_simparam.get_doe_samples(dist_x)


    ## ------------------------------------------------------------------- ###
    ##  Run simulation 
    ## ------------------------------------------------------------------- ###
    ## List of simulation results (Different doe_order, sample size would change, so return list)
    sim_output_quad= run_sim(model_def, quad_simparam)
    # sim_output_fix = run_sim(model_def, fix_simparam)
    # sim_output_mc  = run_sim(model_def, mc_simparam)

    train_input  = quad_simparam
    train_output = sim_output_quad[0]

    print(train_output[0].shape)
    ## ------------------------------------------------------------------- ###
    ##  Build surrogate model 
    ## ------------------------------------------------------------------- ###

    for idoe in np.arange(train_input.ndoe):
        zeta_train  = train_input.sys_input_zeta[idoe][0]
        zeta_weight = train_input.sys_input_zeta[idoe][1]
        y_train     = train_output[idoe]
        # ## Build PCE surrogate model
        # metamodel_class, metamodel_basis = 'PCE', [2,3,4]
        # metamodel_params = {'cal_coeffs': 'GQ'}
        # train_data_path  = '_'.join(['DoE', train_input.doe_method, train_input.doe_rule,'train', metamodel_class])

        # pce_model   = metaModel(metamodel_class, metamodel_basis, dist_zeta, **metamodel_params)
        # pce_model.fit_model(zeta_train, y_train, w=zeta_weight)
        # y_validate  = pce_model.predict(zeta_train)
        # train_data  = [ np.array(train_input.sys_input_vars[idoe]),
                        # np.array(train_output[idoe]),
                        # np.array(train_input.sys_input_zeta[idoe]),
                        # np.array(y_validate),
                        # np.array(x_samples),
                        # np.array(zeta_samples),
                        # np.array(pce_model.predict(zeta_samples))]
        # np.save(os.path.join(MODEL_DIR_DATA, train_data_path + r'{:d}'.format(idoe)), train_data)

        # print('------------------------------------------------------------')
        # print('►►► MCS with Surrogate Models')
        # print('------------------------------------------------------------')
        # np.random.seed(100)
        # predict_data_path = os.path.join(MODEL_DIR_DATA, r'metamodels_{:s}_DoE{:d}'.format(metamodel_class,idoe))
        # for r in range(data_test_params[1]):
            # _zeta_samples = dist_zeta.sample(data_test_params[0], data_test_params[2])[np.newaxis,:]
            # print(np.round(_zeta_samples[:5],4))
            # _y_predict    = pce_model.predict(_zeta_samples)
            # pred_file_name= predict_data_path + '_MCS{}'.format(r) + '.npy' 
            # np.save(pred_file_name, np.array([_zeta_samples, _y_predict]))

        ## Build GPR surrogate model
        kernels  = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e4))]
        metamodel_class, metamodel_basis = 'GPR', kernels 
        metamodel_params = {'n_restarts_optimizer':10}
        train_data_path  = '_'.join(['DoE', train_input.doe_method, train_input.doe_rule,'train', metamodel_class])

        gpr_model   = metaModel(metamodel_class, metamodel_basis, dist_zeta, **metamodel_params)
        gpr_model.fit_model(zeta_train, y_train)
        y_validate  = gpr_model.predict(zeta_train, return_std=True)
        # print('y_validate shape: {}'.format(y_validate.shape))
        train_data  = [ np.array(train_input.sys_input_vars[idoe]),
                        np.array(train_output[idoe]),
                        np.array(train_input.sys_input_zeta[idoe]),
                        np.array(y_validate),
                        np.array(x_samples),
                        np.array(zeta_samples),
                        np.array(gpr_model.predict(zeta_samples, return_std=True))]
        np.save(os.path.join(MODEL_DIR_DATA, train_data_path + r'{:d}'.format(idoe)), train_data)

        print('------------------------------------------------------------')
        print('►►► MCS with Surrogate Models')
        print('------------------------------------------------------------')
        np.random.seed(100)
        predict_data_path = os.path.join(MODEL_DIR_DATA, r'metamodels_{:s}_DoE{:d}'.format(metamodel_class,idoe))
        for r in range(data_test_params[1]):
            _zeta_samples = dist_zeta.sample(data_test_params[0], data_test_params[2])[np.newaxis,:]
            print('_zeta_samples.shape: {}'.format(_zeta_samples.shape))
            print(np.round(_zeta_samples[:5],4))
            _y_predict    = gpr_model.predict(_zeta_samples, return_std=True)
            # print('_y_predict shape: {}'.format(_y_predict.shape))
            pred_file_name= predict_data_path + '_MCS{}'.format(r) + '.npy' 
            np.save(pred_file_name, np.array([_zeta_samples, _y_predict]))


if __name__ == '__main__':
    main()

