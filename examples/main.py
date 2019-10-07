#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""

import context
import museuq
import numpy as np, chaospy as cp, os, sys
import warnings
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel,WhiteKernel)
from tqdm import tqdm
from museuq.utilities import helpers as museuq_helpers 
from museuq.utilities import metrics_collections as museuq_metrics
from museuq.utilities import dataIO as museuq_dataio 
from museuq.environment import Kvitebjorn
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

def main():
    ## ------------------------ Parameters set-up ----------------------- ###
    prob_fails  = 1e-5              # failure probabilities
    model_name  = 'linear_oscillator'
    ## 1. Choose Wiener-Askey scheme random variable
    # dist_zeta = cp.Uniform(-1,1)
    # dist_zeta = cp.Gamma(4,1)
    dist_normal = cp.Normal()
    dist_zeta = cp.Iid(cp.Normal(),2) 

    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## 3. Define independent random variable in physical problems
    # dist_x = cp.Uniform(-np.pi, np.pi)

    error_params=None
    simparams = museuq.simParameters(model_name, dist_zeta, prob_fails=prob_fails)
    simparams.update_dir()
    simparams.set_error(error_params)
    simparams.disp()

    ## ------------------------ Define DoE parameters ---------------------- ###
    doe_method, doe_rule, doe_orders = 'QUAD', 'hem', [10]
    # doe_method, doe_rule, doe_orders = 'MC', 'R', [1e6]*10
    sdof_doe    = museuq.DoE(doe_method, doe_rule, doe_orders, dist_zeta)

    #  >>> comment below out to skip DoE process 
    # samples_zeta= sdof_doe.get_samples()
    # samples_x   = [Kvitebjorn.samples(np.array([dist_normal.cdf(isamples_zeta[0,:]), dist_normal.cdf(isamples_zeta[1,:])])) for isamples_zeta in samples_zeta] 
    # sdof_doe.set_samples(env=samples_x)
    # sdof_doe.disp()
    # sdof_doe.save_data(simparams.data_dir)
    # # sdof_doe.save_data(os.getcwd())
    # assert len(samples_x) == len(samples_zeta)

    #### --------------------------------------------------------------------------- ###
    #### -------------------------------- Run Solver ------------------------------- ###
    #### --------------------------------------------------------------------------- ###

    #### >>> option 1: Run solver directly after DoE 

    # sdof_solver   = museuq.Solver(model_name, samples_x)
    # samples_y     = sdof_solver.run(doe_method=doe_method, return_all=True)
    # filename_tags = [itag+'_y' for itag in sdof_doe.filename_tags]
    # museuq_dataio.save_data(samples_y, sdof_doe.filename, simparams.data_dir, filename_tags)
    # samples_stats = sdof_solver.get_stats()
    # filename_tags = [itag+'_stats' for itag in sdof_doe.filename_tags]
    # museuq_dataio.save_data(samples_stats, sdof_doe.filename, simparams.data_dir, filename_tags)

    #### >>> option 2: Run solver with samples from data files

    # repeat = range(0,10)
    # for r in repeat:
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_McRE5R{:d}.npy'.format(r)))
        # # data_set    = np.load(os.path.join(os.getcwd(), 'DoE_McRE6R{:d}.npy'.format(r)))
        # samples_zeta= data_set[:2,:]
        # samples_x   = data_set[2:,:]

        # sdof_solver   = museuq.Solver(model_name, samples_x)
        # samples_y     = sdof_solver.run(doe_method=doe_method)
        # # print(len(samples_y))
        # # filename_tags = [itag+'_y' for itag in sdof_doe.filename_tags]
        # # print(sdof_doe.filename_tags)
        # filename_tags = [sdof_doe.filename_tags[r] + '_y_stats']
        # print(simparams.data_dir)
        # museuq_dataio.save_data(samples_y, sdof_doe.filename, simparams.data_dir, filename_tags)
        # # museuq_dataio.save_data(samples_y, sdof_doe.filename, os.getcwd(), filename_tags)
        # # samples_y_stats = sdof_solver.get_stats(simparams.qoi2analysis, simparams.stats)
        # # filename_tags = [sdof_doe.filename_tags[r] + '_y_stats']
        # # museuq_dataio.save_data(samples_y_stats, sdof_doe.filename, simparams.data_dir, filename_tags)


    # ------------------------ Define surrogate model parameters ---------------------- ###
    ### --------------------------------------------------------------------------- ###
    ### -------------------------------- Build Surrogate Model -------------------- ###
    ### --------------------------------------------------------------------------- ###

    # metamodel_params= {'cal_coeffs': 'Galerkin', 'dist_zeta': dist_zeta}
    # metamodel_class = 'PCE'
    # # metamodel_basis_setting = [4,5,6,7,8,9]
    
    # for iquad_order in doe_orders:
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}.npy'.format(iquad_order)))
        # train_zeta  = data_set[:2,:]
        # train_w     = data_set[2,:]
        # train_x     = data_set[3:5,:]
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_stats.npy'.format(iquad_order)))
        # train_eta   = np.squeeze(data_set[:, 4, 1])
        # train_y     = np.squeeze(data_set[:, 4, 2])

        # # print('==============================================================')
        # print('Surrogate Model for eta: ') 
        # # print('==============================================================')
        # eta_pce_model = museuq.SurrogateModel(metamodel_class, [iquad_order-1], **metamodel_params)
        # eta_pce_model.fit(train_zeta, train_eta, weight=train_w)
        # print('>>> Validating surrogate model...')
        # eta_validate, eta_validate_scores = eta_pce_model.predict(train_zeta, train_eta)

        # # print('==============================================================')
        # print('Surrogate Model for SDOF response: ') 
        # # print('==============================================================')
        # y_pce_model = museuq.SurrogateModel(metamodel_class, [iquad_order-1], **metamodel_params)
        # y_pce_model.fit(train_zeta, train_y, weight=train_w)
        # print('>>> Validating surrogate model...')
        # y_validate, y_validate_scores = y_pce_model.predict(train_zeta, train_y)


        # data_valid = np.array([eta_validate,y_validate]).T
        # data_scores = np.array([eta_validate_scores, y_validate_scores]).T

        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{}_{}_valid.npy'.format(iquad_order, metamodel_class)),data_valid)
        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{}_{}_scores.npy'.format(iquad_order, metamodel_class)),data_scores)

        # print('>>> Prediction with surrogate models... ') 

        # data_test_params= [1e7, 10, 'R'] ##[nsamples, repeat, sampling rule]
        # pbar = tqdm(range(data_test_params[1]), ascii=True, desc="   - ")
        # for r in pbar:
            # museuq_helpers.blockPrint()
            # dist_zeta   = eta_pce_model.kwparams['dist_zeta']
            # zeta_mcs    = dist_zeta.sample(data_test_params[0], rule=data_test_params[2])
            # eta_pred_mcs= eta_pce_model.predict(zeta_mcs)
            # y_pred_mcs  = y_pce_model.predict(zeta_mcs)
            # data_pred   = np.array([eta_pred_mcs, y_pred_mcs])
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_pred_r{:d}.npy'.format(iquad_order, r)),data_pred)
            # museuq_helpers.enablePrint()

            # # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            # eta_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(eta_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_pred_r{:d}_ecdf_pf{}_eta.npy'.format(iquad_order,r,str(prob_fails)[-1])),eta_pred_mcs_ecdf)
            # y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_pred_r{:d}_ecdf_pf{}_y.npy'.format(iquad_order,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)
            # rfname_mcs  = fname_test_path + '{:d}_ecdf'.format(r) 
            # # np.save(rfname_mcs, y_pred_mcs_ecdf)


    metamodel_params= {'n_restarts_optimizer': 10}

    metamodel_class = 'GPR'
    metamodel_basis_setting = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e4)) + 
            WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)) ]
    
    for iquad_order in doe_orders:
        data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}.npy'.format(iquad_order)))
        train_zeta  = data_set[:2,:]
        train_w     = data_set[2,:]
        train_x     = data_set[3:5,:]
        data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_stats.npy'.format(iquad_order)))
        train_eta   = np.squeeze(data_set[:, 4, 1])
        train_y     = np.squeeze(data_set[:, 4, 2])

        # print('==============================================================')
        print('Surrogate Model for eta: ') 
        # print('==============================================================')
        eta_gpr_model = museuq.SurrogateModel(metamodel_class, metamodel_basis_setting, **metamodel_params)
        eta_gpr_model.fit(train_x, train_eta)
        print('>>> Validating surrogate model...')
        eta_validate, eta_validate_scores = eta_gpr_model.predict(train_x, train_eta)

        # print('==============================================================')
        print('Surrogate Model for SDOF response: ') 
        # print('==============================================================')
        y_gpr_model = museuq.SurrogateModel(metamodel_class, metamodel_basis_setting, **metamodel_params)
        y_gpr_model.fit(train_x, train_y)
        print('>>> Validating surrogate model...')
        y_validate, y_validate_scores = y_gpr_model.predict(train_x, train_y)


        data_valid = np.array([eta_validate,y_validate]).T
        data_scores = np.array([eta_validate_scores, y_validate_scores]).T

        np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_valid.npy'.format(iquad_order, metamodel_class)),data_valid)
        np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_scores.npy'.format(iquad_order, metamodel_class)),data_scores)

        print('>>> Prediction with surrogate models... ') 

        data_test_params= [1e7, 10, 'R'] ##[nsamples, repeat, sampling rule]
        pbar = tqdm(range(data_test_params[1]), ascii=True, desc="   - ")
        for r in pbar:
            museuq_helpers.blockPrint()
            zeta_mcs    = dist_zeta.sample(data_test_params[0], rule=data_test_params[2])
            x_mcs       = Kvitebjorn.samples(np.array([dist_normal.cdf(zeta_mcs[0,:]), dist_normal.cdf(zeta_mcs[1,:])]))
            eta_pred_mcs= eta_gpr_model.predict(x_mcs)
            y_pred_mcs  =   y_gpr_model.predict(x_mcs)
            data_pred   = np.array([eta_pred_mcs, y_pred_mcs])
            np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_pred_r{:d}.npy'.format(iquad_order,metamodel_class, r)),data_pred)
            museuq_helpers.enablePrint()

            # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            eta_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(eta_pred_mcs), prob=simparams.prob_fails)
            np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_pred_r{:d}_ecdf_pf{}_eta.npy'.format(iquad_order,metamodel_class,r,str(prob_fails)[-1])),eta_pred_mcs_ecdf)
            y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_pred_r{:d}_ecdf_pf{}_y.npy'.format(iquad_order,metamodel_class,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)
            # rfname_mcs  = fname_test_path + '{:d}_ecdf'.format(r) 
            # # np.save(rfname_mcs, y_pred_mcs_ecdf)








if __name__ == '__main__':
    main()
