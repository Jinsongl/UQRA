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
    model_name  = 'BENCH4'
    ## 1. Choose Wiener-Askey scheme random variable
    # dist_zeta = cp.Uniform(-1,1)
    # dist_zeta = cp.Gamma(4,1)
    dist_normal = cp.Normal()
    dist_zeta   = cp.Normal()
    dist_x      = cp.Normal(5,5)  ## (mu, sigma)

    ## 2. If transformation needed, like Rosenblatt, need to be done here
    ## Perform Rosenblatt etc

    ## 3. Define independent random variable in physical problems
    # dist_x = cp.Uniform(-np.pi, np.pi)

    simparams = museuq.simParameters(model_name, dist_zeta, prob_fails=prob_fails)
    # simparams.set_error()
    # simparams.set_error('normal',loc=0, scale=50)
    simparams.set_error('normal',cov=0.15)
    simparams.info()

    ## ------------------------ Define DoE parameters ---------------------- ###
    # doe_method, doe_rule, doe_orders = 'QUAD', 'hem', [5]
    doe_method, doe_rule, doe_orders = 'MC', 'R', [1e6]*10
    bench4_doe    = museuq.DoE(doe_method, doe_rule, doe_orders, dist_zeta)

    #  >>> comment below out to skip DoE process 
    samples_zeta= bench4_doe.get_samples()
    samples_x = bench4_doe.mappingto(dist_x)

    # for isamples_zeta in samples_zeta:
        # zeta_cord, zeta_weight = isamples_zeta[:-1,:], isamples_zeta[-1,:]
        # x_cord   = dist_x.inv(dist_zeta.cdf(zeta_cord)) 
        # samples_x.append(np.array([x_cord, zeta_weight]))

    # bench4_doe.set_samples(env=samples_x)
    bench4_doe.info()
    bench4_doe.save_data(simparams.data_dir)
    # bench4_doe.save_data(os.getcwd())
    assert len(samples_x) == len(samples_zeta)

    #### --------------------------------------------------------------------------- ###
    #### -------------------------------- Run Solver ------------------------------- ###
    #### --------------------------------------------------------------------------- ###

    #### >>> option 1: Run solver directly after DoE 

    # sdof_solver   = museuq.Solver(model_name, samples_x)
    sdof_solver   = museuq.Solver(model_name, samples_x, error=simparams.error)
    samples_y     = sdof_solver.run()
    # filename_tags = [itag+'_y' for itag in bench4_doe.filename_tags]
    filename_tags = [itag+'_y_{:s}'.format(simparams.error.name) for itag in bench4_doe.filename_tags]
    museuq_dataio.save_data(samples_y, bench4_doe.filename, simparams.data_dir, filename_tags)
    # samples_stats = sdof_solver.get_stats()
    # filename_tags = [itag+'_stats' for itag in bench4_doe.filename_tags]
    # museuq_dataio.save_data(samples_stats, bench4_doe.filename, simparams.data_dir, filename_tags)

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
        # # filename_tags = [itag+'_y' for itag in bench4_doe.filename_tags]
        # # print(bench4_doe.filename_tags)
        # filename_tags = [bench4_doe.filename_tags[r] + '_y_stats']
        # print(simparams.data_dir)
        # museuq_dataio.save_data(samples_y, bench4_doe.filename, simparams.data_dir, filename_tags)
        # # museuq_dataio.save_data(samples_y, bench4_doe.filename, os.getcwd(), filename_tags)
        # # samples_y_stats = sdof_solver.get_stats(simparams.qoi2analysis, simparams.stats)
        # # filename_tags = [bench4_doe.filename_tags[r] + '_y_stats']
        # # museuq_dataio.save_data(samples_y_stats, bench4_doe.filename, simparams.data_dir, filename_tags)


    # ###------------------------ Define surrogate model parameters ---------------------- ###


    # # -------------------------------- PCE Surrogate Model -------------------- ###
    # metamodel_params= {'cal_coeffs': 'Galerkin', 'dist_zeta': dist_zeta}
    # metamodel_class = 'PCE'
    # metamodel_basis_setting = doe_orders
    
    # for iquad_order in metamodel_basis_setting:
        # ### ============ Get training points ============
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}.npy'.format(iquad_order)))
        # train_zeta  = data_set[0,:].reshape(1,-1)
        # train_w     = np.squeeze(data_set[1,:])
        # train_x     = data_set[2,:].reshape(1,-1)
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_y_{:s}.npy'.format(iquad_order,simparams.error.name)))
        # train_y     = np.squeeze(data_set)

        # ### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for {:s}: '.format(model_name.upper())) 
        # y_pce_model = museuq.SurrogateModel(metamodel_class, [iquad_order-1], **metamodel_params)
        # y_pce_model.fit(train_zeta, train_y, weight=train_w)

        # ### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # y_valid, y_valid_scores = y_pce_model.predict(train_zeta, train_y)



        # ### ============ Make prediction at specified points (hs, tp) ============
        # # y_grid = y_pce_model.predict(hstp_grid_z)
        # zeta_grid   = np.linspace(-3,3,600)
        # x_grid      = zeta_grid * 5 + 5
        # y_grid      = y_pce_model.predict(zeta_grid.reshape(1,-1))
        # res_grid    = np.array([zeta_grid, x_grid, y_grid])


        # print(train_y)
        # print(y_valid)
        # ### ============ Save data  ============
        # # data_scores = np.array([eta_validate_scores, y_valid_scores]).T
        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_score.npy'.format(iquad_order, metamodel_class, simparams.error.name)),y_valid_scores)
        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_valid.npy'.format(iquad_order, metamodel_class, simparams.error.name)),y_valid)
        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_grid.npy'.format(iquad_order, metamodel_class, simparams.error.name)),res_grid)

        # # data_valid = np.array([eta_validate,y_valid]).T
        # # data_valid = np.array([eta_grid,y_grid]).T
        # # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{}_{}_grid.npy'.format(iquad_order, metamodel_class)),data_valid)


        # # ### ============ Make prediction with monte carlo samples ============
        # print('>>> Prediction with surrogate models... ') 
        # data_test_params= [1e7, 10, 'R'] ##[nsamples, repeat, sampling rule]
        # pbar = tqdm(range(data_test_params[1]), ascii=True, desc="   - ")
        # for r in pbar:
            # museuq_helpers.blockPrint()
            # dist_zeta   = y_pce_model.kwparams['dist_zeta']
            # zeta_mcs    = dist_zeta.sample(data_test_params[0], rule=data_test_params[2]).reshape(1,-1)
            # y_pred_mcs  = y_pce_model.predict(zeta_mcs)
            # data_pred   = np.array(y_pred_mcs)

            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_pred_r{:d}.npy'.format(iquad_order, metamodel_class, simparams.error.name, r)),data_pred)
            # museuq_helpers.enablePrint()

            # # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            # # eta_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(eta_pred_mcs), prob=simparams.prob_fails)
            # # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_pred_r{:d}_ecdf_pf{}_eta.npy'.format(iquad_order,r,str(prob_fails)[-1])),eta_pred_mcs_ecdf)
            # y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_pred_r{:d}_ecdf_pf{:s}_y.npy'.format(iquad_order, metamodel_class, simparams.error.name,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)
            # # rfname_mcs  = fname_test_path + '{:d}_ecdf'.format(r) 
            # # np.save(rfname_mcs, y_pred_mcs_ecdf)







    # # ### -------------------------------- GPR Surrogate Model -------------------- ###
    # metamodel_params= {'n_restarts_optimizer': 10}

    # metamodel_class = 'GPR'
    # metamodel_basis_setting = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e4)) + 
            # WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)) ]
    
    # for iquad_order in doe_orders:
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}.npy'.format(iquad_order)))
        # train_zeta  = data_set[0,:].reshape(1,-1)
        # train_w     = np.squeeze(data_set[1,:])
        # train_x     = data_set[2,:].reshape(1,-1)
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_y_{:s}.npy'.format(iquad_order,simparams.error.name)))
        # train_y     = np.squeeze(data_set)

        # #### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for SDOF response: ') 
        # y_gpr_model = museuq.SurrogateModel(metamodel_class, metamodel_basis_setting, **metamodel_params)
        # y_gpr_model.fit(train_x, train_y)


        # #### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # y_valid, y_valid_scores = y_gpr_model.predict(train_x, train_y)

        # ### ============ Make prediction at specified points (hs, tp) ============
        # zeta_grid   = np.linspace(-3,3,600)
        # x_grid      = zeta_grid * 5 + 5
        # y_grid      = y_gpr_model.predict(x_grid.reshape(1,-1))
        # res_grid    = np.array([zeta_grid, x_grid, y_grid])


        # ### ============ Save data  ============

        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_score.npy'.format(iquad_order, metamodel_class, simparams.error.name)),y_valid_scores)
        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_valid.npy'.format(iquad_order, metamodel_class, simparams.error.name)),y_valid)
        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_grid.npy'.format(iquad_order, metamodel_class, simparams.error.name)),res_grid)

        # print('>>> Prediction with surrogate models... ') 
        # data_test_params= [1e7, 10, 'R'] ##[nsamples, repeat, sampling rule]
        # pbar = tqdm(range(data_test_params[1]), ascii=True, desc="   - ")
        # for r in pbar:
            # museuq_helpers.blockPrint()
            # # dist_zeta   = y_gpr_model.kwparams['dist_zeta']
            # zeta_mcs    = dist_zeta.sample(data_test_params[0], rule=data_test_params[2]).reshape(1,-1)
            # x_mcs       = zeta_mcs * 5 + 5
            # y_pred_mcs  = y_gpr_model.predict(x_mcs)
            # data_pred   = np.array(y_pred_mcs)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_pred_r{:d}.npy'.format(iquad_order, metamodel_class, simparams.error.name, r)),data_pred)
            # museuq_helpers.enablePrint()

            # # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            # y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{:s}_{:s}_pred_r{:d}_ecdf_pf{:s}_y.npy'.format(iquad_order, metamodel_class, simparams.error.name,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)
            # # rfname_mcs  = fname_test_path + '{:d}_ecdf'.format(r) 
            # # np.save(rfname_mcs, y_pred_mcs_ecdf)


if __name__ == '__main__':
    main()
