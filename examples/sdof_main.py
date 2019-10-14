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

    simparams = museuq.simParameters(model_name, dist_zeta, prob_fails=prob_fails)
    # simparams.set_error()  # no observation error for sdof
    simparams.info()

    ## ------------------------ Define DoE parameters ---------------------- ###
    doe_method, doe_rule, doe_orders = 'QUAD', 'hem', [7] * 10
    # doe_method, doe_rule, doe_orders = 'MC', 'R', [1e6]*10 + [1e7]*3 + [1e8]
    sdof_doe    = museuq.DoE(doe_method, doe_rule, doe_orders, dist_zeta)
    print(sdof_doe.filename_tags)

    #  >>> comment below out to skip DoE process 
    samples_zeta= sdof_doe.get_samples()
    samples_x   = [Kvitebjorn.samples(np.array([dist_normal.cdf(isamples_zeta[0,:]), dist_normal.cdf(isamples_zeta[1,:])])) for isamples_zeta in samples_zeta] 
    sdof_doe.set_samples(env=samples_x)
    sdof_doe.info()
    sdof_doe.save_data(simparams.data_dir)
    # sdof_doe.save_data(os.getcwd())
    assert len(samples_x) == len(samples_zeta)

    #### --------------------------------------------------------------------------- ###
    #### -------------------------------- Run Solver ------------------------------- ###
    #### --------------------------------------------------------------------------- ###

    #### >>> option 1: Run solver directly after DoE 

    # sdof_solver   = museuq.Solver(model_name, samples_x)
    # # samples_y     = sdof_solver.run()
    # # filename_tags = [itag+'_y' for itag in sdof_doe.filename_tags]
    # # museuq_dataio.save_data(samples_y, sdof_doe.filename, simparams.data_dir, filename_tags)


    # samples_stats =  sdof_solver.run()
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


    ###------------------------ Define surrogate model parameters ---------------------- ###

    # ## -------------------------------- PCE Surrogate Model -------------------- ###
    # metamodel_params= {'cal_coeffs': 'Galerkin', 'dist_zeta': dist_zeta}
    # metamodel_class = 'PCE'
    # metamodel_basis_setting = doe_orders
    
    # for itag, iquad_order in zip(sdof_doe.filename_tags, metamodel_basis_setting):
        # ### ============ Get training points ============
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}.npy'.format(itag)))
        # train_zeta  = data_set[:2,:]
        # train_w     = data_set[2 ,:]
        # train_x     = data_set[3:5,:]
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_stats.npy'.format(itag)))
        # train_eta   = np.squeeze(data_set[:, 4, 0])
        # train_y     = np.squeeze(data_set[:, 4, 1])

        # ### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for eta: ') 
        # eta_pce_model = museuq.SurrogateModel(metamodel_class, [iquad_order-1], **metamodel_params)
        # eta_pce_model.fit(train_zeta, train_eta, weight=train_w)

        # ### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # # eta_validate, eta_validate_scores = eta_pce_model.predict(train_zeta, train_eta)


        # ### ============ Make prediction at specified points (hs, tp) ============
        # hstp_grid   = np.load(os.path.join(simparams.data_dir, 'HsTp_grid118.npy'))
        # hstp_grid_u = Kvitebjorn.cdf(hstp_grid)
        # hstp_grid_z = np.array([cp.Normal().inv(hstp_grid_u[0,:]), cp.Normal().inv(hstp_grid_u[1,:])])
        # eta_grid = eta_pce_model.predict(hstp_grid_z)

        # ### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for SDOF response: ') 
        # y_pce_model = museuq.SurrogateModel(metamodel_class, [iquad_order-1], **metamodel_params)
        # y_pce_model.fit(train_zeta, train_y, weight=train_w)

        # ### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # # y_validate, y_validate_scores = y_pce_model.predict(train_zeta, train_y)

        # ### ============ Make prediction at specified points (hs, tp) ============
        # y_grid = y_pce_model.predict(hstp_grid_z)

        # ### ============ Save data  ============
        # # data_scores = np.array([eta_validate_scores, y_validate_scores]).T
        # # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{}_{}_scores.npy'.format(itag, metamodel_class)),data_scores)

        # # data_valid = np.array([eta_validate,y_validate]).T
        # data_valid = np.array([eta_grid,y_grid]).T
        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{}_{}_grid.npy'.format(itag, metamodel_class)),data_valid)

        # ## ============ Make prediction with monte carlo samples ============
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
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_{:s}_pred_r{:d}.npy'.format(itag, metamodel_class, r)),data_pred)
            # museuq_helpers.enablePrint()

            # # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            # eta_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(eta_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_pred_r{:d}_ecdf_pf{}_eta.npy'.format(itag,r,str(prob_fails)[-1])),eta_pred_mcs_ecdf)
            # y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_pred_r{:d}_ecdf_pf{}_y.npy'.format(itag,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)



    ## -------------------------------- mPCE Surrogate Model -------------------- ###
    metamodel_params= {'cal_coeffs': 'Galerkin', 'dist_zeta': dist_zeta}
    metamodel_class = 'mPCE'
    metamodel_basis_setting = doe_orders
    
    for iquad_order in np.unique(metamodel_basis_setting):
        ### ============ Get training points ============
        tags        = [i for i in sdof_doe.filename_tags if i.startswith(str(iquad_order))]
        train_zeta  = []
        train_w     = [] 
        train_x     = [] 
        train_eta   = []
        train_y     = [] 

        for itag in tags:
            data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}.npy'.format(itag)))
            train_zeta.append(data_set[:2,:])
            train_w.append(data_set[2 ,:])
            train_x.append(data_set[3:5,:])
            data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_stats.npy'.format(itag)))
            train_eta.append(np.squeeze(data_set[:, 4, 0]))
            train_y.append(np.squeeze(data_set[:, 4, 1]))

        ### ============ Get Surrogate Model for each QoI============
        print('Surrogate Model for eta: ') 
        eta_pce_model = museuq.SurrogateModel(metamodel_class, [iquad_order-1], **metamodel_params)
        eta_pce_model.fit(train_zeta, train_eta, weight=train_w)

        ### ============ Validating surrogate models at training points ============
        print('>>> Validating surrogate model...')
        # eta_validate, eta_validate_scores = eta_pce_model.predict(train_zeta, train_eta)


        ### ============ Make prediction at specified points (hs, tp) ============
        hstp_grid   = np.load(os.path.join(simparams.data_dir, 'HsTp_grid118.npy'))
        hstp_grid_u = Kvitebjorn.cdf(hstp_grid)
        hstp_grid_z = np.array([cp.Normal().inv(hstp_grid_u[0,:]), cp.Normal().inv(hstp_grid_u[1,:])])
        eta_grid = eta_pce_model.predict(hstp_grid_z)

        ### ============ Get Surrogate Model for each QoI============
        print('Surrogate Model for SDOF response: ') 
        y_pce_model = museuq.SurrogateModel(metamodel_class, [iquad_order-1], **metamodel_params)
        y_pce_model.fit(train_zeta, train_y, weight=train_w)

        ### ============ Validating surrogate models at training points ============
        print('>>> Validating surrogate model...')
        # y_validate, y_validate_scores = y_pce_model.predict(train_zeta, train_y)

        ### ============ Make prediction at specified points (hs, tp) ============
        y_grid = y_pce_model.predict(hstp_grid_z)

        ### ============ Save data  ============
        # data_scores = np.array([eta_validate_scores, y_validate_scores]).T
        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{}_{}_scores.npy'.format(itag, metamodel_class)),data_scores)

        # data_valid = np.array([eta_validate,y_validate]).T
        data_valid = np.array([eta_grid,y_grid]).T
        np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{}_{}_grid.npy'.format(itag, metamodel_class)),data_valid)

        ## ============ Make prediction with monte carlo samples ============
        print('>>> Prediction with surrogate models... ') 
        data_test_params= [1e6, 10, 'R'] ##[nsamples, repeat, sampling rule]
        pbar = tqdm(range(data_test_params[1]), ascii=True, desc="   - ")
        for r in pbar:
            museuq_helpers.blockPrint()
            dist_zeta   = eta_pce_model.kwparams['dist_zeta']
            zeta_mcs    = dist_zeta.sample(data_test_params[0], rule=data_test_params[2])
            eta_pred_mcs= eta_pce_model.predict(zeta_mcs)
            y_pred_mcs  = y_pce_model.predict(zeta_mcs)
            data_pred   = np.array([eta_pred_mcs, y_pred_mcs])
            np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_{:s}_pred_r{:d}.npy'.format(itag, metamodel_class, r)),data_pred)
            museuq_helpers.enablePrint()

            # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            eta_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(eta_pred_mcs), prob=simparams.prob_fails)
            np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_pred_r{:d}_ecdf_pf{}_eta.npy'.format(itag,r,str(prob_fails)[-1])),eta_pred_mcs_ecdf)
            y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:s}_pred_r{:d}_ecdf_pf{}_y.npy'.format(itag,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)







    ### -------------------------------- GPR Surrogate Model -------------------- ###
    # metamodel_params= {'n_restarts_optimizer': 10}

    # metamodel_class = 'GPR'
    # metamodel_basis_setting = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e4)) + 
            # WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)) ]
    
    # for iquad_order in doe_orders:
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}.npy'.format(iquad_order)))
        # train_zeta  = data_set[:2,:]
        # train_w     = data_set[2,:]
        # train_x     = data_set[3:5,:]
        # data_set    = np.load(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_stats.npy'.format(iquad_order)))
        # train_eta   = np.squeeze(data_set[:, 4, 1])
        # train_y     = np.squeeze(data_set[:, 4, 2])

        # ### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for eta: ') 
        # eta_gpr_model = museuq.SurrogateModel(metamodel_class, metamodel_basis_setting, **metamodel_params)
        # eta_gpr_model.fit(train_x, train_eta)

        # #### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # eta_validate, eta_validate_scores = eta_gpr_model.predict(train_x, train_eta)

        # ### ============ Make prediction at specified points (hs, tp) ============
        # hstp_grid   = np.load(os.path.join(simparams.data_dir, 'HsTp_grid118.npy'))
        # eta_grid = eta_gpr_model.predict(hstp_grid)




        # #### ============ Get Surrogate Model for each QoI============
        # print('Surrogate Model for SDOF response: ') 
        # y_gpr_model = museuq.SurrogateModel(metamodel_class, metamodel_basis_setting, **metamodel_params)
        # y_gpr_model.fit(train_x, train_y)


        # #### ============ Validating surrogate models at training points ============
        # print('>>> Validating surrogate model...')
        # y_validate, y_validate_scores = y_gpr_model.predict(train_x, train_y)

        # ### ============ Make prediction at specified points (hs, tp) ============
        # y_grid = y_gpr_model.predict(hstp_grid)


        # #### ============ Save data  ============
        # # data_valid = np.array([eta_validate,y_validate]).T
        # # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_valid.npy'.format(iquad_order, metamodel_class)),data_valid)

        # # data_scores = np.array([eta_validate_scores, y_validate_scores]).T
        # # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_scores.npy'.format(iquad_order, metamodel_class)),data_scores)

        # data_valid = np.array([eta_grid,y_grid]).T
        # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{}_{}_grid.npy'.format(iquad_order, metamodel_class)),data_valid)



        # print('>>> Prediction with surrogate models... ') 
        # data_test_params= [1e7, 10, 'R'] ##[nsamples, repeat, sampling rule]
        # pbar = tqdm(range(data_test_params[1]), ascii=True, desc="   - ")
        # for r in pbar:
            # museuq_helpers.blockPrint()
            # zeta_mcs    = dist_zeta.sample(data_test_params[0], rule=data_test_params[2])
            # x_mcs       = Kvitebjorn.samples(np.array([dist_normal.cdf(zeta_mcs[0,:]), dist_normal.cdf(zeta_mcs[1,:])]))
            # eta_pred_mcs= eta_gpr_model.predict(x_mcs)
            # y_pred_mcs  =   y_gpr_model.predict(x_mcs)
            # data_pred   = np.array([eta_pred_mcs, y_pred_mcs])
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_pred_r{:d}.npy'.format(iquad_order,metamodel_class, r)),data_pred)
            # museuq_helpers.enablePrint()

            # # museuq_helpers.upload2gdrive(fname_test_path+r'{:d}'.format(r),  y_pred_mcs, simparam.data_dir_id)
            # # print(' > Calculating ECDF of MCS data and retrieve data to plot...')
            # eta_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(eta_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_pred_r{:d}_ecdf_pf{}_eta.npy'.format(iquad_order,metamodel_class,r,str(prob_fails)[-1])),eta_pred_mcs_ecdf)
            # y_pred_mcs_ecdf = museuq_helpers.get_exceedance_data(np.array(y_pred_mcs), prob=simparams.prob_fails)
            # np.save(os.path.join(simparams.data_dir, 'DoE_QuadHem{:d}_{}x_pred_r{:d}_ecdf_pf{}_y.npy'.format(iquad_order,metamodel_class,r,str(prob_fails)[-1])),y_pred_mcs_ecdf)
            # rfname_mcs  = fname_test_path + '{:d}_ecdf'.format(r) 
            # # np.save(rfname_mcs, y_pred_mcs_ecdf)








if __name__ == '__main__':
    main()
